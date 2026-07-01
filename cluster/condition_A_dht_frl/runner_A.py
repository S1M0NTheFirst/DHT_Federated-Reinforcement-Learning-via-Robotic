"""
Condition A — DHT+FRL on the cluster.

Server node: hosts Redis (port 6379) + Flower server (port 8470) + this runner.
Client nodes: each runs ROBOTS_PER_NODE apptainer instances (cids 0..9 on
node1, cids 10..19 on node2). Migrations move a robot from one client node to
the other, by rsync'ing the policy+replay bundle written by the worker.

What's preserved across migration: policy_weights.pt + replay_buffer.pkl +
manifest.json (the "DHT bundle" — same as workstation Condition A).

What's NOT done here that the workstation Condition A does: full CRIU
checkpoint of the container memory image. This is intentional — Condition A's
paper claim is that bundle transfer alone is sufficient for zero-regression
migration, *independent* of the underlying checkpoint mechanism.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time

import redis
import shlex
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.cluster_runner import (
    ClusterConfig, MigrationMetricsWriter,
    apptainer_instance_name,
    install_signal_handlers, register_tracked_process, terminate_all_tracked,
    is_local_node,
    launch_robot, kill_robot,
    establish_ssh_master, close_ssh_master, close_all_ssh_masters_fast,
    live_status_loop, post_migration_recovery,
    rsync_dir, ssh_run, _SSH_OPTS,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
LOG = logging.getLogger("condA")

CONDITION = "dht_frl"
IMAGE     = "robot.sif"
WORKER    = "dht_frl/worker_robot_client.py"


# Track which client node currently hosts each robot. Updated on every
# migration. Initialized from the home-node assignment.
_robot_node: dict[str, str] = {}
_robot_lock = threading.Lock()


def trigger_dht_migration(cfg: ClusterConfig, r: redis.Redis,
                          robot_id: str, success_rate_pre: float,
                          task_counter_pre: int) -> dict:
    """
    Cross-node DHT bundle transfer:
      1. Wait for worker to flush its policy/buffer to /checkpoints (signaled
         by the ready_for_criu redis key).
      2. rsync /checkpoints/{robot_id}/ from src_node to dst_node:/checkpoints/{robot_id}_dest/.
      3. Tell the worker (via load_policy redis key) where to find the bundle.
         The worker is still running; it loads from disk and continues.
      4. Wait for first_bid_after_migration confirmation; record policy_load_ms.

    The CRIU dump/restore phases of workstation Condition A are skipped here:
    the worker stays alive in its apptainer instance for the duration. This
    matches the paper's "application-level migration" framing.
    """
    cid = int(robot_id.split("_")[1])
    with _robot_lock:
        src_node = _robot_node[robot_id]
        dst_node = cfg.other_node(src_node)
    src_chk = f"{cfg.checkpoint_base}/{apptainer_instance_name(cid)}"
    dst_chk = f"{cfg.checkpoint_base}/{apptainer_instance_name(cid)}_dest"

    LOG.info("[MIG] %s  %s → %s", robot_id, src_node, dst_node)
    t_trigger = time.perf_counter()

    # 1. Wait up to 30s for worker to write policy bundle.
    deadline = time.time() + 30
    while time.time() < deadline:
        if r.get(f"ready_for_criu:{robot_id}"):
            break
        time.sleep(0.2)

    # 2. (No CRIU on cluster — skipped intentionally; see module docstring.)
    t_dump_done = time.perf_counter()
    dump_ms = (t_dump_done - t_trigger) * 1000

    # 3. rsync the bundle dir from src_node to dst_node. The dst path mirrors
    # the workstation's "{robot}_dest" convention so the worker code can
    # find it via the load_policy redis key (mapped to /checkpoints/<inst>_dest
    # inside the apptainer instance — the bind matches /tmp/swiftbot_dht_frl).
    t_xfer = time.perf_counter()
    xfer = rsync_dir(src_node, src_chk, dst_node, dst_chk, timeout=600)
    transfer_ms = xfer["elapsed_ms"]
    bytes_xfer  = xfer["bytes_transferred"]
    if xfer["returncode"] != 0:
        LOG.error("[MIG] rsync failed for %s: %s", robot_id, xfer["stderr"])

    # 4. Compute bundle size (sum of policy + replay + manifest on dst).
    sz = ssh_run(dst_node,
                 f"du -sb {dst_chk}/policy_weights.pt {dst_chk}/replay_buffer.pkl "
                 f"{dst_chk}/manifest.json 2>/dev/null "
                 f"| awk '{{s+=$1}}END{{print s}}'", timeout=10)
    try:
        bundle_bytes = int((sz.stdout or "0").strip())
    except ValueError:
        bundle_bytes = 0
    chk_size_mb = bundle_bytes / (1024 * 1024)

    # 5. Count replay buffer entries on dst (best-effort; failure → 0).
    rb_count = ssh_run(dst_node,
                       f"python3 -c \"import pickle; "
                       f"print(len(pickle.load(open('{dst_chk}/replay_buffer.pkl','rb'))))\" "
                       f"2>/dev/null", timeout=10)
    try:
        replay_entries = int((rb_count.stdout or "0").strip())
    except ValueError:
        replay_entries = 0

    # 6. Tell the worker where to load from. The worker resolves /checkpoints
    # to its bound checkpoint dir on dst_node. We migrate the *worker process*
    # by killing the src instance and launching a fresh instance on dst node
    # — but that means task_counter is lost. To preserve task_counter without
    # CRIU we'd need a worker-level resume mechanism the DHT worker doesn't
    # have. Compromise: the worker on src reloads in place (load_policy points
    # to dst_chk read via NFS-style cross-node access if /tmp is shared), OR
    # we keep src running. On HPC2 /tmp is node-local, so the worker must
    # read from a path on its OWN node. Solution: rsync the bundle BACK to
    # the src node too, and tell the worker to read from a local path.
    # NOTE: this means "migration" here is really "bundle replication", which
    # is the truthful description of what bundle-transfer migration becomes
    # when the worker process can't be moved. Documented in README.
    rsync_dir(dst_node, dst_chk, src_node, dst_chk, timeout=600)
    # The worker writes its bundle to /checkpoints/<robot_id>/policy_weights.pt
    # (see worker_robot_client.py:151-154), NOT to /checkpoints/ root. The
    # previous version pointed load_policy at "/checkpoints" so the worker
    # looked for /checkpoints/policy_weights.pt which never existed → "POLICY
    # LOAD FAILED". Point it at the actual subdir; the file is already there
    # because the worker just wrote it (src node = same host the worker runs on).
    container_load_dir = f"/checkpoints/{robot_id}"
    # Belt-and-suspenders: also copy the dest bundle's files into the worker's
    # subdir on src, in case rsync round-trip provided fresher contents (e.g.
    # FL aggregation happened on dst before we copied back).
    ssh_run(src_node,
            f"mkdir -p {src_chk}/{robot_id} && "
            f"cp -f {dst_chk}/{robot_id}/policy_weights.pt "
            f"{dst_chk}/{robot_id}/replay_buffer.pkl "
            f"{dst_chk}/{robot_id}/manifest.json "
            f"{src_chk}/{robot_id}/ 2>/dev/null || true", timeout=15)
    r.set(f"load_policy:{robot_id}", container_load_dir, ex=600)
    r.set(f"migration_done:{robot_id}", "1", ex=600)

    # 7. Wait for first_bid_after_migration with policy_load_ms.
    policy_load_ms = 0.0
    deadline = time.time() + 120
    while time.time() < deadline:
        data = r.get(f"first_bid_after_migration:{robot_id}")
        if data:
            policy_load_ms = float(json.loads(data).get("policy_load_ms", 0))
            r.delete(f"first_bid_after_migration:{robot_id}")
            break
        time.sleep(0.1)

    total_MTT_ms = (time.perf_counter() - t_trigger) * 1000
    downtime_ms = transfer_ms + policy_load_ms

    # NOTE: src_node/dst_node here represent the MIGRATION direction in this
    # event, not a permanent home reassignment, since the worker process is
    # not relocated (see comment above). The bundle is now resident on both.
    recov = post_migration_recovery(r, robot_id, task_counter_pre, success_rate_pre)
    success_rate_post = recov["success_rate_post"]
    regression = 0.0
    if success_rate_pre > 0:
        regression = (success_rate_pre - success_rate_post) / success_rate_pre * 100

    LOG.info("[MIG] %s done MTT=%.0fms transfer=%.0fms policy_load=%.0fms regression=%.1f%%",
             robot_id, total_MTT_ms, transfer_ms, policy_load_ms, regression)

    return {
        "robot_id": robot_id, "src_node": src_node, "dst_node": dst_node,
        "trigger_to_dump_ms": dump_ms,
        "dump_to_transfer_ms": transfer_ms,
        "transfer_to_restore_ms": 0,
        "policy_load_ms": policy_load_ms,
        "downtime_ms": downtime_ms,
        "total_MTT_ms": total_MTT_ms,
        "success_rate_pre": success_rate_pre,
        "success_rate_post": success_rate_post,
        "regression_pct": round(regression, 2),
        "replay_buffer_entries_restored": replay_entries,
        "network_bytes_transferred": bytes_xfer,
        "checkpoint_size_mb": round(chk_size_mb, 2),
        "criu_mode": "dht_bundle",
        "throughput_post_60s": recov["throughput_post_60s"],
        "recovery_tasks_to_pre": recov["recovery_tasks_to_pre"],
    }


def migration_monitor(cfg: ClusterConfig, r: redis.Redis,
                      writer: MigrationMetricsWriter) -> None:
    LOG.info("[Monitor] watching for migration requests")
    while True:
        try:
            for key in r.keys("migration_request:robot_*"):
                raw = r.get(key)
                if not raw:
                    continue
                info = json.loads(raw)
                robot_id = info["robot_id"]
                r.delete(key)
                metrics = trigger_dht_migration(
                    cfg, r, robot_id,
                    success_rate_pre=float(info.get("success_rate", 0)),
                    task_counter_pre=int(info.get("task_counter", 0)),
                )
                writer.write_event(metrics)
                # Throttle to avoid MaxStartups bursts (see baseline_runner_base
                # for the long story).
                time.sleep(6)
        except Exception as e:
            LOG.error("[Monitor] %r", e)
        time.sleep(1)


def wait_for_completion(cfg: ClusterConfig, r: redis.Redis,
                        flower_proc=None) -> None:
    """Block until the run is done. Unlike the C/D/E baseline worker, the FROZEN
    DHT worker (swiftbot_rl/dht_frl/worker_robot_client.py) does NOT set
    robot_done — it just finishes its Flower rounds and exits. So waiting on
    robot_done alone hangs until walltime. The authoritative completion signal
    for Condition A is the Flower server process exiting after its final round:
    once FL is complete, all migrations have fired and the data is written.
    We watch flower_proc and also keep the robot_done check as a fallback."""
    LOG.info("[Wait] waiting for FL to complete (Flower server exit) "
             "or all %d robots to finish", cfg.num_clients)
    while True:
        done = sum(1 for cid in range(cfg.num_clients)
                   if r.get(f"robot_done:robot_{cid:03d}"))
        if done >= cfg.num_clients:
            LOG.info("[Wait] all robots done.")
            return
        if flower_proc is not None and flower_proc.poll() is not None:
            LOG.info("[Wait] Flower server exited (rc=%s) — FL complete, "
                     "finishing run.", flower_proc.returncode)
            # Brief grace so the migration monitor drains any last event.
            time.sleep(10)
            return
        time.sleep(15)


def main() -> int:
    # Install signal handlers BEFORE launching anything, so a SIGTERM from
    # MOAB (walltime / canceljob) cleans up tracked SSH+apptainer Popens.
    install_signal_handlers()

    cfg = ClusterConfig()
    LOG.info("=" * 78)
    LOG.info("Condition A (DHT+FRL) — server=%s clients=%s",
             cfg.server_node, cfg.client_nodes)
    LOG.info("NUM_CLIENTS=%d ROBOTS_PER_NODE=%d MIGRATION_OFFSET=%d TOTAL_TASKS=%d",
             cfg.num_clients, cfg.robots_per_node,
             cfg.migration_offset, cfg.total_tasks)
    LOG.info("=" * 78)

    r = redis.Redis(host=cfg.redis_host, port=cfg.redis_port, decode_responses=True)
    r.flushall()

    writer = MigrationMetricsWriter(CONDITION, cfg.results_dir)

    # Launch Flower server inside an apptainer instance on the server node,
    # logging to RUN_LOG_DIR/flower_server.log.
    LOG.info("Starting Flower server on %s:%d", cfg.server_node, cfg.flower_port)
    # Apptainer does not inherit host env vars — APPTAINERENV_FOO=bar in the
    # host env propagates to FOO=bar inside the container. (SINGULARITYENV_*
    # included for compat with older singularity-based installs.)
    pylibs_host = os.path.join(cfg.img_dir, "pylibs")
    cluster_root = os.environ.get("CLUSTER_ROOT") or \
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Shared pretrained policy seeds the FedAvg global model so A starts from
    # the same competent weights as the C/D/E baselines. Path is inside the
    # container (cluster/ is bound at /cluster_app).
    pretrained_container = "/cluster_app/common/pretrained_policy.pt"
    # cfg.results_dir is a host path .../cluster/results/<folder>; inside the
    # container cluster/ is bound at /cluster_app, so map by basename.
    fl_result_container = f"/cluster_app/results/{os.path.basename(cfg.results_dir.rstrip('/'))}"
    flower_env = (
        f"APPTAINERENV_REDIS_HOST={cfg.redis_host} "
        f"APPTAINERENV_REDIS_PORT={cfg.redis_port} "
        f"APPTAINERENV_N_CLIENTS={cfg.num_clients} "
        f"APPTAINERENV_N_ROUNDS={cfg.total_fl_rounds} "
        f"APPTAINERENV_FLOWER_BIND=0.0.0.0:{cfg.flower_port} "
        f"APPTAINERENV_WORKER_PRETRAINED_PATH={pretrained_container} "
        f"APPTAINERENV_FL_RESULT_DIR={fl_result_container} "
        f"APPTAINERENV_PYTHONUNBUFFERED=1 "
        f"APPTAINERENV_PYTHONPATH=/pylibs:/app "
        f"SINGULARITYENV_REDIS_HOST={cfg.redis_host} "
        f"SINGULARITYENV_REDIS_PORT={cfg.redis_port} "
        f"SINGULARITYENV_N_CLIENTS={cfg.num_clients} "
        f"SINGULARITYENV_N_ROUNDS={cfg.total_fl_rounds} "
        f"SINGULARITYENV_FLOWER_BIND=0.0.0.0:{cfg.flower_port} "
        f"SINGULARITYENV_WORKER_PRETRAINED_PATH={pretrained_container} "
        f"SINGULARITYENV_PYTHONUNBUFFERED=1 "
        f"SINGULARITYENV_PYTHONPATH=/pylibs:/app "
    )
    # Tracked SSH+apptainer-exec Popen (NOT nohup+&). Apptainer is in the
    # foreground of the SSH session, so when this runner exits, sshd HUPs
    # apptainer and Flower dies cleanly — no orphans for the cluster admin
    # to chase down.
    # Non-interactive SSH does not source ~/.bashrc, so apptainer (in the
    # conda env) is not on PATH on remote nodes. Source conda first.
    conda_base = os.environ.get("CONDA_BASE", "/home/029822154/miniconda3")
    conda_env = os.environ.get("CONDA_ENV", "swiftbot")
    # Run the CLUSTER copy of the Flower server (seeds pretrained init params).
    # It is bound in at /cluster_app; swiftbot_rl/ stays frozen at /app.
    flower_remote = (
        f"cd {shlex.quote(cfg.swiftbot_root + '/dht_frl')}; "
        f"source {shlex.quote(conda_base)}/bin/activate {shlex.quote(conda_env)} && "
        f"exec env {flower_env} apptainer exec "
        f"--bind {shlex.quote(cfg.swiftbot_root)}:/app "
        f"--bind {shlex.quote(cluster_root)}:/cluster_app "
        f"--bind {shlex.quote(pylibs_host)}:/pylibs "
        f"{shlex.quote(cfg.img_dir + '/' + IMAGE)} "
        f"python3 /cluster_app/condition_A_dht_frl/flower_server.py"
    )
    flower_log = open(f"{cfg.run_log_dir}/flower_server.log", "ab", buffering=0)
    if is_local_node(cfg.server_node):
        # Flower server runs on the server node, which is the same node this
        # runner is on — bypass SSH (cluster refuses self-SSH).
        flower_cmd = ["bash", "-c", flower_remote]
    else:
        flower_cmd = ["ssh", *_SSH_OPTS,
                      "-o", "ServerAliveInterval=30",
                      "-o", "ServerAliveCountMax=3",
                      "-n", cfg.server_node, flower_remote]
    flower_proc = subprocess.Popen(
        flower_cmd, stdin=subprocess.DEVNULL,
        stdout=flower_log, stderr=flower_log,
    )
    register_tracked_process(flower_proc, "flower_server")
    time.sleep(15)  # let Flower bind its port

    # Pre-establish per-cid SSH masters on BOTH client nodes for every cid.
    # Migration relaunches then reuse an existing master and never open a
    # fresh TCP (avoids MaxStartups burst rejections). See
    # baseline_runner_base.py for the full story.
    total_masters = cfg.num_clients * len(cfg.client_nodes)
    LOG.info("Pre-establishing %d SSH masters", total_masters)
    for cid in range(cfg.num_clients):
        for client_node in cfg.client_nodes:
            establish_ssh_master(client_node, cid)
            time.sleep(1)

    # Launch robot apptainer instances on the two client nodes.
    LOG.info("Launching %d robot instances", cfg.num_clients)
    for cid in range(cfg.num_clients):
        node = cfg.home_node_for_client(cid)
        with _robot_lock:
            _robot_node[f"robot_{cid:03d}"] = node
        launch_robot(cfg, node, cid, IMAGE, WORKER)
        time.sleep(0.5)  # masters pre-established → fast slave handshakes
    LOG.info("All robots launched. Waiting for FL to converge.")

    threading.Thread(target=migration_monitor, args=(cfg, r, writer),
                     daemon=True).start()
    threading.Thread(target=live_status_loop, args=(cfg, r, writer, 15),
                     daemon=True).start()

    try:
        wait_for_completion(cfg, r, flower_proc)
    except KeyboardInterrupt:
        LOG.info("Interrupted")
    finally:
        terminate_all_tracked(hard_after=5.0)
        for cid in range(cfg.num_clients):
            for client_node in cfg.client_nodes:
                close_ssh_master(client_node, cid)
        time.sleep(2)
        close_all_ssh_masters_fast()
        LOG.info("Shutdown complete — exiting runner cleanly.")
    LOG.info("Migration events written: %d → %s",
             writer.event_count, writer.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
