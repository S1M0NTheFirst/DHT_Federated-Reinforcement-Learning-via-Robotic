"""
Condition C — CRIU cold migration cross-node.

For each migration:
  1. Identify the host PID of the worker running inside the apptainer
     instance on the source node.
  2. `criu dump --tree <pid>` on src node → /tmp/swiftbot_<cond>/<inst>/criu/.
     If criu is unavailable or fails (very common with rootless apptainer +
     CUDA), we fall back to SIMULATE — sleep for a synthetic dump time and
     proceed. The paper's headline finding is that CRIU+CUDA is brittle on
     consumer hardware AND on shared HPC where the user lacks CAP_SYS_ADMIN.
  3. rsync the criu image dir from src node to dst node.
  4. `criu restore` on dst node (real CRIU mode), or just launch a fresh
     apptainer instance there (SIMULATE mode — closer in spirit to E, but
     with the dump+transfer overhead recorded).

Cross-node: source picked from current_node(robot_id); destination is the
other client node.
"""
import logging, os, sys, time, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.cluster_runner import (
    ClusterConfig, apptainer_instance_name,
    launch_robot, kill_robot, get_robot_pid,
    remote_criu_dump, remote_criu_restore,
    rsync_dir, ssh_run, simulate_dump_seconds,
)
from common.baseline_runner_base import current_node, update_node, run_baseline

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
LOG = logging.getLogger("condC")

CONDITION = "criu_cold"
IMAGE     = "baseline.sif"
# Use cold_restart's worker for all four baseline conditions on cluster:
# it's the only baseline worker with resume_counter support, which we need
# because the cluster runners always kill+relaunch the apptainer instance
# on the dst node (a CRIU-restored process can't be tracked by apptainer
# instance management). The criu_cold/criu_warm/docker_checkpoint workers
# would restart from task_counter=0 after each migration → robot stuck.
WORKER    = "cold_restart/worker_random_client.py"


def trigger_criu_cold(cfg, r, robot_id, success_rate_pre, task_counter_pre):
    cid = int(robot_id.split("_")[1])
    src = current_node(robot_id)
    dst = cfg.other_node(src)
    inst = apptainer_instance_name(cid)
    src_dir = f"{cfg.checkpoint_base}/{inst}/criu_cold"
    dst_dir = f"{cfg.checkpoint_base}/{inst}/criu_cold"

    LOG.info("[MIG] %s  %s → %s (criu cold)", robot_id, src, dst)
    t0 = time.perf_counter()

    used_simulate = cfg.simulate_criu
    chk_size_mb   = 0.0
    bytes_xfer    = 0

    # 1. Dump on src.
    t_dump_start = time.perf_counter()
    if not used_simulate:
        pid = get_robot_pid(src, cid)
        if pid <= 0:
            LOG.warning("[MIG] could not find pid for %s on %s; falling back to SIMULATE",
                        robot_id, src)
            used_simulate = True
        else:
            ssh_run(src, f"mkdir -p {src_dir}", timeout=10)
            res = remote_criu_dump(src, pid, src_dir, leave_running=True, timeout=120)
            if res["returncode"] != 0:
                LOG.warning("[MIG] criu dump rc=%d for %s; SIMULATE fallback. tail=%s",
                            res["returncode"], robot_id, res["stderr"][-300:])
                used_simulate = True
            else:
                chk_size_mb = res["size_mb"]
    if used_simulate:
        time.sleep(simulate_dump_seconds())
    t_dump_done = time.perf_counter()

    # 2. Transfer.
    t_xfer_start = time.perf_counter()
    if not used_simulate:
        xfer = rsync_dir(src, src_dir, dst, dst_dir, timeout=600)
        bytes_xfer = xfer["bytes_transferred"]
    else:
        # Synthetic transfer — assume 400 MB/s, ~1.4 GB process image
        time.sleep(random.triangular(2.0, 4.0, 3.0))
    t_xfer_done = time.perf_counter()

    # 3. Restore: with real criu, restore on dst; with simulate, kill src and
    # launch fresh on dst (matches Condition E mechanic but with overhead
    # already accounted for in dump+transfer).
    t_restore_start = time.perf_counter()
    # Always kill src + launch fresh apptainer instance on dst, regardless
    # of whether real CRIU "succeeded" — a CRIU-restored process lives
    # outside any apptainer instance, so future kill_robot() / pid lookups
    # for this robot would fail to find it. The CRIU dump+transfer overhead
    # is still recorded above, which is what the paper actually measures.
    if not used_simulate:
        remote_criu_restore(dst, dst_dir,
                            f"{cfg.run_log_dir}/criu_restore_{cid}.log",
                            timeout=120)
    else:
        time.sleep(random.triangular(0.6, 1.0, 1.4))  # simulated cold restore
    kill_robot(cfg, src, cid)
    launch_robot(cfg, dst, cid, IMAGE, WORKER)
    t_restore_done = time.perf_counter()
    update_node(robot_id, dst)

    r.set(f"migration_done:{robot_id}", "1", ex=600)

    total_ms = (t_restore_done - t0) * 1000
    return {
        "robot_id": robot_id, "src_node": src, "dst_node": dst,
        "trigger_to_dump_ms":     (t_dump_done   - t_dump_start) * 1000,
        "dump_to_transfer_ms":    (t_xfer_done   - t_xfer_start) * 1000,
        "transfer_to_restore_ms": (t_restore_done- t_restore_start) * 1000,
        "policy_load_ms": 0,
        "downtime_ms": total_ms,
        "total_MTT_ms": total_ms,
        "success_rate_pre": success_rate_pre,
        "success_rate_post": 0,
        "regression_pct": 0,
        "replay_buffer_entries_restored": 0,
        "network_bytes_transferred": bytes_xfer,
        "checkpoint_size_mb": chk_size_mb,
        "criu_mode": "cold_simulated" if used_simulate else "cold_real",
    }


if __name__ == "__main__":
    sys.exit(run_baseline(
        condition=CONDITION, image=IMAGE,
        worker_script=WORKER, trigger_fn=trigger_criu_cold,
    ))
