"""
Shared host-side runner for ALL task2 conditions. Each condition's runner.py
supplies only: CONDITION name, checkpoint_mode, extra worker env, and a
trigger_fn(cfg, r, robot_id, sr_pre, tc_pre) -> metrics dict.

Reuses task1's low-level SSH / tracked-process / rsync helpers from
common.cluster_runner (so the cluster-admin orphan-safety guarantees still
hold), but launches the task2 ONLINE-SAC worker + task2 Flower server with
task2 binds (pylibs2 for mujoco, /cluster_app/task2/worker on PYTHONPATH) and
CPU forced. Writes an EXTENDED migration_events.csv that includes the
policy-equivalence probe columns.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import shlex
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

import redis

# task1 helpers.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # .../cluster on path
from common.cluster_runner import (          # noqa: E402
    ClusterConfig,
    install_signal_handlers, register_tracked_process, terminate_all_tracked,
    is_local_node, _SSH_OPTS, _ssh_opts_for_cid,
    kill_robot, establish_ssh_master, close_ssh_master,
    close_all_ssh_masters_fast, live_status_loop, post_migration_recovery,
    rsync_dir, ssh_run, apptainer_instance_name,
)

LOG = logging.getLogger("task2_runner")

_robot_node: dict[str, str] = {}
_robot_lock = threading.Lock()


def current_node(rid: str) -> str:
    with _robot_lock:
        return _robot_node[rid]


def update_node(rid: str, node: str) -> None:
    with _robot_lock:
        _robot_node[rid] = node


# --------------------------------------------------------------------------- #
# Extended metrics writer — task1 columns + probe/losslessness columns.        #
# --------------------------------------------------------------------------- #
class Task2MetricsWriter:
    FIELDNAMES = [
        "condition", "robot_id", "migration_event_id", "timestamp", "fl_round",
        "src_node", "dst_node",
        "trigger_to_dump_ms", "dump_to_transfer_ms", "transfer_to_restore_ms",
        "policy_load_ms", "downtime_ms", "total_MTT_ms",
        "success_rate_pre", "success_rate_post", "regression_pct",
        "fl_rounds_to_recover", "replay_buffer_entries_restored",
        "gpu_util_pre_migration", "gpu_util_during_migration", "gpu_util_post_migration",
        "cpu_util_pre_migration", "cpu_util_during_migration", "cpu_util_post_migration",
        "network_bytes_transferred", "checkpoint_size_mb",
        "checkpoint_mode",                       # dht_bundle/app/tcp/dmtcp/none
        "throughput_post_60s", "recovery_tasks_to_pre",
        "background_bandwidth_mb", "concurrency_level",
        "fault_injected", "retry_count", "total_recovery_ms",
        # --- policy-equivalence probe (behavioral losslessness) ---
        "policy_action_mse", "policy_weight_l2",
    ]

    def __init__(self, condition: str, results_dir: str):
        self.condition = condition
        self.path = os.path.join(results_dir, "migration_events.csv")
        self._lock = threading.Lock()
        self._n = 0
        os.makedirs(results_dir, exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()

    def write_event(self, metrics: dict) -> None:
        with self._lock:
            self._n += 1
            row = {k: metrics.get(k, 0) for k in self.FIELDNAMES}
            row["condition"] = self.condition
            row["migration_event_id"] = self._n
            row["timestamp"] = time.time()
            with open(self.path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDNAMES).writerow(row)

    @property
    def event_count(self) -> int:
        return self._n


# --------------------------------------------------------------------------- #
# task2 container launch (worker + flower).                                    #
# --------------------------------------------------------------------------- #
def _worker_env(cfg: ClusterConfig, cid: int, extra: Optional[dict]) -> dict:
    e = {
        "REDIS_HOST": cfg.redis_host, "REDIS_PORT": cfg.redis_port,
        "MASTER_ADDRESS": f"{cfg.server_node}:{cfg.flower_port}",
        "CONDITION": cfg.condition,
        "TOTAL_FL_ROUNDS": os.environ.get("TOTAL_FL_ROUNDS", "150"),
        "STEPS_PER_ROUND": os.environ.get("STEPS_PER_ROUND", "1000"),
        "MIN_BUFFER_FILL": os.environ.get("MIN_BUFFER_FILL", "1000"),
        "BUFFER_CAPACITY": os.environ.get("BUFFER_CAPACITY", "100000"),
        "SAC_BATCH": os.environ.get("SAC_BATCH", "256"),
        "EVAL_EPISODES": os.environ.get("EVAL_EPISODES", "3"),
        "EVAL_SUCCESS_RETURN": os.environ.get("EVAL_SUCCESS_RETURN", "800"),
        "TASK2_ENV": os.environ.get("TASK2_ENV", "Hopper-v4"),
        "SHARED_SEED": os.environ.get("SHARED_SEED", "12345"),
        "MIGRATION_ROUNDS": os.environ.get("MIGRATION_ROUNDS", "30,60,90,120,140"),
        "MIGRATION_OFFSET": os.environ.get("MIGRATION_OFFSET", "0"),
        "PYTHONUNBUFFERED": 1,
        "PYTHONPATH": "/pylibs2:/pylibs:/cluster_app/task2/worker",
        # FORCE CPU + cap math threads (20 mujoco sims share ppn cores).
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": os.environ.get("WORKER_MATH_THREADS", "2"),
        "OPENBLAS_NUM_THREADS": os.environ.get("WORKER_MATH_THREADS", "2"),
        "MKL_NUM_THREADS": os.environ.get("WORKER_MATH_THREADS", "2"),
    }
    if extra:
        e.update(extra)
    return e


def launch_task2_robot(cfg: ClusterConfig, node: str, cid: int, *,
                       extra_env: Optional[dict] = None) -> None:
    chk = f"{cfg.checkpoint_base}/{apptainer_instance_name(cid)}"
    log = f"{cfg.run_log_dir}/robot_{cid:03d}.log"
    img_dir = cfg.img_dir
    cluster_root = os.environ.get("CLUSTER_ROOT")
    pylibs = os.path.join(img_dir, "pylibs")
    pylibs2 = os.environ["TASK2_PYLIBS2"]
    conda_base = os.environ.get("CONDA_BASE")
    conda_env = os.environ.get("CONDA_ENV", "base")

    env_pairs = _worker_env(cfg, cid, extra_env)
    env_prefix = " ".join(
        f"APPTAINERENV_{k}={shlex.quote(str(v))} "
        f"SINGULARITYENV_{k}={shlex.quote(str(v))}"
        for k, v in env_pairs.items()
    )
    remote = (
        f"mkdir -p {shlex.quote(chk)}; "
        f"source {shlex.quote(conda_base)}/bin/activate {shlex.quote(conda_env)} && "
        f"exec env {env_prefix} apptainer exec "
        f"--bind {shlex.quote(cluster_root)}:/cluster_app "
        f"--bind {shlex.quote(chk)}:/checkpoints "
        f"--bind {shlex.quote(pylibs)}:/pylibs "
        f"--bind {shlex.quote(pylibs2)}:/pylibs2 "
        f"{shlex.quote(img_dir + '/robot.sif')} "
        f"python3 /cluster_app/task2/worker/online_sac_worker.py "
        f"--client-id {cid} --container-type cpu_specialist"
    )
    log_fh = open(log, "ab", buffering=0)
    if is_local_node(node):
        cmd = ["bash", "-c", remote]
    else:
        cmd = ["ssh", *_ssh_opts_for_cid(cid),
               "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
               "-n", node, remote]
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                         stdout=log_fh, stderr=log_fh)
    register_tracked_process(p, f"task2_robot_{cid:03d}@{node}")


def launch_task2_flower(cfg: ClusterConfig) -> subprocess.Popen:
    img_dir = cfg.img_dir
    cluster_root = os.environ.get("CLUSTER_ROOT")
    pylibs = os.path.join(img_dir, "pylibs")
    pylibs2 = os.environ["TASK2_PYLIBS2"]
    conda_base = os.environ.get("CONDA_BASE")
    conda_env = os.environ.get("CONDA_ENV", "base")
    # Map the host results dir (under task2/results) to its container path.
    # cluster_root is bound at /cluster_app, so a results dir anywhere under it
    # (e.g. .../cluster/task2/results/task2_dht_frl) maps to
    # /cluster_app/task2/results/task2_dht_frl — keeps fl_*.csv + task_logs.csv
    # inside the task2 folder instead of cluster/results.
    rel = os.path.relpath(cfg.results_dir, cluster_root)
    fl_result_container = f"/cluster_app/{rel}".replace(os.sep, "/")
    env = {
        "N_CLIENTS": cfg.num_clients, "N_ROUNDS": cfg.total_fl_rounds,
        "FLOWER_BIND": f"0.0.0.0:{cfg.flower_port}",
        "FL_RESULT_DIR": fl_result_container,
        "REDIS_HOST": cfg.redis_host, "REDIS_PORT": cfg.redis_port,
        "SHARED_SEED": os.environ.get("SHARED_SEED", "12345"),
        "PYTHONUNBUFFERED": 1,
        "PYTHONPATH": "/pylibs2:/pylibs:/cluster_app/task2/worker",
        "CUDA_VISIBLE_DEVICES": "",
    }
    env_prefix = " ".join(
        f"APPTAINERENV_{k}={shlex.quote(str(v))} "
        f"SINGULARITYENV_{k}={shlex.quote(str(v))}"
        for k, v in env.items()
    )
    remote = (
        f"source {shlex.quote(conda_base)}/bin/activate {shlex.quote(conda_env)} && "
        f"exec env {env_prefix} apptainer exec "
        f"--bind {shlex.quote(cluster_root)}:/cluster_app "
        f"--bind {shlex.quote(pylibs)}:/pylibs "
        f"--bind {shlex.quote(pylibs2)}:/pylibs2 "
        f"{shlex.quote(img_dir + '/robot.sif')} "
        f"python3 /cluster_app/task2/flower_server.py"
    )
    flog = open(f"{cfg.run_log_dir}/flower_server.log", "ab", buffering=0)
    if is_local_node(cfg.server_node):
        cmd = ["bash", "-c", remote]
    else:
        cmd = ["ssh", *_SSH_OPTS, "-o", "ServerAliveInterval=30",
               "-o", "ServerAliveCountMax=3", "-n", cfg.server_node, remote]
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                        stdout=flog, stderr=flog)
    register_tracked_process(p, "task2_flower_server")
    return p


# --------------------------------------------------------------------------- #
# Probe helper — read the worker's probe metrics after resume.                 #
# --------------------------------------------------------------------------- #
def read_probe_metrics(r: redis.Redis, robot_id: str,
                       timeout_s: float = 120.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        raw = r.get(f"probe_metrics:{robot_id}")
        if raw:
            r.delete(f"probe_metrics:{robot_id}")
            try:
                return json.loads(raw)
            except Exception:
                break
        time.sleep(0.3)
    return {"policy_action_mse": -1, "policy_weight_l2": -1,
            "policy_load_ms": 0, "replay_entries_post": 0}


# --------------------------------------------------------------------------- #
# Orchestrator.                                                                #
# --------------------------------------------------------------------------- #
def run_task2(*, condition: str, checkpoint_mode: str, trigger_fn: Callable,
              initial_extra_env: Optional[dict] = None) -> int:
    install_signal_handlers()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    cfg = ClusterConfig()
    LOG.info("=" * 78)
    LOG.info("task2 %s — server=%s clients=%s rounds=%d",
             condition, cfg.server_node, cfg.client_nodes, cfg.total_fl_rounds)
    LOG.info("=" * 78)

    r = redis.Redis(host=cfg.redis_host, port=cfg.redis_port,
                    decode_responses=True)
    r.flushall()
    writer = Task2MetricsWriter(condition, cfg.results_dir)

    flower_proc = launch_task2_flower(cfg)
    time.sleep(15)

    for cid in range(cfg.num_clients):
        for cn in cfg.client_nodes:
            establish_ssh_master(cn, cid)
            time.sleep(1)

    for cid in range(cfg.num_clients):
        node = cfg.home_node_for_client(cid)
        update_node(f"robot_{cid:03d}", node)
        launch_task2_robot(cfg, node, cid, extra_env=initial_extra_env)
        time.sleep(0.5)
    LOG.info("All %d robots launched.", cfg.num_clients)

    def monitor():
        LOG.info("[Monitor] watching migration requests")
        while True:
            try:
                for key in r.keys("migration_request:robot_*"):
                    raw = r.get(key)
                    if not raw:
                        continue
                    info = json.loads(raw)
                    rid = info["robot_id"]
                    r.delete(key)
                    try:
                        metrics = trigger_fn(
                            cfg, r, rid,
                            float(info.get("success_rate", 0)),
                            int(info.get("task_counter", 0)),
                        )
                        metrics["fl_round"] = int(info.get("fl_round", 0))
                        metrics["checkpoint_mode"] = checkpoint_mode
                        metrics["concurrency_level"] = 1
                        writer.write_event(metrics)
                    except Exception as e:
                        import traceback
                        LOG.error("[Monitor] trigger_fn %s failed: %r\n%s",
                                  rid, e, traceback.format_exc())
                        # unblock the worker so the run doesn't hang
                        r.set(f"migration_done:{rid}", "1", ex=600)
                    time.sleep(6)
            except Exception as e:
                LOG.error("[Monitor] %r", e)
            time.sleep(1)

    threading.Thread(target=monitor, daemon=True).start()
    threading.Thread(target=live_status_loop, args=(cfg, r, writer, 15),
                     daemon=True).start()

    try:
        # Completion: Flower server exits after the final round (workers also set
        # robot_done). Watch both.
        while True:
            done = sum(1 for cid in range(cfg.num_clients)
                       if r.get(f"robot_done:robot_{cid:03d}"))
            if done >= cfg.num_clients:
                LOG.info("[Wait] all robots done.")
                break
            if flower_proc.poll() is not None:
                LOG.info("[Wait] Flower exited rc=%s — FL complete.",
                         flower_proc.returncode)
                time.sleep(10)
                break
            time.sleep(15)
    finally:
        terminate_all_tracked(hard_after=5.0)
        for cid in range(cfg.num_clients):
            for cn in cfg.client_nodes:
                close_ssh_master(cn, cid)
        time.sleep(2)
        close_all_ssh_masters_fast()
    LOG.info("Migration events: %d → %s", writer.event_count, writer.path)
    return 0
