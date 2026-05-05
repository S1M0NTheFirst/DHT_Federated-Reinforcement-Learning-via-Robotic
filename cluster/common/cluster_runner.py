"""
Shared host-side orchestration helpers used by every condition runner on the
cluster. The host-side runner runs on the SERVER node (the one that also
hosts Redis and, for Condition A, the Flower server).

Key differences vs the workstation runners:
  - No Docker SDK. Containers are apptainer instances launched via SSH.
  - No localhost. Redis and Flower live on $SERVER_NODE; clients live on
    $CLIENT_NODE_1 and $CLIENT_NODE_2.
  - Migration is always cross-node (a robot on C1 migrates to C2 and vice
    versa). The runner picks the destination per event.
  - CRIU images are written under $CHECKPOINT_BASE on the source node and
    rsync'd to the destination node before restore.

Environment variables (set by run_X.sh before invoking the runner):
    SERVER_NODE, CLIENT_NODE_1, CLIENT_NODE_2,
    REDIS_HOST, REDIS_PORT, FLOWER_PORT,
    NUM_CLIENTS, ROBOTS_PER_NODE, MIGRATION_OFFSET, TOTAL_TASKS,
    PROJECT_ROOT, SWIFTBOT_RL_ROOT, IMG_DIR,
    RUN_LOG_DIR, RESULTS_DIR, CONDITION,
    SIMULATE_CRIU
"""
from __future__ import annotations

import csv
import json
import logging
import os
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Optional

LOG = logging.getLogger("cluster_runner")


# --------------------------------------------------------------------------- #
# Config — read once at import time from environment.                         #
# --------------------------------------------------------------------------- #

def _env(name: str, default: Optional[str] = None) -> str:
    v = os.environ.get(name, default)
    if v is None:
        raise RuntimeError(f"Required env var {name} is not set")
    return v


class ClusterConfig:
    def __init__(self) -> None:
        self.server_node    = _env("SERVER_NODE")
        self.client_nodes   = [_env("CLIENT_NODE_1"), _env("CLIENT_NODE_2")]
        self.redis_host     = _env("REDIS_HOST", self.server_node)
        self.redis_port     = int(_env("REDIS_PORT", "6470"))
        self.flower_port    = int(_env("FLOWER_PORT", "8470"))
        self.num_clients    = int(_env("NUM_CLIENTS", "20"))
        self.robots_per_node = int(_env("ROBOTS_PER_NODE", "10"))
        self.migration_offset = int(_env("MIGRATION_OFFSET", "10"))
        self.total_tasks    = int(_env("TOTAL_TASKS", "1200"))
        self.project_root   = _env("PROJECT_ROOT")
        self.swiftbot_root  = _env("SWIFTBOT_RL_ROOT")
        self.img_dir        = _env("IMG_DIR")
        self.run_log_dir    = _env("RUN_LOG_DIR")
        self.results_dir    = _env("RESULTS_DIR")
        self.condition      = _env("CONDITION")
        self.simulate_criu   = _env("SIMULATE_CRIU", "0") == "1"
        self.total_fl_rounds = int(_env("TOTAL_FL_ROUNDS", "60"))
        self.checkpoint_base = f"/tmp/swiftbot_{self.condition}"

    def home_node_for_client(self, cid: int) -> str:
        """Initial node for client cid before any migration (round-robin)."""
        return self.client_nodes[cid // self.robots_per_node]

    def other_node(self, node: str) -> str:
        return self.client_nodes[1] if node == self.client_nodes[0] else self.client_nodes[0]


# --------------------------------------------------------------------------- #
# SSH helpers — every cross-node action goes through these.                   #
# --------------------------------------------------------------------------- #

_SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
]

def ssh_run(node: str, command: str, *, timeout: int = 60,
            check: bool = False) -> subprocess.CompletedProcess:
    """Run a remote shell command and capture output."""
    cmd = ["ssh", *_SSH_OPTS, "-n", node, command]
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, check=check)


def ssh_run_async(node: str, command: str, log_path: str) -> subprocess.Popen:
    """Run a remote command, streaming stdout/stderr to log_path on the runner host.
    Returns a Popen handle that the caller can wait on or terminate.
    """
    fh = open(log_path, "ab", buffering=0)
    cmd = ["ssh", *_SSH_OPTS, "-n", node, command]
    return subprocess.Popen(cmd, stdout=fh, stderr=fh, stdin=subprocess.DEVNULL)


def ssh_detached(node: str, command: str) -> None:
    """Fire-and-forget: launch a remote command via nohup and return immediately."""
    wrapped = f"nohup bash -lc {shlex.quote(command)} >/dev/null 2>&1 &"
    subprocess.run(["ssh", *_SSH_OPTS, "-n", "-f", node, wrapped],
                   timeout=15, check=False)


def rsync_dir(src_node: str, src_path: str, dst_node: str, dst_path: str,
              timeout: int = 600) -> dict:
    """rsync a directory between two cluster nodes via SSH. Returns
    dict(returncode, bytes_transferred, elapsed_ms)."""
    t0 = time.perf_counter()
    # Pull-style rsync: ssh into dst_node and have IT rsync from src_node.
    # This way the source's images don't have to first be copied to the
    # runner host (which may not even have access to the src node's /tmp).
    cmd = (
        f"mkdir -p {shlex.quote(dst_path)} && "
        f"rsync -aH --info=stats2 -e 'ssh -o StrictHostKeyChecking=no -o BatchMode=yes' "
        f"{shlex.quote(src_node)}:{shlex.quote(src_path)}/ "
        f"{shlex.quote(dst_path)}/"
    )
    rr = ssh_run(dst_node, cmd, timeout=timeout)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    bytes_xfer = 0
    for line in (rr.stdout or "").splitlines():
        if line.lower().startswith("total transferred file size:"):
            bytes_xfer = int(line.split(":", 1)[1].strip().split()[0].replace(",", ""))
    return {"returncode": rr.returncode, "bytes_transferred": bytes_xfer,
            "elapsed_ms": elapsed_ms,
            "stderr": (rr.stderr or "")[-500:]}


# --------------------------------------------------------------------------- #
# Apptainer instance launching.                                               #
# --------------------------------------------------------------------------- #

def apptainer_instance_name(cid: int) -> str:
    return f"swiftbot_robot_{cid:03d}"


def launch_robot(cfg: ClusterConfig, node: str, cid: int, image: str,
                 worker_script: str, *, extra_env: Optional[dict] = None,
                 container_type: str = "gpu_specialist") -> None:
    """SSH into `node` and launch one apptainer instance running the worker
    script. The instance is named swiftbot_robot_{cid:03d} so we can find it
    later for migration / kill. Logs are tee'd into RUN_LOG_DIR/robot_{cid}.log
    on this (server) node by piping through ssh.
    """
    inst   = apptainer_instance_name(cid)
    chk    = f"{cfg.checkpoint_base}/{inst}"
    log    = f"{cfg.run_log_dir}/robot_{cid:03d}.log"

    # Apptainer does NOT inherit host env vars into the container. The most
    # portable way to pass them (works on every apptainer & singularity
    # version) is the APPTAINERENV_FOO=bar / SINGULARITYENV_FOO=bar prefix
    # mechanism: set them in the host env via `env`, apptainer strips the
    # prefix and exports the rest inside the container.
    env_pairs = {
        "REDIS_HOST":       cfg.redis_host,
        "REDIS_PORT":       cfg.redis_port,
        "MASTER_ADDRESS":   f"{cfg.server_node}:{cfg.flower_port}",
        "NUM_CLIENTS":      cfg.num_clients,
        "MIGRATION_OFFSET": cfg.migration_offset,
        "TOTAL_TASKS":      cfg.total_tasks,
        "TOTAL_FL_ROUNDS":  cfg.total_fl_rounds,
        "PYTHONUNBUFFERED": 1,
    }
    if extra_env:
        env_pairs.update(extra_env)
    env_prefix = " ".join(
        f"APPTAINERENV_{k}={shlex.quote(str(v))} "
        f"SINGULARITYENV_{k}={shlex.quote(str(v))}"
        for k, v in env_pairs.items()
    )

    # bind-mount swiftbot_rl/ at /app so the existing worker scripts run
    # without any image rebuild. Each robot gets its own /checkpoints dir.
    # Apptainer instance start is daemonized; the second exec actually runs
    # the worker (kept separate from the instance startscript so we can
    # cleanly tee stdout/stderr to a per-robot log on the run-log share).
    remote_cmd = (
        f"mkdir -p {chk} && "
        f"apptainer instance stop -s KILL {inst} 2>/dev/null || true; "
        f"apptainer instance start --nv "
        f"  --bind {cfg.swiftbot_root}:/app "
        f"  --bind {chk}:/checkpoints "
        f"  {cfg.img_dir}/{image} {inst} && "
        f"sleep 3 && "
        f"nohup env {env_prefix} apptainer exec instance://{inst} "
        f"  python3 /app/{worker_script} "
        f"    --client-id {cid} --num-clients {cfg.num_clients} "
        f"    --container-type {container_type} "
        f"    > {log} 2>&1 &"
    )
    ssh_run(node, remote_cmd, timeout=120, check=False)


def kill_robot(cfg: ClusterConfig, node: str, cid: int) -> None:
    """Forcefully stop a robot. The cold_restart-style worker sits in a
    `while True: time.sleep(5)` loop after requesting migration with no
    shutdown check inside the loop, so SIGTERM (apptainer's default) just
    interrupts one sleep iteration. We send SIGKILL to the apptainer
    instance AND pkill -9 the python worker as a belt-and-suspenders kill.
    """
    inst = apptainer_instance_name(cid)
    ssh_run(node,
            f"apptainer instance stop -s KILL {inst} 2>/dev/null; "
            f"pkill -9 -u $USER -f 'client-id {cid} --num-clients' || true",
            timeout=30, check=False)


def get_robot_pid(node: str, cid: int) -> int:
    """Host PID of the python worker inside apptainer instance for cid.
    Returns 0 if not found (instance not running).
    """
    inst = apptainer_instance_name(cid)
    rr = ssh_run(
        node,
        f"pgrep -u $USER -f 'apptainer.*{inst}.*client-id {cid}' "
        f"| head -1",
        timeout=10,
    )
    try:
        return int((rr.stdout or "").strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0


# --------------------------------------------------------------------------- #
# Migration metrics CSV writer — same column set as workstation runs.         #
# --------------------------------------------------------------------------- #

class MigrationMetricsWriter:
    FIELDNAMES = [
        "condition", "robot_id", "migration_event_id", "timestamp",
        "src_node", "dst_node",
        "trigger_to_dump_ms", "dump_to_transfer_ms", "transfer_to_restore_ms",
        "policy_load_ms", "downtime_ms", "total_MTT_ms",
        "success_rate_pre", "success_rate_post", "regression_pct",
        "fl_rounds_to_recover",
        "replay_buffer_entries_restored",
        "gpu_util_pre_migration", "gpu_util_during_migration", "gpu_util_post_migration",
        "cpu_util_pre_migration", "cpu_util_during_migration", "cpu_util_post_migration",
        "network_bytes_transferred",
        "checkpoint_size_mb",
        "criu_mode",
    ]

    def __init__(self, condition: str, results_dir: str) -> None:
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
# Post-migration success-rate probe (reads task_logs from Redis).             #
# --------------------------------------------------------------------------- #

def post_migration_success_rate(r, robot_id: str, baseline_count: int,
                                 *, n: int = 10, timeout_s: float = 120.0) -> float:
    deadline = time.time() + timeout_s
    seen = set()
    new_tasks = []
    while time.time() < deadline and len(new_tasks) < n:
        for raw in r.lrange("task_logs", 0, 600):
            try:
                e = json.loads(raw)
            except Exception:
                continue
            if e.get("robot_id") != robot_id:
                continue
            tc = e.get("task_counter", 0)
            if tc > baseline_count and tc not in seen:
                seen.add(tc); new_tasks.append(e)
        time.sleep(0.5)
    if not new_tasks:
        return 0.0
    return sum(1 for t in new_tasks[:n] if t.get("status") == "success") / min(n, len(new_tasks))


# --------------------------------------------------------------------------- #
# Live status thread — same shape as workstation, prints every interval s.    #
# --------------------------------------------------------------------------- #

def live_status_loop(cfg: ClusterConfig, r, writer: MigrationMetricsWriter,
                     interval: int = 15) -> None:
    LOG.info("[Status] live status thread started (every %ds)", interval)
    time.sleep(interval)
    while True:
        try:
            latest: dict = {}
            for raw in r.lrange("task_logs", 0, 1500):
                try:
                    e = json.loads(raw)
                except Exception:
                    continue
                rid = e.get("robot_id")
                if rid and rid not in latest:
                    latest[rid] = e
                if len(latest) >= cfg.num_clients:
                    break
            rows = []
            for cid in range(cfg.num_clients):
                rid = f"robot_{cid:03d}"
                e = latest.get(rid)
                if not e:
                    rows.append(f"  {rid}: <no tasks yet>")
                    continue
                rows.append(
                    f"  {rid}: tasks={e.get('task_counter',0):>4}  "
                    f"fl_round={e.get('fl_round',0):>2}  "
                    f"success10={e.get('success_rate_rolling10',0):.2f}  "
                    f"reward={e.get('reward',0):+.2f}  "
                    f"step={e.get('training_step',0)}"
                )
            pending = r.keys("migration_request:robot_*")
            extra = f" migrations_done={writer.event_count}"
            if pending:
                extra += f" pending={len(pending)}"
            LOG.info("=" * 78)
            LOG.info("[Status] live snapshot%s", extra)
            for line in rows:
                LOG.info(line)
            LOG.info("=" * 78)
        except Exception as e:
            LOG.error("[Status] %r", e)
        time.sleep(interval)


# --------------------------------------------------------------------------- #
# CRIU helpers — these run via SSH on the source/destination node.            #
# --------------------------------------------------------------------------- #

def remote_criu_dump(node: str, pid: int, images_dir: str,
                     parent_dir: str = "", *, pre_dump: bool = False,
                     leave_running: bool = True, timeout: int = 120) -> dict:
    """Run `criu dump` on a remote node. Returns dict with returncode,
    size_mb, stderr (tail). Falls back to a SIMULATE marker if criu is missing
    or the dump fails — caller should check returncode.
    """
    if pid <= 0:
        return {"returncode": -1, "size_mb": 0.0,
                "stderr": "pid<=0 (instance not found)"}
    subcmd = "pre-dump" if pre_dump else "dump"
    parent_flag = ""
    if parent_dir:
        # CRIU expects a `parent` symlink in images_dir → ../parent_basename
        parent_flag = (
            f"if [ -d {shlex.quote(parent_dir)} ]; then "
            f"  ln -sfn $(realpath --relative-to={shlex.quote(images_dir)} "
            f"{shlex.quote(parent_dir)}) {shlex.quote(images_dir)}/parent; "
            f"fi && "
        )
    leave_flag = "--leave-running" if (leave_running and not pre_dump) else ""
    track_flag = "--track-mem" if (parent_dir or pre_dump) else ""
    cmd = (
        f"mkdir -p {shlex.quote(images_dir)} && {parent_flag} "
        f"criu {subcmd} --tree {pid} --images-dir {shlex.quote(images_dir)} "
        f"--tcp-established --shell-job --ext-unix-sk --manage-cgroups=soft "
        f"{leave_flag} {track_flag} 2>&1 | tee {shlex.quote(images_dir)}/criu.log; "
        f"echo __CRIU_RC__$?"
    )
    t0 = time.perf_counter()
    rr = ssh_run(node, cmd, timeout=timeout)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    rc = -1
    out = rr.stdout or ""
    for line in out.splitlines():
        if line.startswith("__CRIU_RC__"):
            try:
                rc = int(line[len("__CRIU_RC__"):])
            except ValueError:
                pass
    # Get image-dir size.
    sz = ssh_run(node, f"du -sb {shlex.quote(images_dir)} 2>/dev/null | cut -f1",
                 timeout=10)
    size_mb = 0.0
    try:
        size_mb = int((sz.stdout or "0").strip()) / (1024 * 1024)
    except ValueError:
        pass
    return {"returncode": rc, "size_mb": round(size_mb, 2),
            "stderr": out[-1500:], "elapsed_ms": elapsed_ms}


def remote_criu_restore(node: str, images_dir: str, log_path: str,
                        timeout: int = 120) -> dict:
    """Run `criu restore --restore-detached` on a remote node. The restored
    process becomes a child of init on that node.
    """
    cmd = (
        f"criu restore --images-dir {shlex.quote(images_dir)} "
        f"--tcp-established --shell-job --ext-unix-sk --manage-cgroups=soft "
        f"--restore-detached --restore-sibling 2>&1 | tee {shlex.quote(log_path)}; "
        f"echo __CRIU_RC__$?"
    )
    t0 = time.perf_counter()
    rr = ssh_run(node, cmd, timeout=timeout)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    rc = -1
    for line in (rr.stdout or "").splitlines():
        if line.startswith("__CRIU_RC__"):
            try:
                rc = int(line[len("__CRIU_RC__"):])
            except ValueError:
                pass
    return {"returncode": rc, "elapsed_ms": elapsed_ms,
            "stderr": (rr.stdout or "")[-1500:]}


def simulate_dump_seconds() -> float:
    """Synthetic dump time matching observed full-process CRIU on consumer
    GPUs (~5–9s). Used when real CRIU isn't available — keeps the comparison
    against Condition A meaningful even on this cluster."""
    import random
    return random.triangular(5.0, 9.0, 7.0)
