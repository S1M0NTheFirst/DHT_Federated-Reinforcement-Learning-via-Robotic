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

import atexit
import csv
import json
import logging
import os
import shlex
import shutil
import signal
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

# SSH connection multiplexing: launching 10 robots/node back-to-back hit
# sshd's MaxStartups rate limiter (saw 6/10 "Connection closed by … port 22"
# even with a 3s gap between launches). ControlMaster reuses ONE TCP
# connection per (user,host,port) for all subsequent ssh calls, so we open
# one transport per node and stream all robot launches through it.
#
# ControlPath includes %h (host), %r (user), %p (port) AND PBS_JOBID so
# concurrent jobs don't fight over the same socket.
_SSH_CTRL_DIR = "/tmp/ssh-mux-%s" % os.environ.get("PBS_JOBID", str(os.getpid()))
os.makedirs(_SSH_CTRL_DIR, exist_ok=True)
_SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ControlMaster=auto",
    "-o", f"ControlPath={_SSH_CTRL_DIR}/%r@%h:%p",
    "-o", "ControlPersist=10m",
]


# Cache the local hostname at import time. Some clusters (CSULB HPC2 included)
# refuse SSH from a node to itself, so any "remote" command targeting the
# local node must be run as a plain bash subprocess instead of through SSH.
_LOCAL_FQDN = socket.getfqdn()
try:
    _LOCAL_SHORT = socket.gethostname().split(".")[0]
except Exception:
    _LOCAL_SHORT = _LOCAL_FQDN.split(".")[0]


def is_local_node(node: str) -> bool:
    """Return True if `node` refers to the host this Python process runs on."""
    if not node:
        return False
    if node == _LOCAL_FQDN or node == _LOCAL_SHORT:
        return True
    # Match short-name + any domain suffix ("n034" vs "n034.cluster.pssclabs.com")
    n_short = node.split(".")[0]
    return n_short == _LOCAL_SHORT

def ssh_run(node: str, command: str, *, timeout: int = 60,
            check: bool = False) -> subprocess.CompletedProcess:
    """Run a remote shell command and capture output. If `node` is the local
    host, runs the command via a local bash subprocess (this cluster refuses
    SSH from a node to itself).
    """
    if is_local_node(node):
        return subprocess.run(["bash", "-c", command],
                              capture_output=True, text=True,
                              timeout=timeout, check=check)
    cmd = ["ssh", *_SSH_OPTS, "-n", node, command]
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, check=check)


def ssh_run_async(node: str, command: str, log_path: str) -> subprocess.Popen:
    """Run a remote command, streaming stdout/stderr to log_path on the runner host.
    Falls back to local bash for self-targeted commands.
    """
    fh = open(log_path, "ab", buffering=0)
    if is_local_node(node):
        return subprocess.Popen(["bash", "-c", command],
                                stdout=fh, stderr=fh, stdin=subprocess.DEVNULL)
    cmd = ["ssh", *_SSH_OPTS, "-n", node, command]
    return subprocess.Popen(cmd, stdout=fh, stderr=fh, stdin=subprocess.DEVNULL)


# NOTE: ssh_detached() removed deliberately. The previous implementation
# used `ssh -f` + `nohup ... &` which double-detached the remote process —
# exactly the orphan pattern the cluster admin asked us to stop using
# (caused stuck containers requiring node reboot). Every remote process
# must now be launched as a tracked Popen child of the runner. Use
# ssh_run_async() and register_tracked_process() instead.


# --------------------------------------------------------------------------- #
# Tracked-process registry — every long-lived SSH/apptainer Popen we spawn    #
# is recorded here so we can terminate it cleanly on signal/exit. This is    #
# the cluster admin's requirement: "Ensure all processes stay tied to the   #
# job ... ensure Apptainer is run in the foreground within the job script". #
# --------------------------------------------------------------------------- #

_tracked_lock = threading.Lock()
_tracked_processes: list[tuple[subprocess.Popen, str]] = []
_robot_procs: dict[int, tuple[subprocess.Popen, str]] = {}
_signal_handlers_installed = False


def register_tracked_process(p: subprocess.Popen, desc: str = "") -> None:
    with _tracked_lock:
        _tracked_processes.append((p, desc))


def _terminate_one(p: subprocess.Popen, desc: str, *, hard_after: float) -> None:
    if p.poll() is not None:
        return
    try:
        p.terminate()
    except Exception:
        pass
    try:
        p.wait(timeout=hard_after)
    except subprocess.TimeoutExpired:
        try:
            p.kill()
        except Exception:
            pass
        try:
            p.wait(timeout=2)
        except Exception:
            pass


def terminate_all_tracked(*, hard_after: float = 5.0) -> None:
    with _tracked_lock:
        snap = list(_tracked_processes)
    LOG.info("[Cleanup] terminating %d tracked SSH/apptainer processes", len(snap))
    threads = []
    for p, desc in snap:
        t = threading.Thread(target=_terminate_one,
                             args=(p, desc),
                             kwargs={"hard_after": hard_after},
                             daemon=True)
        t.start(); threads.append(t)
    for t in threads:
        t.join(timeout=hard_after + 2)


def install_signal_handlers() -> None:
    """Install SIGTERM/SIGINT handlers and an atexit hook that terminate
    every tracked SSH/apptainer Popen. Idempotent.
    """
    global _signal_handlers_installed
    if _signal_handlers_installed:
        return

    def _handler(signum, _frame):
        LOG.warning("[Signal] received signal %d — terminating all tracked workers",
                    signum)
        terminate_all_tracked(hard_after=5.0)
        # Re-raise default behavior so the parent shell sees the exit code.
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT,  _handler)
    atexit.register(lambda: terminate_all_tracked(hard_after=2.0))
    _signal_handlers_installed = True


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
    """Launch one robot worker on `node` as a tracked SSH+apptainer-exec child.

    Important behavior changes vs the previous version (do NOT revert):
      - Uses `apptainer exec` in the FOREGROUND of the SSH session. We do
        NOT call `apptainer instance start` anymore — that creates a
        persistent "Apptainer runtime parent" that survives the job.
      - The SSH process is a tracked Popen child of THIS runner. When the
        runner dies (clean exit, signal, or job termination) the SSH
        client dies, the remote sshd HUPs apptainer-exec, which kills the
        worker. No orphans.
      - The remote command uses `exec env ...` so apptainer-exec REPLACES
        the bash that sshd spawned, ensuring SIGHUP propagates straight to
        apptainer (bash by default doesn't forward HUP to children).
    """
    chk = f"{cfg.checkpoint_base}/{apptainer_instance_name(cid)}"
    log = f"{cfg.run_log_dir}/robot_{cid:03d}.log"

    pylibs_host = os.path.join(cfg.img_dir, "pylibs")
    cluster_root = os.environ.get("CLUSTER_ROOT") or \
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    worker_path = worker_script if worker_script.startswith("/") else f"/app/{worker_script}"

    env_pairs = {
        "REDIS_HOST":       cfg.redis_host,
        "REDIS_PORT":       cfg.redis_port,
        "MASTER_ADDRESS":   f"{cfg.server_node}:{cfg.flower_port}",
        "NUM_CLIENTS":      cfg.num_clients,
        "MIGRATION_OFFSET": cfg.migration_offset,
        "TOTAL_TASKS":      cfg.total_tasks,
        "TOTAL_FL_ROUNDS":  cfg.total_fl_rounds,
        "PYTHONUNBUFFERED": 1,
        "PYTHONPATH":       "/pylibs:/app:/robot_lib",
    }
    if extra_env:
        env_pairs.update(extra_env)
    env_prefix = " ".join(
        f"APPTAINERENV_{k}={shlex.quote(str(v))} "
        f"SINGULARITYENV_{k}={shlex.quote(str(v))}"
        for k, v in env_pairs.items()
    )

    # Foreground apptainer exec. `exec` makes apptainer the immediate child
    # of sshd so SIGHUP on connection close propagates without bash in the way.
    # Non-interactive SSH does not source ~/.bashrc, so apptainer (in the
    # conda env) is not on PATH on remote nodes. Source conda first.
    conda_base = os.environ.get("CONDA_BASE", "/home/029822154/miniconda3")
    conda_env = os.environ.get("CONDA_ENV", "swiftbot")
    # Workers do `sys.path.insert(0, "/app/robot")` then `import task_generator`
    # but the actual file lives at swiftbot_rl/dht_frl/robot/task_generator.py.
    # Binding to /app/robot fails ("file exists" inside the /app bind layer),
    # so bind it elsewhere and add to PYTHONPATH — Python finds it before the
    # broken sys.path.insert kicks in.
    robot_dir_host = os.path.join(cfg.swiftbot_root, "dht_frl", "robot")
    remote_cmd = (
        f"mkdir -p {shlex.quote(chk)}; "
        f"source {shlex.quote(conda_base)}/bin/activate {shlex.quote(conda_env)} && "
        f"exec env {env_prefix} apptainer exec "
        f"--bind {shlex.quote(cfg.swiftbot_root)}:/app "
        f"--bind {shlex.quote(robot_dir_host)}:/robot_lib "
        f"--bind {shlex.quote(cluster_root)}:/cluster_app "
        f"--bind {shlex.quote(chk)}:/checkpoints "
        f"--bind {shlex.quote(pylibs_host)}:/pylibs "
        f"{shlex.quote(cfg.img_dir + '/' + image)} "
        # NUM_CLIENTS passed via APPTAINERENV_NUM_CLIENTS, not CLI — the
        # frozen swiftbot_rl/cold_restart/worker_random_client.py doesn't
        # accept --num-clients, and our cluster worker has a default+env.
        f"python3 {shlex.quote(worker_path)} "
        f"--client-id {cid} "
        f"--container-type {shlex.quote(container_type)}"
    )

    log_fh = open(log, "ab", buffering=0)
    if is_local_node(node):
        # Local launch — apptainer exec runs as a direct child of this Python
        # process. No SSH needed (and on this cluster impossible). The Popen
        # is still tracked so cleanup terminates it.
        cmd = ["bash", "-c", remote_cmd]
    else:
        cmd = ["ssh", *_SSH_OPTS,
               "-o", "ServerAliveInterval=30",
               "-o", "ServerAliveCountMax=3",
               "-n", node, remote_cmd]
    p = subprocess.Popen(cmd,
                         stdin=subprocess.DEVNULL,
                         stdout=log_fh, stderr=log_fh)
    register_tracked_process(p, f"robot_{cid:03d}@{node}")
    with _tracked_lock:
        _robot_procs[cid] = (p, node)


def kill_robot(cfg: ClusterConfig, node: str, cid: int) -> None:
    """Stop a robot by terminating its tracked SSH Popen, then belt-and-suspenders
    pkill on the remote node to catch anything that didn't die from SIGHUP.
    """
    with _tracked_lock:
        pair = _robot_procs.pop(cid, None)
    if pair is not None:
        p, _node = pair
        _terminate_one(p, f"robot_{cid:03d}", hard_after=5.0)
    # Remote pkill — catches any worker whose parent SSH was already gone
    # (e.g. retry case) or that ignored SIGHUP for some reason.
    ssh_run(node,
            f"pkill -9 -u $USER -f 'client-id {cid} --num-clients' 2>/dev/null || true",
            timeout=15, check=False)


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


# NOTE: CRIU helpers (remote_criu_dump, remote_criu_restore,
# simulate_dump_seconds) and get_robot_pid were removed. CRIU is unavailable
# on this cluster; Conditions C/D now use application-level torch.save/load
# (see cluster/workers/worker_app_checkpoint.py) and don't need the host PID.
