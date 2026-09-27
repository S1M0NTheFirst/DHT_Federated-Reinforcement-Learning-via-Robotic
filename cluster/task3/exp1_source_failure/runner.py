"""
Exp1 runner (host side, on the server node): a task2-style FL fleet in which
every migration's source is killed at a chosen point. Reuses task2's Flower
server, worker env and SSH/tracked-process helpers; the Exp1 state machine is
in exp1_logic.run_migration, the real actions are in ClusterOps below.

Env (from run.sh): the task2/cluster vars (SERVER_NODE, CLIENT_NODE_1/2,
REDIS_*, FLOWER_PORT, NUM_CLIENTS, ...), CONDITION=exp1_<cond>,
EXP1_CONDITION, EXP1_* knobs, EXP1_DHT_HOSTS="node:port,node:port,node:port".
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time

import redis

HERE = os.path.dirname(os.path.abspath(__file__))
CLUSTER = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(CLUSTER, "task2", "common"))
sys.path.insert(0, CLUSTER)

from common.cluster_runner import (                     # noqa: E402
    ClusterConfig, install_signal_handlers, register_tracked_process,
    terminate_all_tracked, is_local_node, _ssh_opts_for_cid, _terminate_one,
    establish_ssh_master, close_ssh_master, close_all_ssh_masters_fast,
    live_status_loop, apptainer_instance_name,
)
from runner_base import (                               # noqa: E402
    _worker_env, current_node, update_node,
)
from exp1_logic import (                                # noqa: E402
    CONDITIONS, CHECKPOINT_MODE, FILES, Event, EventWriter, fill_recovery,
    kill_point_for, parse_kill_points, run_migration,
)

LOG = logging.getLogger("exp1_runner")
WORKER_PY = "/cluster_app/task3/exp1_source_failure/exp1_worker.py"
FLOWER_PY = "/cluster_app/task3/exp1_source_failure/exp1_flower_server.py"
WARMUP_PY = "/cluster_app/task3/exp1_source_failure/exp1_warmup.py"
# ssh used INSIDE copies (dst pulls from src / replica). known_hosts goes to
# /dev/null so ssh never reads or writes the NFS home for it; keep-alives and
# rsync --timeout end a stalled transfer instead of letting it hang.
_SSH_COPY = ("ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
             "-o LogLevel=ERROR -o BatchMode=yes -o ConnectTimeout=10 "
             "-o ServerAliveInterval=5 -o ServerAliveCountMax=3")

_procs: dict[int, subprocess.Popen] = {}
_procs_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# DHT control client (one short TCP connection per call: thread-safe).        #
# --------------------------------------------------------------------------- #
class DHTClient:
    def __init__(self, spec: str):
        self.hosts = []
        for s in spec.split(","):
            h, p = s.rsplit(":", 1)
            self.hosts.append((h, int(p)))

    def idx(self, node: str) -> int:
        short = node.split(".")[0]
        for i, (h, _) in enumerate(self.hosts):
            if h == node or h.split(".")[0] == short:
                return i
        raise KeyError(f"no DHT host for {node}")

    def call(self, i: int, req: dict, timeout: float = 120.0) -> dict:
        h, p = self.hosts[i]
        with socket.create_connection((h, p), timeout=timeout) as s:
            f = s.makefile("rwb")
            f.write((json.dumps(req) + "\n").encode())
            f.flush()
            line = f.readline()
        if not line:
            raise RuntimeError(f"DHT host {i} closed the connection")
        return json.loads(line)

    def wait_ready(self, timeout: float = 300.0) -> None:
        deadline = time.time() + timeout
        while True:
            pings = [self.call(i, {"cmd": "ping"}) for i in range(len(self.hosts))]
            errs = [p["error"] for p in pings if p.get("error")]
            if errs:
                raise RuntimeError(f"DHT ring failed to start: {errs}")
            if all(p["ready"] for p in pings):
                n = sum(len(p["nodes"]) for p in pings)
                LOG.info("[DHT] ring ready: %d nodes on %d hosts", n, len(pings))
                return
            if time.time() > deadline:
                raise RuntimeError("DHT ring not ready before timeout")
            time.sleep(1.0)

    def refresh(self, min_contacts: int) -> None:
        for i in range(len(self.hosts)):
            r = self.call(i, {"cmd": "refresh", "min_contacts": min_contacts})
            LOG.info("[DHT] host %d refreshed: min/median contacts %s/%s, "
                     "rejoined %s", i, r.get("min_contacts"),
                     r.get("median_contacts"), r.get("rejoined"))
            if int(r.get("min_contacts", 0)) < min_contacts:
                raise RuntimeError(f"DHT overlay not formed on host {i}: {r}")

    def put(self, key: str, value: str, via: str) -> dict:
        return self.call(self.idx(via), {"cmd": "put", "key": key, "value": value})

    def get(self, key: str, via: str) -> dict:
        return self.call(self.idx(via), {"cmd": "get", "key": key})

    def quit_all(self) -> None:
        for i in range(len(self.hosts)):
            try:
                self.call(i, {"cmd": "quit"}, timeout=10)
            except Exception:                            # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# SSH (plain, no ControlMaster) + robot launch / kill.                        #
# --------------------------------------------------------------------------- #
# Control commands and relaunches use a fresh ssh each time. The per-robot
# ControlMaster sockets hang on some of this cluster's nodes (see the
# establish_ssh_master note in common/cluster_runner.py); in the first Exp1
# run that turned single kill/copy steps into 40 s timeouts x retries.
_PLAIN_SSH = ["ssh", "-o", "StrictHostKeyChecking=no",
              "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
              "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
_CTRL_ALIVE = ["-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3"]


def _killpg(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except OSError:
        pass


def _run(argv: list, timeout: float) -> subprocess.CompletedProcess:
    """subprocess.run with a deadline that cannot be escaped. The first quick
    rerun lost 6 min in one `ssh ... stat` retry that never hit its 60 s
    subprocess timeout (run() can only enforce it inside communicate(); a hang
    while starting or reaping the process escapes it). Here the command runs in
    its own thread and process group: past the deadline the whole group is
    SIGKILLed and, if it still cannot be reaped, abandoned. rc 124 = timeout."""
    box: dict = {}

    def work() -> None:
        try:
            p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True,
                                 start_new_session=True)
            box["p"] = p
            try:
                out, err = p.communicate(timeout=timeout)
                box["res"] = subprocess.CompletedProcess(argv, p.returncode, out, err)
            except subprocess.TimeoutExpired:
                _killpg(p)
                try:
                    out, err = p.communicate(timeout=5)
                except Exception:                        # noqa: BLE001
                    out, err = "", ""
                box["res"] = subprocess.CompletedProcess(
                    argv, 124, out or "",
                    (err or "") + f" [timed out after {timeout:.0f}s]")
        except Exception as e:                           # noqa: BLE001
            box["res"] = subprocess.CompletedProcess(argv, 125, "", repr(e))

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout + 10)
    if t.is_alive():
        if "p" in box:
            _killpg(box["p"])
        return subprocess.CompletedProcess(
            argv, 124, "", f"[hung > {timeout + 10:.0f}s, abandoned]")
    return box["res"]


def _remote(command: str, timeout: float) -> str:
    # Also bound the command on the far side, so nothing outlives the deadline.
    return f"timeout -k 5 {max(5, int(timeout) - 2)} bash -c {shlex.quote(command)}"


def launch_exp1_robot(cfg: ClusterConfig, node: str, cid: int, tag: str,
                      extra_env: dict, mux: bool = True) -> subprocess.Popen:
    """Start robot `cid` on `node` as a tracked ssh(+apptainer) child. `mux`
    (initial fleet launch) goes through the pre-opened per-robot master, as in
    task2; relaunches use a fresh connection, retried if sshd refuses it."""
    chk = f"{cfg.checkpoint_base}/{apptainer_instance_name(cid)}"
    log = f"{cfg.run_log_dir}/robot_{cid:03d}.log"
    cluster_root = os.environ["CLUSTER_ROOT"]
    pylibs = os.path.join(cfg.img_dir, "pylibs")
    pylibs2 = os.environ["TASK2_PYLIBS2"]
    conda_base = os.environ["CONDA_BASE"]
    conda_env = os.environ.get("CONDA_ENV", "base")
    env = _worker_env(cfg, cid, {
        "EXP1_MAX_LIFETIME": os.environ.get("EXP1_MAX_LIFETIME", "21600"),
        "EXP1_SOURCE_GRACE": os.environ.get("EXP1_SOURCE_GRACE", "900"),
        "EXP1_PRECOPY_ROUNDS": (os.environ.get("EXP1_PRECOPY_ROUNDS", "1")
                                if os.environ.get("EXP1_CONDITION") == "app_warm" else "0"),
        **extra_env})
    env_prefix = " ".join(
        f"APPTAINERENV_{k}={shlex.quote(str(v))} "
        f"SINGULARITYENV_{k}={shlex.quote(str(v))}" for k, v in env.items())
    remote = (
        f"mkdir -p {shlex.quote(chk)}; "
        f"source {shlex.quote(conda_base)}/bin/activate {shlex.quote(conda_env)} && "
        f"exec env {env_prefix} apptainer exec "
        f"--bind {shlex.quote(cluster_root)}:/cluster_app "
        f"--bind {shlex.quote(chk)}:/checkpoints "
        f"--bind {shlex.quote(pylibs)}:/pylibs "
        f"--bind {shlex.quote(pylibs2)}:/pylibs2 "
        f"{shlex.quote(cfg.img_dir + '/robot.sif')} "
        f"python3 {WORKER_PY} --client-id {cid} --container-type cpu_specialist "
        f"--run-tag {tag}"
    )
    fh = open(log, "ab", buffering=0)
    for attempt in range(4):
        if is_local_node(node):
            cmd = ["bash", "-c", remote]
        elif mux:
            cmd = ["ssh", *_ssh_opts_for_cid(cid), "-o", "ServerAliveInterval=30",
                   "-o", "ServerAliveCountMax=3", "-n", node, remote]
        else:
            cmd = [*_PLAIN_SSH, "-o", "ServerAliveInterval=30",
                   "-o", "ServerAliveCountMax=3", "-n", node, remote]
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=fh, stderr=fh)
        # sshd drops bursts of connections at once (exit 255): relaunch now
        # instead of discovering it only when the probe times out.
        try:
            rc = p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            rc = None
        if rc != 255:
            break
        LOG.warning("[Launch] robot %03d: ssh to %s refused (attempt %d)", cid,
                    node, attempt + 1)
        time.sleep(2 * (attempt + 1))
    register_tracked_process(p, f"exp1_robot_{cid:03d}@{node}")
    with _procs_lock:
        _procs[cid] = p
    return p


# Copy errors that mean "this copy is gone" (as opposed to a flaky connection).
_MISSING = ("No such file", "vanished", "not found")


_RETRY_RC = (255, 124, 125)      # ssh failed / timed out / could not start


def run_on(node: str, cid: int, command: str, timeout: int = 30,
           tries: int = 3) -> subprocess.CompletedProcess:
    """Run `command` on `node` (plain bash if it is this node) with a hard
    deadline per attempt. ssh failures (255), timeouts (124) and start errors
    are retried quickly; anything else is the command's own result. Worst case
    per call: tries x (timeout + 10 s)."""
    rr = subprocess.CompletedProcess(command, 255, "", "not run")
    for attempt in range(tries):
        if is_local_node(node):
            argv = ["bash", "-c", _remote(command, timeout)]
        else:
            argv = [*_PLAIN_SSH, *_CTRL_ALIVE, "-n", node, _remote(command, timeout)]
        rr = _run(argv, timeout)
        if rr.returncode not in _RETRY_RC:
            return rr
        LOG.warning("[ssh] %s rc=%s (attempt %d/%d): %s | %s", node.split(".")[0],
                    rr.returncode, attempt + 1, tries, command[:70].replace("\n", " "),
                    (rr.stderr or "").strip()[-120:])
        time.sleep(1 + attempt)
    return rr


def run_copy(node: str, cid: int, command: str, what: str,
             timeout: int = 90) -> subprocess.CompletedProcess:
    """A copy pulled by `node`: up to 2 attempts of at most `timeout` s each.
    Any failure is retried (ssh refused/timeout, rsync stream error 12 or I/O
    timeout 30, ...) unless the source copy is really gone, which fails at once."""
    rr = subprocess.CompletedProcess(command, 255, "", "not run")
    for attempt in range(2):
        rr = run_on(node, cid, command, timeout=timeout, tries=1)
        if rr.returncode == 0 or any(m in (rr.stderr or "") for m in _MISSING):
            return rr
        LOG.warning("[copy] %s failed rc=%s (attempt %d/2): %s", what, rr.returncode,
                    attempt + 1, (rr.stderr or "").strip()[-160:])
        time.sleep(1 + attempt)
    return rr


# Find and SIGKILL robot <cid>'s worker python, then verify it is gone.
# A candidate (from pgrep, or the PID the worker published itself) must have
# argv = <python*> <.../exp1_worker.py> --client-id <cid> ... --run-tag <tag>
# exactly, so neither apptainer (whose argv repeats the worker args; killing it
# orphans its FUSE mounts) nor any other process that merely mentions the
# worker can be selected. The first run's pattern was anchored at "^python3 "
# and matched nothing, and "nothing left" was taken as success: every source
# survived. Here "no candidate" is exit 2 and "still alive" exit 1. A zombie
# (killed, not yet reaped by apptainer) counts as gone.
_KILL_SH = r'''cid=__CID__; tag=__TAG__; hint="__HINT__"
set -f
cands=""
for p in $(pgrep -u "$(id -u)" -f "exp1_worker[.]py --client-id $cid ") $hint; do
  [ -r /proc/$p/cmdline ] || continue
  set -- $(tr '\0' ' ' < /proc/$p/cmdline)
  case "$(basename "${1:-x}")" in python*) ;; *) continue ;; esac
  case "${2:-}" in *exp1_worker.py) ;; *) continue ;; esac
  [ "${3:-}" = --client-id ] && [ "${4:-}" = "$cid" ] || continue
  case " $* " in *" --run-tag $tag "*) cands="$cands $p" ;; esac
done
cands=$(for p in $cands; do echo $p; done | sort -u | tr '\n' ' ')
[ -z "$cands" ] && { echo NOTFOUND; exit 2; }
kill -9 $cands 2>/dev/null
i=0
while [ $i -lt 50 ]; do
  alive=""
  for p in $cands; do
    if [ -d /proc/$p ] && ! grep -q '^State:[[:space:]]*Z' /proc/$p/status 2>/dev/null; then
      alive="$alive $p"
    fi
  done
  [ -z "$alive" ] && { echo "KILLED $cands"; exit 0; }
  kill -9 $alive 2>/dev/null; sleep 0.2; i=$((i+1))
done
echo "ALIVE $alive"; exit 1
'''


def kill_script(cid: int, tag: str, hint_pid: str = "") -> str:
    hint = hint_pid if hint_pid.isdigit() else ""
    return (_KILL_SH.replace("__CID__", str(int(cid)))
            .replace("__TAG__", shlex.quote(tag)).replace("__HINT__", hint))


def _reap_later(p: subprocess.Popen, desc: str) -> None:
    """The dead worker's apptainer/ssh parent normally exits within seconds;
    never block a migration on it (the first run lost ~20 s per kill here)."""
    def reap():
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            _terminate_one(p, desc, hard_after=5.0)
    threading.Thread(target=reap, daemon=True).start()


def sigkill_worker(node: str, cid: int, tag: str, r=None) -> bool:
    """SIGKILL robot `cid`'s worker on `node`; True once it is verified gone."""
    hint = ""
    if r is not None:
        try:
            info = json.loads(r.get(f"exp1_pid:robot_{cid:03d}") or "{}")
            if info.get("host", "").split(".")[0] == node.split(".")[0]:
                hint = str(info.get("pid", ""))
        except (ValueError, TypeError):
            pass
    rr = run_on(node, cid, kill_script(cid, tag, hint), timeout=25)
    out = (rr.stdout or "").strip()
    ok = rr.returncode == 0
    if ok:
        LOG.info("[Kill] robot %03d on %s: %s", cid, node.split(".")[0], out)
    else:
        LOG.error("[Kill] robot %03d on %s FAILED rc=%s: %s %s", cid, node, rr.returncode,
                  out[-200:], (rr.stderr or "").strip()[-160:])
    with _procs_lock:
        p = _procs.pop(cid, None)
    if p is not None:
        _reap_later(p, f"exp1_robot_{cid:03d}")
    return ok


def warm_up(cfg: ClusterConfig, nodes: list, timeout: int = 600) -> None:
    """Run exp1_warmup.py in the robot container on every node, in parallel,
    so no condition pays the cold first reads of the image and libraries."""
    cluster_root = os.environ["CLUSTER_ROOT"]
    env = _worker_env(cfg, 0, {})
    env_prefix = " ".join(
        f"APPTAINERENV_{k}={shlex.quote(str(v))} "
        f"SINGULARITYENV_{k}={shlex.quote(str(v))}" for k, v in env.items())
    cmd = (
        f"source {shlex.quote(os.environ['CONDA_BASE'])}/bin/activate "
        f"{shlex.quote(os.environ.get('CONDA_ENV', 'base'))} && "
        f"env {env_prefix} apptainer exec "
        f"--bind {shlex.quote(cluster_root)}:/cluster_app "
        f"--bind {shlex.quote(os.path.join(cfg.img_dir, 'pylibs'))}:/pylibs "
        f"--bind {shlex.quote(os.environ['TASK2_PYLIBS2'])}:/pylibs2 "
        f"{shlex.quote(cfg.img_dir + '/robot.sif')} python3 {WARMUP_PY}")
    res: dict = {}

    def one(n: str) -> None:
        t = time.time()
        rr = run_on(n, 0, cmd, timeout=timeout, tries=1)
        res[n] = (time.time() - t, rr)

    t0 = time.time()
    threads = [threading.Thread(target=one, args=(n,), daemon=True) for n in nodes]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout + 30)
    for n in nodes:
        dt, rr = res.get(n, (float("nan"), None))
        tail = ((rr.stdout or "") + (rr.stderr or "")).strip().splitlines()[-1:] if rr else []
        LOG.info("[Warm-up] %s: %.1fs rc=%s %s", n.split(".")[0], dt,
                 rr.returncode if rr else "?", tail[0][-120:] if tail else "")
    LOG.info("[Warm-up] all nodes in %.1fs", time.time() - t0)


def launch_exp1_flower(cfg: ClusterConfig) -> subprocess.Popen:
    """task2's launch_task2_flower, but running the Exp1 server (task2 server +
    round_timeout + ghost pruning) and passing EXP1_ROUND_TIMEOUT."""
    cluster_root = os.environ["CLUSTER_ROOT"]
    pylibs = os.path.join(cfg.img_dir, "pylibs")
    pylibs2 = os.environ["TASK2_PYLIBS2"]
    conda_base = os.environ["CONDA_BASE"]
    conda_env = os.environ.get("CONDA_ENV", "base")
    rel = os.path.relpath(cfg.results_dir, cluster_root)
    env = {
        "N_CLIENTS": cfg.num_clients, "N_ROUNDS": cfg.total_fl_rounds,
        "FLOWER_BIND": f"0.0.0.0:{cfg.flower_port}",
        "FL_RESULT_DIR": f"/cluster_app/{rel}".replace(os.sep, "/"),
        "REDIS_HOST": cfg.redis_host, "REDIS_PORT": cfg.redis_port,
        "SHARED_SEED": os.environ.get("SHARED_SEED", "12345"),
        "EXP1_ROUND_TIMEOUT": os.environ.get("EXP1_ROUND_TIMEOUT", "150"),
        "PYTHONUNBUFFERED": 1,
        "PYTHONPATH": "/pylibs2:/pylibs:/cluster_app/task2/worker",
        "CUDA_VISIBLE_DEVICES": "",
    }
    env_prefix = " ".join(
        f"APPTAINERENV_{k}={shlex.quote(str(v))} "
        f"SINGULARITYENV_{k}={shlex.quote(str(v))}" for k, v in env.items())
    remote = (
        f"source {shlex.quote(conda_base)}/bin/activate {shlex.quote(conda_env)} && "
        f"exec env {env_prefix} apptainer exec "
        f"--bind {shlex.quote(cluster_root)}:/cluster_app "
        f"--bind {shlex.quote(pylibs)}:/pylibs "
        f"--bind {shlex.quote(pylibs2)}:/pylibs2 "
        f"{shlex.quote(cfg.img_dir + '/robot.sif')} "
        f"python3 {FLOWER_PY}"
    )
    flog = open(f"{cfg.run_log_dir}/flower_server.log", "ab", buffering=0)
    if is_local_node(cfg.server_node):
        cmd = ["bash", "-c", remote]
    else:
        cmd = [*_PLAIN_SSH, "-o", "ServerAliveInterval=30",
               "-o", "ServerAliveCountMax=3", "-n", cfg.server_node, remote]
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=flog, stderr=flog)
    register_tracked_process(p, "exp1_flower_server")
    return p


# --------------------------------------------------------------------------- #
# app_warm background pre-copy.                                               #
# --------------------------------------------------------------------------- #
class PreCopies:
    """The worker announces a pre-copy bundle (exp1_precopy:<robot>) one or
    more rounds before its migration; a background thread pulls it from the
    robot's node to the node it will migrate to. Not part of the downtime (the
    robot keeps training meanwhile); its time and bytes are recorded."""

    def __init__(self, cfg, wait_s: float):
        self.cfg, self.wait_s = cfg, wait_s
        self.state: dict = {}              # robot_id -> dict (+ "done" Event)
        self.lock = threading.Lock()

    def start(self, info: dict) -> None:
        rid = info["robot_id"]
        cid = int(rid.split("_")[1])
        src = current_node(rid)
        dst = self.cfg.other_node(src)
        d = f"{self.cfg.checkpoint_base}/{apptainer_instance_name(cid)}/{rid}.pre"
        st = {"for_round": int(info.get("for_round", -1)),
              "round": int(info.get("round", -1)), "dst": dst, "dir": d,
              "replay_entries": int(info.get("replay_entries", -1)),
              "ok": False, "ms": 0.0, "bytes": 0, "done": threading.Event()}
        with self.lock:
            self.state[rid] = st

        def work() -> None:
            t = time.perf_counter()
            q = shlex.quote(d)
            cmd = (f"rm -rf {q} && mkdir -p {q} && rsync -a --timeout=30 -e "
                   f"'{_SSH_COPY}' {src}:{q}/ {q}/ && du -sb {q} | cut -f1")
            rr = run_copy(dst, cid, cmd, f"{rid} pre-copy -> {dst.split('.')[0]}")
            st["ms"] = (time.perf_counter() - t) * 1000
            if rr.returncode == 0:
                st["ok"] = True
                try:
                    st["bytes"] = int((rr.stdout or "0").strip().splitlines()[-1])
                except (ValueError, IndexError):
                    pass
            LOG.info("[%s] pre-copy of round %s -> %s: ok=%s %.1fs", rid, st["round"],
                     dst.split(".")[0], st["ok"], st["ms"] / 1000)
            st["done"].set()

        threading.Thread(target=work, daemon=True).start()

    def result(self, rid: str, fl_round: int, dst: str):
        """The pre-copy made for THIS migration (same round, same destination);
        waits up to wait_s if it is still running, else counts as missing."""
        with self.lock:
            st = self.state.get(rid)
        if st is None or st["for_round"] != fl_round or st["dst"] != dst:
            return None
        if not st["done"].wait(self.wait_s):
            LOG.warning("[%s] pre-copy still running at the migration: not used", rid)
            return {k: v for k, v in st.items() if k != "done"} | {"ok": False}
        return {k: v for k, v in st.items() if k != "done"}


# --------------------------------------------------------------------------- #
# Real actions for exp1_logic.run_migration.                                  #
# --------------------------------------------------------------------------- #
class ClusterOps:
    def __init__(self, cfg, r, dht, ev: Event, tag: str, probe_timeout: float,
                 files: dict | None = None, precopy=None):
        self.cfg, self.r, self.dht, self.ev = cfg, r, dht, ev
        self.tag, self.probe_timeout = tag, probe_timeout
        self.cid = ev.cid
        self.files = files or {}      # bundle file sizes the worker reported
        self.precopy = precopy        # PreCopies (app_warm only)

    def precopy_status(self):
        if self.precopy is None:
            return None
        return self.precopy.result(self.ev.robot_id, self.ev.fl_round, self.ev.dst)

    def seed_dir(self, node: str, from_dir: str, to_dir: str) -> bool:
        """to_dir := a local copy of from_dir on `node` (the rsync delta basis)."""
        a, b = shlex.quote(from_dir), shlex.quote(to_dir)
        rr = run_on(node, self.cid, f"test -d {a} && rm -rf {b} && cp -a {a} {b}")
        return rr.returncode == 0

    def promote_dir(self, node: str, from_dir: str, to_dir: str) -> bool:
        """to_dir := from_dir (rename, so the bundle dir only ever holds a
        complete bundle)."""
        a, b = shlex.quote(from_dir), shlex.quote(to_dir)
        rr = run_on(node, self.cid, f"test -d {a} && rm -rf {b} && mv {a} {b}")
        return rr.returncode == 0

    now = staticmethod(time.perf_counter)

    def wait_saved(self) -> None:
        key = f"ready_for_criu:{self.ev.robot_id}"
        deadline = time.time() + 120
        while time.time() < deadline:
            if self.r.get(key):
                self.r.delete(key)
                return
            time.sleep(0.2)
        LOG.warning("[%s] bundle-saved flag never appeared", self.ev.robot_id)

    def stat_files(self, node: str, d: str) -> dict:
        # The worker reports its bundle's file sizes with the migration request;
        # only an old/odd request costs an extra ssh stat on the source.
        if all(f in self.files for f in FILES):
            return {f: int(self.files[f]) for f in FILES}
        rr = run_on(node, self.cid, f"cd {shlex.quote(d)} 2>/dev/null && "
                                    f"stat -c '%n %s' {' '.join(FILES)} 2>/dev/null")
        sizes = {}
        for line in (rr.stdout or "").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] in FILES:
                sizes[parts[0]] = int(parts[1])
        return sizes

    def copy_dir(self, src: str, src_dir: str, node: str, dst_dir: str) -> bool:
        q = shlex.quote(dst_dir)
        if src == node:
            copy = f"cp -a {shlex.quote(src_dir)}/. {q}/"
        else:
            copy = f"rsync -a --timeout=30 -e '{_SSH_COPY}' {src}:{shlex.quote(src_dir)}/ {q}/"
        rr = run_copy(node, self.cid, f"rm -rf {q} && mkdir -p {q} && {copy}",
                      f"{self.ev.robot_id} replica -> {node}")
        if rr.returncode != 0:
            LOG.warning("[%s] replica copy to %s failed: %s", self.ev.robot_id,
                        node, (rr.stderr or "").strip()[-200:])
        return rr.returncode == 0

    def copy_file(self, src: str, src_path: str, dst: str, dst_path: str,
                  expect, transport: str) -> bool:
        q_src, q_dst = shlex.quote(src_path), shlex.quote(dst_path)
        parent = shlex.quote(os.path.dirname(dst_path))
        if src == dst or (is_local_node(src) and is_local_node(dst)):
            copy = f"cp {q_src} {q_dst}"
        elif transport == "scp":
            copy = (f"scp -q -S ssh {_SSH_COPY[4:]} {src}:{q_src} {q_dst}")
        else:
            copy = f"rsync -a --timeout=30 -e '{_SSH_COPY}' {src}:{q_src} {q_dst}"
        rr = run_copy(dst, self.cid,
                      f"mkdir -p {parent} && {copy} && stat -c %s {q_dst}",
                      f"{self.ev.robot_id} {os.path.basename(src_path)} "
                      f"{src.split('.')[0]}->{dst.split('.')[0]}")
        if rr.returncode != 0:
            return False
        try:
            size = int((rr.stdout or "").strip().splitlines()[-1])
        except (ValueError, IndexError):
            return False
        return size > 0 and (expect is None or size == int(expect))

    def dht_put(self, key: str, value: str, via: str) -> dict:
        try:
            return self.dht.put(key, value, via)
        except Exception as e:                           # noqa: BLE001
            LOG.error("[%s] DHT put failed: %r", self.ev.robot_id, e)
            return {"ok": 0, "ms": 0.0}

    def dht_get(self, key: str, via: str) -> dict:
        try:
            return self.dht.get(key, via)
        except Exception as e:                           # noqa: BLE001
            LOG.error("[%s] DHT get failed: %r", self.ev.robot_id, e)
            return {"ok": 0, "ms": 0.0}

    def kill_source(self) -> bool:
        ev = self.ev
        LOG.info("[%s] KILL source %s (%s)", ev.robot_id, ev.src, ev.kill_point)
        ok = sigkill_worker(ev.src, ev.cid, self.tag, self.r)
        # Tell the Flower server: this robot's proxies from before now are dead.
        self.r.set(f"exp1_killed:{ev.robot_id}", str(time.time()), ex=86400)
        # The source's local storage is lost with it (bundle and any pre-copy).
        run_on(ev.src, ev.cid, f"rm -rf {shlex.quote(ev.src_bundle)} "
                               f"{shlex.quote(ev.src_bundle + '.pre')}")
        return ok

    def kill_dst(self) -> None:
        sigkill_worker(self.ev.dst, self.ev.cid, self.tag, self.r)
        self.r.set(f"exp1_killed:{self.ev.robot_id}", str(time.time()), ex=86400)

    def clear_dir(self, node: str, d: str) -> None:
        run_on(node, self.cid, f"rm -rf {shlex.quote(d)}")

    def launch(self, restore: bool) -> None:
        ev = self.ev
        self.r.delete(f"probe_metrics:{ev.robot_id}")
        env = {"EXP1_RELAUNCH": "1"}
        if restore:
            env["RESTORE_DIR"] = f"/checkpoints/{ev.robot_id}"
        LOG.info("[%s] relaunch on %s (%s)", ev.robot_id, ev.dst,
                 "restore" if restore else "cold")
        self.proc = launch_exp1_robot(self.cfg, ev.dst, ev.cid, self.tag, env,
                                      mux=False)

    def wait_probe(self):
        key = f"probe_metrics:{self.ev.robot_id}"
        deadline = time.time() + self.probe_timeout
        while time.time() < deadline:
            p = getattr(self, "proc", None)
            if p is not None and p.poll() is not None and not self.r.get(key):
                LOG.warning("[%s] relaunched worker exited rc=%s before reporting",
                            self.ev.robot_id, p.returncode)
                return None
            raw = self.r.get(key)
            if raw:
                self.r.delete(key)
                try:
                    return json.loads(raw)
                except ValueError:
                    return {}
            time.sleep(0.5)
        return None


TASK_LOG_FIELDS = ["robot_id", "fl_round", "training_step", "reward",
                   "success_rate_rolling10", "policy_entropy", "status",
                   "eval_return", "eval_episode_len", "eval_success"]


def snapshot_task_logs(r, path: str) -> None:
    """Same rows/columns as the Flower server's final task_logs.csv, written
    atomically; the figure script falls back to it if the run was cut short."""
    import csv
    try:
        rows = []
        for item in r.lrange("task_logs", 0, -1):
            try:
                rows.append(json.loads(item))
            except ValueError:
                continue
        if not rows:
            return
        rows.sort(key=lambda e: (e.get("robot_id", ""), e.get("fl_round", 0)))
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TASK_LOG_FIELDS)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in TASK_LOG_FIELDS})
        os.replace(tmp, path)
    except Exception as e:                               # noqa: BLE001
        LOG.warning("[Wait] task_logs snapshot failed: %r", e)


# --------------------------------------------------------------------------- #
# Orchestrator.                                                               #
# --------------------------------------------------------------------------- #
def main() -> int:
    install_signal_handlers()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    cond = os.environ["EXP1_CONDITION"]
    spec = CONDITIONS[cond]
    tag = os.environ["CONDITION"]                    # exp1_<cond>, unique per condition
    points = parse_kill_points(os.environ.get("EXP1_KILL_POINTS",
                                              "after_save,mid_transfer,after_transfer"))
    n_rep = int(os.environ.get("EXP1_REPLICAS", "1")) if spec.replicas else 0
    excl = os.environ.get("EXP1_REPLICA_EXCLUDE_DST", "1") == "1"
    probe_timeout = float(os.environ.get("EXP1_PROBE_TIMEOUT", "300"))

    cfg = ClusterConfig()
    nodes = [cfg.server_node, *cfg.client_nodes]
    LOG.info("=" * 78)
    LOG.info("Exp1 %s — server=%s clients=%s robots=%d rounds=%d kill_points=%s "
             "replicas=%d exclude_dst=%s", cond, cfg.server_node, cfg.client_nodes,
             cfg.num_clients, cfg.total_fl_rounds, ",".join(points), n_rep, excl)
    LOG.info("=" * 78)

    os.makedirs(cfg.results_dir, exist_ok=True)
    with open(os.path.join(cfg.results_dir, "exp1_meta.json"), "w") as f:
        json.dump({"condition": cond, "spec": spec.__dict__, "kill_points": points,
                   "replicas": n_rep, "exclude_dst": excl, "nodes": nodes,
                   "num_clients": cfg.num_clients,
                   "rounds": cfg.total_fl_rounds,
                   "migration_rounds": os.environ.get("MIGRATION_ROUNDS"),
                   "steps_per_round": os.environ.get("STEPS_PER_ROUND"),
                   "dht_nodes_per_host": os.environ.get("EXP1_DHT_NODES_PER_HOST"),
                   "dht_ksize": os.environ.get("EXP1_DHT_KSIZE"),
                   "precopy_rounds": (os.environ.get("EXP1_PRECOPY_ROUNDS", "1")
                                      if spec.precopy else 0),
                   "started": time.time()}, f, indent=1)

    r = redis.Redis(host=cfg.redis_host, port=cfg.redis_port, decode_responses=True)
    r.flushall()
    writer = EventWriter(cond, cfg.results_dir)

    dht = None
    if spec.dht:
        dht = DHTClient(os.environ["EXP1_DHT_HOSTS"])
        dht.wait_ready()
        k = int(os.environ.get("EXP1_DHT_KSIZE", "3"))
        dht.refresh(min(k, 3 * int(os.environ.get("EXP1_DHT_NODES_PER_HOST", "8")) - 1))

    if os.environ.get("EXP1_WARMUP", "1") == "1":
        warm_up(cfg, nodes)
    flower = launch_exp1_flower(cfg)
    time.sleep(15)
    for cid in range(cfg.num_clients):
        for cn in cfg.client_nodes:
            establish_ssh_master(cn, cid)
            time.sleep(1)
    for cid in range(cfg.num_clients):
        node = cfg.home_node_for_client(cid)
        update_node(f"robot_{cid:03d}", node)
        launch_exp1_robot(cfg, node, cid, tag, {"EXP1_RELAUNCH": "0"}, mux=True)
        time.sleep(0.5)
    LOG.info("All %d robots launched.", cfg.num_clients)

    precopies = PreCopies(cfg, float(os.environ.get("EXP1_PRECOPY_WAIT", "30")))
    mig_count: dict[str, int] = {}
    inflight: set[str] = set()
    lock = threading.Lock()

    def handle(rid: str, info: dict, concurrency: int) -> None:
        cid = int(rid.split("_")[1])
        with lock:
            k = mig_count.get(rid, 0)
            mig_count[rid] = k + 1
        src = current_node(rid)
        dst = cfg.other_node(src)
        inst = apptainer_instance_name(cid)
        bundle = f"{cfg.checkpoint_base}/{inst}/{rid}"
        ev = Event(robot_id=rid, cid=cid, migration_index=k, src=src, dst=dst,
                   nodes=nodes, kill_point=kill_point_for(cid, k, points),
                   src_bundle=bundle, dst_bundle=bundle,
                   replica_dir=lambda n: f"{cfg.checkpoint_base}/replicas/{rid}",
                   n_replicas=n_rep, exclude_dst=excl, ns=tag,
                   fl_round=int(info.get("fl_round", 0)),
                   replay_entries_pre=int(info.get("replay_entries", -1)))
        ops = ClusterOps(cfg, r, dht, ev, tag, probe_timeout, info.get("files"),
                         precopies if spec.precopy else None)
        LOG.info("[%s] migration #%d round %s %s -> %s kill_point=%s", rid, k,
                 info.get("fl_round"), src, dst, ev.kill_point)
        try:
            m = run_migration(ev, spec, ops)
        except Exception as e:                           # noqa: BLE001
            import traceback
            LOG.error("[%s] migration failed in runner: %r\n%s", rid, e,
                      traceback.format_exc())
            # The robot must come back somewhere or FedAvg (min_fit_clients =
            # all robots) stalls: make sure the source is gone, start cold.
            m = {"robot_id": rid, "src_node": src, "dst_node": dst,
                 "kill_point": ev.kill_point, "migration_index": k,
                 "restore_error": f"runner exception: {e!r}"[:200]}
            try:
                ops.kill_source()
                ops.kill_dst()          # never two copies of one robot
                ops.clear_dir(dst, bundle)
                ops.launch(False)
                p = ops.wait_probe()
                m["robot_back"] = int(p is not None)
            except Exception as e2:                      # noqa: BLE001
                LOG.error("[%s] cold fallback failed too: %r", rid, e2)
        update_node(rid, dst)
        m.update(fl_round=int(info.get("fl_round", 0)),
                 checkpoint_mode=CHECKPOINT_MODE[cond],
                 concurrency_level=concurrency,
                 success_rate_pre=float(info.get("success_rate", 0)),
                 eval_return_pre=float(info.get("eval_return_pre", 0)),
                 # filled from task_logs when the run ends (fill_recovery)
                 success_rate_post="", regression_pct="", fl_rounds_to_recover="")
        writer.write_event(m)
        LOG.info("[%s] done: restored=%s served_by=%s downtime=%.1fs", rid,
                 m.get("restored"), m.get("served_by"),
                 float(m.get("downtime_ms", 0) or 0) / 1000)

    def run_handle(rid, info, n):
        try:
            handle(rid, info, n)
        finally:
            with lock:
                inflight.discard(rid)

    def monitor() -> None:
        LOG.info("[Monitor] watching migration requests (concurrent)")
        while True:
            try:
                for key in r.keys("migration_request:robot_*"):
                    raw = r.get(key)
                    if not raw:
                        continue
                    info = json.loads(raw)
                    rid = info["robot_id"]
                    with lock:
                        if rid in inflight:
                            continue
                        inflight.add(rid)
                        n = len(inflight)
                    r.delete(key)
                    threading.Thread(target=run_handle, args=(rid, info, n),
                                     daemon=True).start()
                if spec.precopy:
                    for key in r.keys("exp1_precopy:robot_*"):
                        raw = r.get(key)
                        r.delete(key)
                        if raw:
                            precopies.start(json.loads(raw))
            except Exception as e:                       # noqa: BLE001
                LOG.error("[Monitor] %r", e)
            time.sleep(0.5)

    threading.Thread(target=monitor, daemon=True).start()
    threading.Thread(target=live_status_loop, args=(cfg, r, writer, 30),
                     daemon=True).start()
    # FedAvg waits for ALL robots, so a robot that never comes back stalls the
    # run forever. Stop when no eval row has arrived for EXP1_STALL_TIMEOUT s
    # (finished events are already on disk), and keep a task_logs snapshot:
    # Flower only writes task_logs.csv when it finishes normally.
    stall_s = float(os.environ.get("EXP1_STALL_TIMEOUT", "1800"))
    partial = os.path.join(cfg.results_dir, "task_logs.partial.csv")
    n_last, t_change, t_snap, stalled = -1, time.time(), 0.0, False
    try:
        while True:
            done = sum(1 for cid in range(cfg.num_clients)
                       if r.get(f"robot_done:robot_{cid:03d}"))
            if done >= cfg.num_clients:
                LOG.info("[Wait] all robots done.")
                break
            if flower.poll() is not None:
                LOG.info("[Wait] Flower exited rc=%s — FL complete.", flower.returncode)
                time.sleep(10)
                break
            n = r.llen("task_logs")
            if n != n_last:
                n_last, t_change = n, time.time()
            elif time.time() - t_change > stall_s:
                LOG.error("[Wait] FL STALLED: no new eval results for %.0fs (a robot "
                          "never came back?). Stopping; finished results are kept.",
                          stall_s)
                stalled = True
                break
            if time.time() - t_snap > 60:
                snapshot_task_logs(r, partial)
                t_snap = time.time()
            time.sleep(15)
    finally:
        snapshot_task_logs(r, partial)
        terminate_all_tracked(hard_after=5.0)
        for cid in range(cfg.num_clients):
            for cn in cfg.client_nodes:
                close_ssh_master(cn, cid)
        time.sleep(2)
        close_all_ssh_masters_fast()
        # The DHT ring is NOT stopped here: in `all` mode the next condition
        # reuses it; run.sh stops the ring hosts when the job ends.
        try:
            rows = []
            for x in r.lrange("task_logs", 0, -1):
                try:
                    rows.append(json.loads(x))
                except ValueError:
                    continue
            n = fill_recovery(writer.path, rows)
            LOG.info("recovery columns filled for %d events", n)
        except Exception as e:                           # noqa: BLE001
            LOG.warning("could not fill recovery columns: %r", e)
    LOG.info("Migration events: %d -> %s", writer.event_count, writer.path)
    return 1 if stalled else 0


if __name__ == "__main__":
    sys.exit(main())
