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
    _worker_env, launch_task2_flower, current_node, update_node,
)
from exp1_logic import (                                # noqa: E402
    CONDITIONS, CHECKPOINT_MODE, FILES, Event, EventWriter, kill_point_for,
    parse_kill_points, run_migration,
)

LOG = logging.getLogger("exp1_runner")
WORKER_PY = "/cluster_app/task3/exp1_source_failure/exp1_worker.py"
_SSH_COPY = "ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10"

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
# Robot launch / kill.                                                        #
# --------------------------------------------------------------------------- #
def worker_pattern(cid: int, tag: str) -> str:
    # Anchored at python3 so it never matches apptainer (whose argv repeats the
    # worker args) — SIGKILLing apptainer would orphan its squashfuse mounts.
    return (f"^python3 [^ ]*exp1_worker[.]py --client-id {cid} "
            f"--container-type [^ ]+ --run-tag {tag}$")


def launch_exp1_robot(cfg: ClusterConfig, node: str, cid: int, tag: str,
                      extra_env: dict) -> None:
    chk = f"{cfg.checkpoint_base}/{apptainer_instance_name(cid)}"
    log = f"{cfg.run_log_dir}/robot_{cid:03d}.log"
    cluster_root = os.environ["CLUSTER_ROOT"]
    pylibs = os.path.join(cfg.img_dir, "pylibs")
    pylibs2 = os.environ["TASK2_PYLIBS2"]
    conda_base = os.environ["CONDA_BASE"]
    conda_env = os.environ.get("CONDA_ENV", "base")
    env = _worker_env(cfg, cid, {
        "EXP1_MAX_LIFETIME": os.environ.get("EXP1_MAX_LIFETIME", "21600"),
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
    if is_local_node(node):
        cmd = ["bash", "-c", remote]
    else:
        cmd = ["ssh", *_ssh_opts_for_cid(cid), "-o", "ServerAliveInterval=30",
               "-o", "ServerAliveCountMax=3", "-n", node, remote]
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=fh, stderr=fh)
    register_tracked_process(p, f"exp1_robot_{cid:03d}@{node}")
    with _procs_lock:
        _procs[cid] = p


# Copy errors that mean "this copy is gone" (as opposed to a flaky connection).
_MISSING = ("No such file", "vanished", "not found")


def run_on(node: str, cid: int, command: str, timeout: int = 60,
           tries: int = 4) -> subprocess.CompletedProcess:
    """Run a command on `node` through robot `cid`'s SSH master (plain bash if
    it is this node). Exit 255 means ssh itself failed (sshd drops bursts of
    connections) -> retry, so a flaky connection is never recorded as a dead
    source or a lost copy."""
    rr = subprocess.CompletedProcess(command, 255, "", "not run")
    for attempt in range(tries):
        try:
            if is_local_node(node):
                return subprocess.run(["bash", "-c", command], capture_output=True,
                                      text=True, timeout=timeout)
            rr = subprocess.run(["ssh", *_ssh_opts_for_cid(cid), "-n", node, command],
                                capture_output=True, text=True, timeout=timeout)
            if rr.returncode != 255:
                return rr
            LOG.warning("[ssh] %s refused (attempt %d): %s", node, attempt + 1,
                        (rr.stderr or "").strip()[-120:])
        except subprocess.TimeoutExpired:
            LOG.warning("[ssh] %s timed out (attempt %d): %s", node, attempt + 1,
                        command[:80])
            rr = subprocess.CompletedProcess(command, 255, "", "timeout")
        time.sleep(2 * (attempt + 1))
    return rr


def run_copy(node: str, cid: int, command: str, what: str,
             timeout: int = 300) -> subprocess.CompletedProcess:
    """A copy pulled by `node`; retried unless the source copy is really gone."""
    rr = run_on(node, cid, command, timeout=timeout)
    for attempt in range(2):
        if rr.returncode == 0 or any(m in (rr.stderr or "") for m in _MISSING):
            return rr
        LOG.warning("[copy] %s failed rc=%s (retry %d): %s", what, rr.returncode,
                    attempt + 1, (rr.stderr or "").strip()[-160:])
        time.sleep(3 * (attempt + 1))
        rr = run_on(node, cid, command, timeout=timeout)
    return rr


def sigkill_worker(node: str, cid: int, tag: str) -> None:
    """SIGKILL the robot's python process on `node` and wait until it is gone."""
    pat = shlex.quote(worker_pattern(cid, tag))
    cmd = (f"pkill -9 -u $USER -f {pat} 2>/dev/null; "
           f"for i in $(seq 1 15); do pgrep -u $USER -f {pat} >/dev/null || exit 0; "
           f"sleep 1; pkill -9 -u $USER -f {pat} 2>/dev/null; done; exit 1")
    rr = run_on(node, cid, cmd, timeout=40)
    if rr.returncode != 0:
        LOG.warning("[Kill] robot %03d on %s: rc=%s %s", cid, node, rr.returncode,
                    (rr.stderr or "").strip()[-120:])
    # The apptainer/ssh parent exits once its python is gone; reap it.
    with _procs_lock:
        p = _procs.pop(cid, None)
    if p is not None:
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            _terminate_one(p, f"exp1_robot_{cid:03d}", hard_after=5.0)


# --------------------------------------------------------------------------- #
# Real actions for exp1_logic.run_migration.                                  #
# --------------------------------------------------------------------------- #
class ClusterOps:
    def __init__(self, cfg, r, dht, ev: Event, tag: str, probe_timeout: float):
        self.cfg, self.r, self.dht, self.ev = cfg, r, dht, ev
        self.tag, self.probe_timeout = tag, probe_timeout
        self.cid = ev.cid

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
            copy = f"rsync -a -e '{_SSH_COPY}' {src}:{shlex.quote(src_dir)}/ {q}/"
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
            copy = (f"scp -q -o StrictHostKeyChecking=no -o BatchMode=yes "
                    f"-o ConnectTimeout=10 {src}:{q_src} {q_dst}")
        else:
            copy = f"rsync -a -e '{_SSH_COPY}' {src}:{q_src} {q_dst}"
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

    def kill_source(self) -> None:
        ev = self.ev
        LOG.info("[%s] KILL source %s (%s)", ev.robot_id, ev.src, ev.kill_point)
        sigkill_worker(ev.src, ev.cid, self.tag)
        # The source's local storage is lost with it.
        run_on(ev.src, ev.cid, f"rm -rf {shlex.quote(ev.src_bundle)}")

    def kill_dst(self) -> None:
        sigkill_worker(self.ev.dst, self.ev.cid, self.tag)

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
        launch_exp1_robot(self.cfg, ev.dst, ev.cid, self.tag, env)

    def wait_probe(self):
        key = f"probe_metrics:{self.ev.robot_id}"
        deadline = time.time() + self.probe_timeout
        while time.time() < deadline:
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

    flower = launch_task2_flower(cfg)
    time.sleep(15)
    for cid in range(cfg.num_clients):
        for cn in cfg.client_nodes:
            establish_ssh_master(cn, cid)
            time.sleep(1)
    for cid in range(cfg.num_clients):
        node = cfg.home_node_for_client(cid)
        update_node(f"robot_{cid:03d}", node)
        launch_exp1_robot(cfg, node, cid, tag, {"EXP1_RELAUNCH": "0"})
        time.sleep(0.5)
    LOG.info("All %d robots launched.", cfg.num_clients)

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
                   n_replicas=n_rep, exclude_dst=excl, ns=tag)
        ops = ClusterOps(cfg, r, dht, ev, tag, probe_timeout)
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
                 success_rate_post=-1, regression_pct=-1, fl_rounds_to_recover=-1)
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
        if dht is not None:
            dht.quit_all()
    LOG.info("Migration events: %d -> %s", writer.event_count, writer.path)
    return 1 if stalled else 0


if __name__ == "__main__":
    sys.exit(main())
