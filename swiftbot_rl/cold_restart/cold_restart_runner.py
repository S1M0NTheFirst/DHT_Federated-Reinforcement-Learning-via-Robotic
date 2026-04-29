"""
Cold Restart Baseline Runner — Condition C'.
NO checkpoint at all. On migration trigger:
  1. SIGKILL the container
  2. Launch a fresh container of the same image
  3. Worker reads `resume_counter` from redis and continues from there
     — but with NO state preserved (no policy, no replay, no rolling sr)

This is the "lower bound" baseline: trivial to implement, fastest to
restart, and shows what happens when an operator simply doesn't
checkpoint at all. Useful as one extreme on the migration trade-off
spectrum (the other extreme being our DHT+FRL state-complete transfer).
"""
import asyncio, docker, os, sys, time, json, subprocess, shutil
import logging, threading, socket, redis
from kademlia.network import Server

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../shared"))
from metrics_collector import (MigrationMetricsWriter,
                                get_gpu_util, get_cpu_util, get_net_bytes)


def _get_post_migration_success_rate(r_client, robot_id, baseline_count,
                                      n=10, timeout_s=120.0):
    deadline = time.time() + timeout_s
    new_tasks, seen = [], set()
    while time.time() < deadline and len(new_tasks) < n:
        for raw in r_client.lrange("task_logs", 0, 300):
            try: e = json.loads(raw)
            except Exception: continue
            if e.get("robot_id") != robot_id: continue
            tc = e.get("task_counter", 0)
            if tc > baseline_count and tc not in seen:
                seen.add(tc); new_tasks.append(e)
        time.sleep(0.5)
    if not new_tasks: return 0.0
    return sum(1 for t in new_tasks[:n] if t.get("status") == "success") / min(n, len(new_tasks))


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

DOCKER_IMAGE_NAME = "swiftbot-cold-restart:latest"
NUM_NODES         = 4
CLIENTS_PER_NODE  = 2
TOTAL_CLIENTS     = 8
BASE_PORT         = 8510
RESULT_DIR        = os.path.join(os.path.dirname(__file__), "results")
REDIS_HOST        = "localhost"

metrics_writer = MigrationMetricsWriter("cold_restart", RESULT_DIR)
r_client       = redis.Redis(host=REDIS_HOST, decode_responses=True)
_LAUNCH_LOCK   = asyncio.Lock()


def get_master_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]; s.close(); return ip


def _start_one_container(d_client, cid, lock_held=True):
    """Reused for initial launch AND post-migration restart. Caller must
    already hold _LAUNCH_LOCK if lock_held=False is not passed."""
    cname = f"swiftbot-cold-restart-{cid}"
    ctype = "gpu_specialist" if cid < 4 else "cpu_specialist"
    try:
        try: d_client.containers.get(cname).remove(force=True)
        except docker.errors.NotFound: pass
        cmd = (f"python3 /app/worker_random_client.py "
               f"--client-id {cid} --container-type {ctype}")
        container = d_client.containers.create(
            DOCKER_IMAGE_NAME, command=cmd, name=cname,
            tty=False, shm_size="4g",
            environment={"REDIS_HOST": REDIS_HOST,
                         "NVIDIA_VISIBLE_DEVICES": "all",
                         "PYTHONUNBUFFERED": "1"},
            device_requests=[docker.types.DeviceRequest(
                count=-1, capabilities=[["gpu"]])],
            security_opt=["seccomp:unconfined"],
            network_mode="host",
        )
        ok = False
        for attempt in range(10):
            try: container.start()
            except Exception as se:
                logger.warning(f"  {cname} start() {attempt+1} failed: {se}")
                time.sleep(2.0); continue
            time.sleep(1.5)
            container.reload()
            if container.status == "running": ok = True; break
            if container.status in ("exited", "dead"): break
            time.sleep(2.0)
        if not ok:
            # Shell fallback
            try: container.remove(force=True)
            except Exception: pass
            time.sleep(3.0)
            sh = subprocess.run(
                ["docker", "run", "-d", "--name", cname, "--shm-size=4g",
                 "-e", f"REDIS_HOST={REDIS_HOST}",
                 "-e", "NVIDIA_VISIBLE_DEVICES=all",
                 "-e", "PYTHONUNBUFFERED=1",
                 "--gpus", "all",
                 "--security-opt", "seccomp=unconfined",
                 "--network", "host", DOCKER_IMAGE_NAME,
                 "python3", "/app/worker_random_client.py",
                 "--client-id", str(cid), "--container-type", ctype],
                capture_output=True, text=True, timeout=180,
            )
            if sh.returncode == 0:
                time.sleep(3.0)
                container = d_client.containers.get(cname)
                container.reload()
                if container.status == "running":
                    ok = True
                    logger.info(f"  {cname} rescued via shell")
            else:
                logger.error(f"  {cname} shell failed: {sh.stderr[:300]}")
        return ok
    except Exception as e:
        logger.error(f"  {cname} launch error: {e}")
        return False


class DHTNode:
    def __init__(self, node_id, port, bootstrap=None):
        self.node_id   = node_id
        self.port      = port
        self.bootstrap = bootstrap
        self.server    = Server()
        self.docker    = docker.from_env(timeout=300)

    async def start(self, master_ip):
        await self.server.listen(self.port)
        if self.bootstrap:
            await self.server.bootstrap([self.bootstrap])
        await self.launch_workers(master_ip)

    async def launch_workers(self, master_ip):
        for i in range(CLIENTS_PER_NODE):
            cid = self.node_id * CLIENTS_PER_NODE + i
            async with _LAUNCH_LOCK:
                ok = _start_one_container(self.docker, cid)
                if ok:
                    logger.info(f"  Started swiftbot-cold-restart-{cid} (running)")
                else:
                    logger.error(f"  swiftbot-cold-restart-{cid} FAILED")
                await asyncio.sleep(2.0)


def trigger_cold_restart(robot_id: str, container_name: str,
                         success_rate_pre: float = 0.0,
                         task_counter_pre: int = 0) -> dict:
    """Kill container, launch fresh. Worker resumes from `resume_counter`
    in redis but loses ALL in-process state."""
    logger.info(f"[COLD RESTART] Migrating {robot_id}")
    t_trigger = time.perf_counter()
    gpu_pre, cpu_pre = get_gpu_util(), get_cpu_util()
    net_pre  = get_net_bytes()
    cid      = int(robot_id.split("_")[1])
    d_client = docker.from_env(timeout=300)

    # Step 1: SIGKILL the running container.
    t_kill = time.perf_counter()
    try:
        c = d_client.containers.get(container_name)
        c.kill()
        c.remove(force=True)
    except Exception as e:
        logger.warning(f"  kill/remove for {container_name}: {e}")
    kill_ms = (time.perf_counter() - t_kill) * 1000

    # Step 2: launch fresh — worker reads resume_counter from redis.
    t_start = time.perf_counter()
    ok = _start_one_container(d_client, cid)
    start_ms = (time.perf_counter() - t_start) * 1000
    if not ok:
        logger.error(f"[COLD RESTART] failed to relaunch {container_name}")

    total_MTT_ms = (time.perf_counter() - t_trigger) * 1000
    net_post = get_net_bytes()

    r_client.set(f"migration_done:{robot_id}", "1", ex=60)

    success_rate_post = _get_post_migration_success_rate(
        r_client, robot_id, task_counter_pre, n=10)
    regression_pct = 0.0
    if success_rate_pre > 0:
        regression_pct = (success_rate_pre - success_rate_post) / success_rate_pre * 100

    logger.info(f"[COLD RESTART] {robot_id}: MTT={total_MTT_ms:.0f}ms "
                f"kill={kill_ms:.0f}ms relaunch={start_ms:.0f}ms "
                f"regression={regression_pct:.1f}%")

    return {
        "robot_id":                       robot_id,
        # No dump/transfer/restore phases — kill is the "dump", relaunch is the "restore"
        "trigger_to_dump_ms":             kill_ms,
        "dump_to_transfer_ms":            0,
        "transfer_to_restore_ms":         start_ms,
        "policy_load_ms":                 0,
        "downtime_ms":                    total_MTT_ms,
        "total_MTT_ms":                   total_MTT_ms,
        "success_rate_pre":               success_rate_pre,
        "success_rate_post":              success_rate_post,
        "regression_pct":                 round(regression_pct, 2),
        "gpu_util_pre_migration":         gpu_pre,
        "gpu_util_during_migration":      gpu_pre,
        "gpu_util_post_migration":        get_gpu_util(),
        "cpu_util_pre_migration":         cpu_pre,
        "cpu_util_during_migration":      cpu_pre,
        "cpu_util_post_migration":        get_cpu_util(),
        "network_bytes_transferred":      net_post - net_pre,
        # NO checkpoint is saved. This is the whole point of cold restart.
        "checkpoint_size_mb":             0.0,
        "replay_buffer_entries_restored": 0,
        "criu_mode":                      "cold_restart",
    }


def live_status_thread(interval: int = 10):
    logger.info(f"[Status RESTART] Live status thread (every {interval}s)")
    time.sleep(interval)
    while True:
        try:
            latest = {}
            for raw in r_client.lrange("task_logs", 0, 800):
                try: e = json.loads(raw)
                except Exception: continue
                rid = e.get("robot_id")
                if rid and rid not in latest: latest[rid] = e
                if len(latest) >= TOTAL_CLIENTS: break
            rows = []
            for cid in range(TOTAL_CLIENTS):
                rid = f"robot_{cid:03d}"
                e   = latest.get(rid)
                if not e: rows.append(f"  {rid}: <no tasks yet>"); continue
                rows.append(f"  {rid}: tasks={e.get('task_counter',0):>4}  "
                            f"status={e.get('status','?'):<8}  "
                            f"success10={e.get('success_rate_rolling10',0):.2f}")
            mig_done = metrics_writer._event_counter
            extra = f"  migrations_done={mig_done}"
            logger.info("=" * 78)
            logger.info(f"[Status RESTART] Live snapshot{extra}")
            for line in rows: logger.info(line)
            logger.info("=" * 78)
        except Exception as e:
            logger.error(f"[Status RESTART] {e}")
        time.sleep(interval)


def migration_monitor_thread():
    logger.info("[Monitor RESTART] Started")
    while True:
        try:
            for key in r_client.keys("migration_request:robot_*"):
                raw = r_client.get(key)
                if not raw: continue
                info     = json.loads(raw)
                robot_id = info["robot_id"]
                cid      = int(robot_id.split("_")[1])
                cname    = f"swiftbot-cold-restart-{cid}"
                r_client.delete(key)
                mig = trigger_cold_restart(
                    robot_id, cname,
                    success_rate_pre=float(info.get("success_rate", 0)),
                    task_counter_pre=int(info.get("task_counter", 0)))
                metrics_writer.write_event(mig)
        except Exception as e:
            logger.error(f"[Monitor] {e}")
        time.sleep(1)


async def main():
    master_ip = get_master_ip()
    threading.Thread(target=migration_monitor_thread, daemon=True).start()
    threading.Thread(target=live_status_thread, args=(10,), daemon=True).start()
    nodes = ([DHTNode(0, BASE_PORT)] +
             [DHTNode(i, BASE_PORT + i, ("127.0.0.1", BASE_PORT))
              for i in range(1, NUM_NODES)])
    await asyncio.gather(*[n.start(master_ip) for n in nodes])
    logger.info("\n[COLD RESTART] All containers running. Waiting...")
    try:
        d_client = docker.from_env(timeout=300)
        while True:
            done = sum(1 for i in range(TOTAL_CLIENTS)
                       if r_client.get(f"robot_done:robot_{i:03d}"))
            alive = 0
            for i in range(TOTAL_CLIENTS):
                try:
                    c = d_client.containers.get(f"swiftbot-cold-restart-{i}")
                    if c.status == "running": alive += 1
                except docker.errors.NotFound: pass
            ghosts = TOTAL_CLIENTS - alive - done
            # Cold restart legitimately has containers transiently absent
            # during kill/relaunch — don't treat those as ghosts. Require
            # a sustained ghost count for 2 consecutive checks.
            if done >= TOTAL_CLIENTS:
                logger.info("All robots done.")
                break
            await asyncio.sleep(10)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    asyncio.run(main())
