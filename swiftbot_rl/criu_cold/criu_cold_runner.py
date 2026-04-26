"""
CRIU Cold Baseline Runner — Condition B.
Launches 8 containers with random policy workers.
Triggers CRIU cold (stop-and-copy) migration on overload.
No FedAvg, no PPO, no Flower server.
"""
import asyncio, docker, os, sys, time, json, subprocess, shutil
import logging, threading, socket, platform, redis
from kademlia.network import Server

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../shared"))
from metrics_collector import MigrationMetricsWriter, get_gpu_util, get_cpu_util, get_net_bytes


def _get_post_migration_success_rate(r_client, robot_id: str, baseline_count: int,
                                      n: int = 10, timeout_s: float = 120.0) -> float:
    deadline = time.time() + timeout_s
    new_tasks = []
    seen: set = set()
    while time.time() < deadline and len(new_tasks) < n:
        for raw in r_client.lrange("task_logs", 0, 300):
            try:
                entry = json.loads(raw)
            except Exception:
                continue
            if entry.get("robot_id") != robot_id:
                continue
            tc = entry.get("task_counter", 0)
            if tc > baseline_count and tc not in seen:
                seen.add(tc)
                new_tasks.append(entry)
        time.sleep(0.5)
    if not new_tasks:
        return 0.0
    return sum(1 for t in new_tasks[:n] if t.get("status") == "success") / min(n, len(new_tasks))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

DOCKER_IMAGE_NAME = "swiftbot-baseline:latest"
NUM_NODES         = 4
CLIENTS_PER_NODE  = 2
TOTAL_CLIENTS     = 8
BASE_PORT         = 8480   # different port to avoid conflict
CHECKPOINT_BASE   = "/tmp/swiftbot_checkpoints_criu_cold"
RESULT_DIR        = os.path.join(os.path.dirname(__file__), "results")
REDIS_HOST        = "localhost"

metrics_writer = MigrationMetricsWriter("criu_cold", RESULT_DIR)
r_client       = redis.Redis(host=REDIS_HOST, decode_responses=True)


def get_master_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]; s.close(); return ip


class DHTNode:
    def __init__(self, node_id, port, bootstrap=None):
        self.node_id   = node_id
        self.port      = port
        self.bootstrap = bootstrap
        self.server    = Server()
        self.docker    = docker.from_env()

    async def start(self, master_ip):
        await self.server.listen(self.port)
        if self.bootstrap:
            await self.server.bootstrap([self.bootstrap])
        await self.launch_workers(master_ip)

    async def launch_workers(self, master_ip):
        for i in range(CLIENTS_PER_NODE):
            cid   = self.node_id * CLIENTS_PER_NODE + i
            cname = f"swiftbot-criu-cold-{cid}"
            ctype = "gpu_specialist" if cid < 4 else "cpu_specialist"
            try:
                try:
                    self.docker.containers.get(cname).remove(force=True)
                except docker.errors.NotFound:
                    pass
                cmd = (f"python3 /app/worker_random_client.py "
                       f"--client-id {cid} --container-type {ctype}")
                os.makedirs(f"{CHECKPOINT_BASE}/robot_{cid:03d}", exist_ok=True)
                self.docker.containers.run(
                    DOCKER_IMAGE_NAME, command=cmd, name=cname,
                    detach=True, tty=True, shm_size="4g",
                    environment={
                        "REDIS_HOST": REDIS_HOST,
                        "NVIDIA_VISIBLE_DEVICES": "all",
                        "PYTHONUNBUFFERED": "1",
                    },
                    device_requests=[docker.types.DeviceRequest(
                        count=-1, capabilities=[["gpu"]])],
                    volumes={CHECKPOINT_BASE: {"bind": "/checkpoints", "mode": "rw"}},
                    security_opt=["seccomp:unconfined"],
                    network_mode="host",
                )
                logger.info(f"  Started {cname}")
            except Exception as e:
                logger.error(f"  Failed {cname}: {e}")
            await asyncio.sleep(0.5)


def trigger_criu_cold_migration(robot_id: str, container_name: str,
                                 success_rate_pre: float = 0.0,
                                 task_counter_pre: int = 0) -> dict:
    """CRIU cold: fully stop container, dump, transfer, restore."""
    logger.info(f"[CRIU COLD] Migrating {robot_id}")
    t_trigger  = time.perf_counter()
    gpu_pre    = get_gpu_util()
    cpu_pre    = get_cpu_util()
    net_pre    = get_net_bytes()

    chk_src = os.path.join(CHECKPOINT_BASE, robot_id)
    chk_dst = os.path.join(CHECKPOINT_BASE, f"{robot_id}_dest")
    os.makedirs(chk_dst, exist_ok=True)
    criu_dir = os.path.join(chk_src, "criu_cold")
    os.makedirs(criu_dir, exist_ok=True)

    # Step 1: STOP container and dump (cold — no --leave-running).
    # Real `docker checkpoint create`. Requires Docker daemon experimental=true
    # and `criu` installed on host. If this fails, the error is logged and
    # the metrics row reflects that (no simulator fallback — Ubuntu 22.04
    # bare metal must support real CRIU per the experiment spec).
    t_dump_start = time.perf_counter()
    gpu_during   = get_gpu_util()
    cpu_during   = get_cpu_util()

    result = subprocess.run([
        "docker", "checkpoint", "create",
        f"--checkpoint-dir={criu_dir}",
        container_name, "cold_chk"      # no --leave-running = container stops
    ], capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        logger.error(f"[CRIU COLD] docker checkpoint failed for {robot_id} "
                     f"(rc={result.returncode}): {result.stderr.strip()[:300]}")

    dump_ms = (time.perf_counter() - t_dump_start) * 1000
    chk_size_mb = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(criu_dir) for f in files
    ) / (1024 * 1024) if os.path.exists(criu_dir) else 0

    # Step 2: Transfer (sequential — must complete dump first)
    t_xfer = time.perf_counter()
    shutil.copytree(criu_dir, os.path.join(chk_dst, "criu_cold"), dirs_exist_ok=True)
    transfer_ms = (time.perf_counter() - t_xfer) * 1000

    # Step 3: Restore (container was stopped by checkpoint create — restore same name)
    t_restore = time.perf_counter()
    rr = subprocess.run([
        "docker", "start",
        f"--checkpoint-dir={os.path.join(chk_dst, 'criu_cold')}",
        f"--checkpoint=cold_chk",
        container_name
    ], capture_output=True, text=True, timeout=60)
    if rr.returncode != 0:
        logger.error(f"[CRIU COLD] docker start --checkpoint failed for "
                     f"{robot_id}: {rr.stderr.strip()[:300]}")
    restore_ms = (time.perf_counter() - t_restore) * 1000

    total_MTT_ms = (time.perf_counter() - t_trigger) * 1000
    net_post     = get_net_bytes()

    r_client.set(f"migration_done:{robot_id}", "1", ex=60)

    # Measure post-migration success rate
    success_rate_post = _get_post_migration_success_rate(
        r_client, robot_id, task_counter_pre, n=10
    )
    regression_pct = 0.0
    if success_rate_pre > 0:
        regression_pct = (success_rate_pre - success_rate_post) / success_rate_pre * 100

    logger.info(f"[CRIU COLD] {robot_id}: MTT={total_MTT_ms:.0f}ms "
                f"dump={dump_ms:.0f}ms transfer={transfer_ms:.0f}ms "
                f"regression={regression_pct:.1f}%")

    return {
        "robot_id":                    robot_id,
        "trigger_to_dump_ms":          dump_ms,
        "dump_to_transfer_ms":         transfer_ms,
        "transfer_to_restore_ms":      restore_ms,
        "policy_load_ms":              0,
        "downtime_ms":                 total_MTT_ms,
        "total_MTT_ms":                total_MTT_ms,
        "success_rate_pre":            success_rate_pre,
        "success_rate_post":           success_rate_post,
        "regression_pct":              round(regression_pct, 2),
        "gpu_util_pre_migration":      gpu_pre,
        "gpu_util_during_migration":   gpu_during,
        "gpu_util_post_migration":     get_gpu_util(),
        "cpu_util_pre_migration":      cpu_pre,
        "cpu_util_during_migration":   cpu_during,
        "cpu_util_post_migration":     get_cpu_util(),
        "network_bytes_transferred":   net_post - net_pre,
        "checkpoint_size_mb":          round(chk_size_mb, 2),
        # CRIU baselines never transfer a replay buffer — that's the whole
        # point of the comparison. Always 0 here.
        "replay_buffer_entries_restored": 0,
        "criu_mode":                   "cold",
    }


def live_status_thread(interval: int = 10):
    """
    Periodic snapshot of per-robot progress for the CRIU COLD baseline.
    Reads the same `task_logs` list the DHT+FRL runner uses, plus a
    monotonic counter of migrations performed in this process.
    Mirrors the DHT+FRL `live_status_thread` so the operator sees consistent
    output across all three conditions.
    """
    logger.info(f"[Status COLD] Live status thread started (every {interval}s)")
    time.sleep(interval)
    while True:
        try:
            latest: dict = {}
            for raw in r_client.lrange("task_logs", 0, 800):
                try:
                    e = json.loads(raw)
                except Exception:
                    continue
                rid = e.get("robot_id")
                if rid and rid not in latest:
                    latest[rid] = e
                if len(latest) >= TOTAL_CLIENTS:
                    break

            rows = []
            for cid in range(TOTAL_CLIENTS):
                rid = f"robot_{cid:03d}"
                e   = latest.get(rid)
                if not e:
                    rows.append(f"  {rid}: <no tasks yet>")
                    continue
                rows.append(
                    f"  {rid}: tasks={e.get('task_counter',0):>4}  "
                    f"status={e.get('status','?'):<8}  "
                    f"success10={e.get('success_rate_rolling10',0):.2f}  "
                    f"bid={e.get('bid_value',0):.2f}  "
                    f"reward={e.get('reward',0):+.2f}"
                )

            pending = r_client.keys("migration_request:robot_*")
            mig_done = metrics_writer._event_counter
            extra = f"  migrations_done={mig_done}"
            if pending:
                extra += f"  pending={len(pending)}"

            logger.info("=" * 78)
            logger.info(f"[Status COLD] Live snapshot{extra}")
            for line in rows:
                logger.info(line)
            logger.info("=" * 78)
        except Exception as e:
            logger.error(f"[Status COLD] Error: {e}")
        time.sleep(interval)


def migration_monitor_thread():
    logger.info("[Monitor COLD] Started")
    while True:
        try:
            for key in r_client.keys("migration_request:robot_*"):
                raw = r_client.get(key)
                if not raw: continue
                info     = json.loads(raw)
                robot_id = info["robot_id"]
                cid      = int(robot_id.split("_")[1])
                cname    = f"swiftbot-criu-cold-{cid}"
                r_client.delete(key)
                mig = trigger_criu_cold_migration(
                    robot_id, cname,
                    success_rate_pre=float(info.get("success_rate", 0)),
                    task_counter_pre=int(info.get("task_counter", 0)),
                )
                metrics_writer.write_event(mig)
        except Exception as e:
            logger.error(f"[Monitor] {e}")
        time.sleep(1)


async def main():
    master_ip = get_master_ip()
    os.makedirs(CHECKPOINT_BASE, exist_ok=True)
    threading.Thread(target=migration_monitor_thread, daemon=True).start()
    threading.Thread(target=live_status_thread, args=(10,), daemon=True).start()
    nodes = (
        [DHTNode(0, BASE_PORT)] +
        [DHTNode(i, BASE_PORT + i, ("127.0.0.1", BASE_PORT)) for i in range(1, NUM_NODES)]
    )
    await asyncio.gather(*[n.start(master_ip) for n in nodes])
    logger.info("\n[CRIU COLD] All containers running. Waiting for experiment to complete...")
    try:
        while True:
            done = sum(1 for i in range(TOTAL_CLIENTS)
                       if r_client.get(f"robot_done:robot_{i:03d}"))
            if done >= TOTAL_CLIENTS:
                logger.info("All robots done.")
                break
            await asyncio.sleep(10)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    asyncio.run(main())
