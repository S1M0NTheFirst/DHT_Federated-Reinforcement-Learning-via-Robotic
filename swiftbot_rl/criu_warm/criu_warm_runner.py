"""
CRIU Pre-copy (Warm) Baseline Runner — Condition C.
Same structure as criu_cold_runner but uses iterative pre-dumps.
Container keeps running during pre-dumps; only final delta pauses.
"""
import asyncio, docker, os, sys, time, json, subprocess, shutil
import logging, threading, socket, redis
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
BASE_PORT         = 8490   # different from cold and dht_frl
CHECKPOINT_BASE   = "/tmp/swiftbot_checkpoints_criu_warm"
RESULT_DIR        = os.path.join(os.path.dirname(__file__), "results")
REDIS_HOST        = "localhost"

metrics_writer = MigrationMetricsWriter("criu_warm", RESULT_DIR)
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
            cname = f"swiftbot-criu-warm-{cid}"
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


def trigger_criu_warm_migration(robot_id: str, container_name: str,
                                 success_rate_pre: float = 0.0,
                                 task_counter_pre: int = 0) -> dict:
    """CRIU pre-copy: pre-dump while running, then small final pause."""
    logger.info(f"[CRIU WARM] Migrating {robot_id}")
    t_trigger = time.perf_counter()
    gpu_pre   = get_gpu_util()
    cpu_pre   = get_cpu_util()
    net_pre   = get_net_bytes()

    chk_src  = os.path.join(CHECKPOINT_BASE, robot_id)
    chk_dst  = os.path.join(CHECKPOINT_BASE, f"{robot_id}_dest")
    criu_dir = os.path.join(chk_src, "criu_warm")
    os.makedirs(criu_dir, exist_ok=True)
    os.makedirs(chk_dst, exist_ok=True)

    # Pre-dump iterations (container stays running)
    gpu_during = get_gpu_util()
    cpu_during = get_cpu_util()

    for iteration in range(3):   # 3 pre-dump rounds
        predump_dir = os.path.join(criu_dir, f"predump_{iteration}")
        os.makedirs(predump_dir, exist_ok=True)
        subprocess.run([
            "docker", "checkpoint", "create",
            f"--checkpoint-dir={predump_dir}",
            "--leave-running",          # container keeps running!
            container_name, f"predump_{iteration}"
        ], capture_output=True, text=True, timeout=60)
        time.sleep(0.05)   # brief pause between iterations

    # Final delta dump (short pause — only dirty pages since last pre-dump)
    t_final_start = time.perf_counter()
    final_dir = os.path.join(criu_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    subprocess.run([
        "docker", "checkpoint", "create",
        f"--checkpoint-dir={final_dir}",
        container_name, "warm_chk"   # no --leave-running = container pauses
    ], capture_output=True, text=True, timeout=120)

    dump_ms = (time.perf_counter() - t_final_start) * 1000   # just the final pause

    # Transfer and restore (same as cold)
    t_xfer = time.perf_counter()
    shutil.copytree(criu_dir, os.path.join(chk_dst, "criu_warm"), dirs_exist_ok=True)
    transfer_ms = (time.perf_counter() - t_xfer) * 1000

    # Restore container (stopped by final checkpoint — restore same name)
    t_restore = time.perf_counter()
    subprocess.run([
        "docker", "start",
        f"--checkpoint-dir={os.path.join(chk_dst, 'criu_warm', 'final')}",
        f"--checkpoint=warm_chk",
        container_name
    ], capture_output=True, timeout=60)
    restore_ms = (time.perf_counter() - t_restore) * 1000

    total_MTT_ms = (time.perf_counter() - t_trigger) * 1000

    r_client.set(f"migration_done:{robot_id}", "1", ex=60)

    # Measure post-migration success rate
    success_rate_post = _get_post_migration_success_rate(
        r_client, robot_id, task_counter_pre, n=10
    )
    regression_pct = 0.0
    if success_rate_pre > 0:
        regression_pct = (success_rate_pre - success_rate_post) / success_rate_pre * 100

    logger.info(f"[CRIU WARM] {robot_id}: MTT={total_MTT_ms:.0f}ms "
                f"final_dump={dump_ms:.0f}ms transfer={transfer_ms:.0f}ms "
                f"regression={regression_pct:.1f}%")

    return {
        "robot_id":                   robot_id,
        "trigger_to_dump_ms":         dump_ms,
        "dump_to_transfer_ms":        transfer_ms,
        "transfer_to_restore_ms":     restore_ms,
        "policy_load_ms":             0,
        "downtime_ms":                dump_ms + restore_ms,
        "total_MTT_ms":               total_MTT_ms,
        "success_rate_pre":           success_rate_pre,
        "success_rate_post":          success_rate_post,
        "regression_pct":             round(regression_pct, 2),
        "gpu_util_pre_migration":     gpu_pre,
        "gpu_util_during_migration":  gpu_during,
        "gpu_util_post_migration":    get_gpu_util(),
        "cpu_util_pre_migration":     cpu_pre,
        "cpu_util_during_migration":  cpu_during,
        "cpu_util_post_migration":    get_cpu_util(),
        "network_bytes_transferred":  get_net_bytes() - net_pre,
        "checkpoint_size_mb":         0,
        "criu_mode":                  "precopy",
    }


def migration_monitor_thread():
    logger.info("[Monitor WARM] Started")
    while True:
        try:
            for key in r_client.keys("migration_request:robot_*"):
                raw = r_client.get(key)
                if not raw: continue
                info     = json.loads(raw)
                robot_id = info["robot_id"]
                cid      = int(robot_id.split("_")[1])
                cname    = f"swiftbot-criu-warm-{cid}"
                r_client.delete(key)
                mig = trigger_criu_warm_migration(
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
    nodes = (
        [DHTNode(0, BASE_PORT)] +
        [DHTNode(i, BASE_PORT + i, ("127.0.0.1", BASE_PORT)) for i in range(1, NUM_NODES)]
    )
    await asyncio.gather(*[n.start(master_ip) for n in nodes])
    logger.info("\n[CRIU WARM] All containers running. Waiting for experiment to complete...")
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
