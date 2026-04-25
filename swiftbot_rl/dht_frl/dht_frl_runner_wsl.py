"""
DHT + FRL Orchestrator — WSL2 version (Condition A).
Replaces Docker container spawning with subprocess.Popen workers.
Replaces 'docker checkpoint' CRIU calls with CRIUSimulator("unified").
All metric collection logic is unchanged from dht_frl_runner.py.

Run from ~/swiftbot_rl:
    python3 dht_frl/dht_frl_runner_wsl.py
"""
import asyncio, os, sys, time, json, shutil, logging, threading, signal, subprocess
import platform, socket, redis
from kademlia.network import Server

# Resolve shared/ relative to project root
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, os.path.join(_ROOT, "shared"))
from metrics_collector import MigrationMetricsWriter, get_gpu_util, get_cpu_util, get_net_bytes
from criu_simulator import CRIUSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# --- CONFIG ---
NUM_NODES          = 4
CLIENTS_PER_NODE   = 2
TOTAL_CLIENTS      = NUM_NODES * CLIENTS_PER_NODE   # = 8
BASE_PORT          = 8470
CHECKPOINT_BASE    = "/tmp/swiftbot_checkpoints"
RESULT_DIR         = os.path.join(_HERE, "results")
REDIS_HOST         = "localhost"

metrics_writer = MigrationMetricsWriter("dht_frl", RESULT_DIR)
r_client       = redis.Redis(host=REDIS_HOST, decode_responses=True)

# Module-level process registry: robot_id → Popen
PROCESS_REGISTRY: dict = {}


def get_master_ip() -> str:
    # Subprocess workers are local processes — always 127.0.0.1 on WSL2
    return "127.0.0.1"


def _get_post_migration_success_rate(robot_id: str,
                                      baseline_count: int,
                                      n: int = 10,
                                      timeout_s: float = 120.0) -> float:
    """
    Wait for n new task_log entries from robot_id after migration,
    then return success rate over those n tasks.
    """
    deadline = time.time() + timeout_s
    new_tasks = []
    seen = set()
    while time.time() < deadline and len(new_tasks) < n:
        raw_list = r_client.lrange("task_logs", 0, 200)
        for raw in raw_list:
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


class DHTNode:
    """Kademlia DHT node — manages 2 robot subprocess workers each."""

    def __init__(self, node_id, port, bootstrap=None):
        self.node_id   = node_id
        self.port      = port
        self.bootstrap = bootstrap
        self.server    = Server()

    async def start(self, master_ip):
        await self.server.listen(self.port)
        if self.bootstrap:
            await self.server.bootstrap([self.bootstrap])
        await self.launch_workers(master_ip)

    async def launch_workers(self, master_ip):
        logger.info(f"[Node {self.node_id}] Launching subprocess workers...")
        for i in range(CLIENTS_PER_NODE):
            cid   = self.node_id * CLIENTS_PER_NODE + i
            ctype = "gpu_specialist" if cid < 4 else "cpu_specialist"
            robot_id = f"robot_{cid:03d}"

            os.makedirs(os.path.join(CHECKPOINT_BASE, robot_id), exist_ok=True)
            os.makedirs(RESULT_DIR, exist_ok=True)

            log_path = os.path.join(RESULT_DIR, f"worker_{cid}.log")
            env = os.environ.copy()
            env["MASTER_ADDRESS"]    = f"{master_ip}:8080"
            env["REDIS_HOST"]        = REDIS_HOST
            env["PYTHONUNBUFFERED"]  = "1"

            worker_script = os.path.join(_HERE, "worker_robot_client_wsl.py")
            proc = subprocess.Popen(
                [sys.executable, worker_script,
                 "--client-id", str(cid),
                 "--num-clients", str(TOTAL_CLIENTS),
                 "--container-type", ctype],
                env=env,
                stdout=open(log_path, "w"),
                stderr=subprocess.STDOUT,
            )
            PROCESS_REGISTRY[robot_id] = proc
            logger.info(f"  Spawned {robot_id} ({ctype}) PID={proc.pid} → {log_path}")
            await asyncio.sleep(0.3)


# ------------------------------------------------------------------ #
# UNIFIED MIGRATION — subprocess + CRIUSimulator
# ------------------------------------------------------------------ #

def trigger_unified_migration(robot_id: str, success_rate_pre: float,
                               task_counter_pre: int) -> dict:
    logger.info(f"[MIGRATION] Starting unified migration for {robot_id}")

    t_trigger  = time.perf_counter()
    gpu_pre    = get_gpu_util()
    cpu_pre    = get_cpu_util()
    net_pre    = get_net_bytes()

    chk_src = os.path.join(CHECKPOINT_BASE, robot_id)
    chk_dst = os.path.join(CHECKPOINT_BASE, f"{robot_id}_dest")
    os.makedirs(chk_dst, exist_ok=True)

    # Wait for worker to save policy + buffer (sets ready_for_criu key)
    logger.info(f"  Waiting for {robot_id} to save policy weights...")
    deadline = time.time() + 30
    while time.time() < deadline:
        if r_client.get(f"ready_for_criu:{robot_id}"):
            break
        time.sleep(0.2)

    # --- Simulated CRIU unified checkpoint (warm-style, policy is payload) ---
    t_dump_start = time.perf_counter()
    gpu_during   = get_gpu_util()
    cpu_during   = get_cpu_util()

    criu_dir = os.path.join(chk_src, "criu")
    sim      = CRIUSimulator("unified")
    dump_res = sim.simulate_checkpoint(robot_id, criu_dir)
    dump_ms  = dump_res["dump_ms"]

    chk_size_mb = dump_res["size_mb"]

    # --- Parallel transfer: CRIU sim dir + policy files ---
    t_transfer_start = time.perf_counter()
    transfer_results = {}

    def transfer_criu():
        t = time.perf_counter()
        res = sim.simulate_transfer(criu_dir, os.path.join(chk_dst, "criu"))
        transfer_results["criu_ms"] = res["transfer_ms"]

    def transfer_policy():
        t = time.perf_counter()
        for fname in ["policy_weights.pt", "replay_buffer.pkl", "manifest.json"]:
            src = os.path.join(chk_src, fname)
            dst = os.path.join(chk_dst, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        transfer_results["policy_ms"] = (time.perf_counter() - t) * 1000

    import threading as _t
    t1 = _t.Thread(target=transfer_criu)
    t2 = _t.Thread(target=transfer_policy)
    t1.start(); t2.start()
    t1.join();  t2.join()

    transfer_ms = (time.perf_counter() - t_transfer_start) * 1000

    # --- Simulated restore ---
    t_restore_start = time.perf_counter()
    restore_res     = sim.simulate_restore(criu_dir, robot_id)
    restore_ms      = restore_res["restore_ms"]

    # Signal worker to load policy (worker measures policy_load_ms)
    r_client.set(f"load_policy:{robot_id}", chk_dst, ex=60)

    # Wait for worker's first bid after migration
    deadline_bid   = time.time() + 30
    policy_load_ms = 0.0
    while time.time() < deadline_bid:
        data = r_client.get(f"first_bid_after_migration:{robot_id}")
        if data:
            try:
                info = json.loads(data)
                policy_load_ms = float(info.get("policy_load_ms", 0))
            except Exception:
                pass
            r_client.delete(f"first_bid_after_migration:{robot_id}")
            break
        time.sleep(0.1)

    t_fully_operational = time.perf_counter()
    total_MTT_ms = (t_fully_operational - t_trigger) * 1000
    downtime_ms  = total_MTT_ms

    net_post  = get_net_bytes()
    gpu_post  = get_gpu_util()
    cpu_post  = get_cpu_util()
    net_bytes = net_post - net_pre

    # Signal migration complete to worker
    r_client.set(f"migration_done:{robot_id}", "1", ex=60)

    # Measure post-migration success rate (wait for 10 more tasks)
    success_rate_post = _get_post_migration_success_rate(
        robot_id, task_counter_pre, n=10
    )
    regression_pct = 0.0
    if success_rate_pre > 0:
        regression_pct = (success_rate_pre - success_rate_post) / success_rate_pre * 100

    logger.info(f"[MIGRATION] {robot_id} complete: MTT={total_MTT_ms:.0f}ms "
                f"dump={dump_ms:.0f}ms transfer={transfer_ms:.0f}ms "
                f"policy_load={policy_load_ms:.0f}ms "
                f"regression={regression_pct:.1f}%")

    return {
        "robot_id":                    robot_id,
        "trigger_to_dump_ms":          dump_ms,
        "dump_to_transfer_ms":         transfer_ms,
        "transfer_to_restore_ms":      restore_ms,
        "policy_load_ms":              policy_load_ms,
        "downtime_ms":                 downtime_ms,
        "total_MTT_ms":                total_MTT_ms,
        "success_rate_pre":            success_rate_pre,
        "success_rate_post":           success_rate_post,
        "regression_pct":              round(regression_pct, 2),
        "gpu_util_pre_migration":      gpu_pre,
        "gpu_util_during_migration":   gpu_during,
        "gpu_util_post_migration":     gpu_post,
        "cpu_util_pre_migration":      cpu_pre,
        "cpu_util_during_migration":   cpu_during,
        "cpu_util_post_migration":     cpu_post,
        "network_bytes_transferred":   net_bytes,
        "checkpoint_size_mb":          round(chk_size_mb, 2),
        "criu_mode":                   "unified",
    }


def migration_monitor_thread():
    logger.info("[Monitor] DHT+FRL migration monitor started")
    while True:
        try:
            keys = r_client.keys("migration_request:robot_*")
            for key in keys:
                raw = r_client.get(key)
                if not raw:
                    continue
                info     = json.loads(raw)
                robot_id = info["robot_id"]
                r_client.delete(key)

                mig = trigger_unified_migration(
                    robot_id,
                    success_rate_pre=float(info.get("success_rate", 0)),
                    task_counter_pre=int(info.get("task_counter", 0)),
                )
                metrics_writer.write_event(mig)
        except Exception as e:
            logger.error(f"[Monitor] Error: {e}")
        time.sleep(1)


async def main():
    master_ip = get_master_ip()
    os.makedirs(CHECKPOINT_BASE, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    logger.info(f"Master IP: {master_ip}")

    monitor = threading.Thread(target=migration_monitor_thread, daemon=True)
    monitor.start()

    nodes = (
        [DHTNode(0, BASE_PORT)] +
        [DHTNode(i, BASE_PORT + i, ("127.0.0.1", BASE_PORT))
         for i in range(1, NUM_NODES)]
    )

    logger.info(f"Launching {TOTAL_CLIENTS} subprocess workers...")
    await asyncio.gather(*[n.start(master_ip) for n in nodes])

    logger.info("\n[SUCCESS] All workers started.")
    logger.info("Flower server must be running in another terminal:")
    logger.info("  cd ~/swiftbot_rl && python3 dht_frl/flower_server.py")
    logger.info("\nPress Ctrl+C to stop when experiment completes.\n")

    try:
        while True:
            await asyncio.sleep(10)
    except KeyboardInterrupt:
        logger.info("Stopping workers...")
        for rid, proc in PROCESS_REGISTRY.items():
            try:
                proc.terminate()
            except Exception:
                pass
        logger.info("All workers terminated.")


if __name__ == "__main__":
    asyncio.run(main())
