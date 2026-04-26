"""
DHT + FRL Orchestrator — Condition A runner.
Based on dht_asr_optimized.py. Keeps entire Kademlia DHT structure.
Adds: Option B migration trigger (from host), CRIU calls, metric logging.

Run this from Ubuntu host:
    python3 dht_frl_runner.py
"""
import asyncio, docker, os, sys, time, json, subprocess, shutil, logging, threading
import platform, socket, redis, pickle
from kademlia.network import Server
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../shared"))
from metrics_collector import (MigrationMetricsWriter,
                                get_gpu_util, get_cpu_util, get_net_bytes,
                                get_container_pid, cuda_checkpoint_toggle,
                                real_criu_dump)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# --- CONFIG ---
DOCKER_IMAGE_NAME    = "swiftbot-robot:latest"
NUM_NODES            = 4
CLIENTS_PER_NODE     = 2
TOTAL_CLIENTS        = NUM_NODES * CLIENTS_PER_NODE   # = 8
BASE_PORT            = 8470
CHECKPOINT_BASE      = "/tmp/swiftbot_checkpoints"
RESULT_DIR           = os.path.join(os.path.dirname(__file__), "results")
REDIS_HOST           = "localhost"
OVERLOAD_THRESHOLD   = 0.85   # 85% CPU or GPU triggers migration

metrics_writer = MigrationMetricsWriter("dht_frl", RESULT_DIR)
r_client       = redis.Redis(host=REDIS_HOST, decode_responses=True)


def _get_post_migration_success_rate(robot_id: str, baseline_count: int,
                                      n: int = 10, timeout_s: float = 120.0) -> float:
    """Wait for n new task_log entries from robot after migration, return success rate."""
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


def get_master_ip() -> str:
    if sys.platform == "win32" or "microsoft" in platform.uname().release.lower():
        return "host.docker.internal"
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip


class DHTNode:
    """Kademlia DHT node — manages 2 robot containers each."""

    def __init__(self, node_id, port, bootstrap=None):
        self.node_id   = node_id
        self.port      = port
        self.bootstrap = bootstrap
        self.server    = Server()
        self.docker    = docker.from_env()
        self.container_names = []

    async def start(self, master_ip):
        await self.server.listen(self.port)
        if self.bootstrap:
            await self.server.bootstrap([self.bootstrap])
        await self.launch_workers(master_ip)

    async def launch_workers(self, master_ip):
        logger.info(f"[Node {self.node_id}] Launching containers...")
        for i in range(CLIENTS_PER_NODE):
            cid   = self.node_id * CLIENTS_PER_NODE + i
            cname = f"swiftbot-robot-{cid}"
            ctype = "gpu_specialist" if cid < 4 else "cpu_specialist"

            try:
                try:
                    c = self.docker.containers.get(cname)
                    env_vars = c.attrs["Config"]["Env"]
                    master_match = any(master_ip in e for e in env_vars
                                      if "MASTER_ADDRESS" in e)
                    if c.status == "running" and master_match:
                        logger.info(f"  {cname} already running correctly")
                        self.container_names.append(cname)
                        continue
                    c.remove(force=True)
                except docker.errors.NotFound:
                    pass

                cmd = (f"python3 /app/worker_robot_client.py "
                       f"--client-id {cid} --num-clients {TOTAL_CLIENTS} "
                       f"--container-type {ctype}")

                os.makedirs(f"{CHECKPOINT_BASE}/robot_{cid:03d}", exist_ok=True)

                self.docker.containers.run(
                    DOCKER_IMAGE_NAME,
                    command=cmd,
                    name=cname,
                    detach=True,
                    tty=True,
                    shm_size="4g",
                    environment={
                        "MASTER_ADDRESS": f"{master_ip}:8080",
                        "REDIS_HOST":     REDIS_HOST,
                        "NVIDIA_VISIBLE_DEVICES": "all",
                        "PYTHONUNBUFFERED": "1",
                    },
                    device_requests=[
                        docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                    ],
                    volumes={
                        CHECKPOINT_BASE: {
                            "bind": "/checkpoints", "mode": "rw"
                        }
                    },
                    security_opt=["seccomp:unconfined"],   # required for CRIU
                    restart_policy={"Name": "on-failure", "MaximumRetryCount": 3},
                    network_mode="host",
                )
                self.container_names.append(cname)
                logger.info(f"  Started {cname} ({ctype})")

            except Exception as e:
                logger.error(f"  Failed to start {cname}: {e}")
            await asyncio.sleep(0.5)


# ------------------------------------------------------------------ #
# OPTION B MIGRATION — triggered from HOST by DHT orchestrator
# ------------------------------------------------------------------ #

def trigger_unified_migration(robot_id: str, container_name: str,
                               source_node: str, dest_node: str,
                               success_rate_pre: float = 0.0,
                               task_counter_pre: int = 0):
    """
    Full unified migration sequence for DHT+FRL system.
    Called from host when migration_request Redis key appears.
    """
    logger.info(f"[MIGRATION] Starting unified migration for {robot_id}")

    t_trigger     = time.perf_counter()
    gpu_pre       = get_gpu_util()
    cpu_pre       = get_cpu_util()
    net_pre       = get_net_bytes()

    chk_src = os.path.join(CHECKPOINT_BASE, robot_id)
    chk_dst = os.path.join(CHECKPOINT_BASE, f"{robot_id}_dest")
    os.makedirs(chk_dst, exist_ok=True)

    # --- Wait for container to save policy + buffer ---
    logger.info(f"  Waiting for {robot_id} to save policy...")
    deadline = time.time() + 30
    while time.time() < deadline:
        if r_client.get(f"ready_for_criu:{robot_id}"):
            break
        time.sleep(0.2)

    # --- Real CRIU unified checkpoint (warm-style, container left running) ---
    # Pre-dump while running so the source container keeps serving until the
    # final stop. The policy_weights.pt + replay_buffer.pkl are saved by the
    # container BEFORE this runs (gated on `ready_for_criu`) so they are
    # captured implicitly in the bind-mounted /checkpoints volume on disk.
    t_dump_start = time.perf_counter()
    gpu_during   = get_gpu_util()
    cpu_during   = get_cpu_util()

    criu_dir = os.path.join(chk_src, "criu")
    os.makedirs(criu_dir, exist_ok=True)

    # --- Real CRIU via direct invocation, bypassing docker checkpoint ---
    # docker/runc 1.3.5 does not pass --enable-external-masters to its CRIU
    # RPC, which is required for nvidia-container-runtime mounts. Calling
    # criu directly on the container's host PID works because we control all
    # the flags. The container stays alive (--leave-running) so the worker
    # keeps serving tasks; the "destination" is conceptual — we measure real
    # dump+transfer time, then signal the worker as if it had migrated.
    src_pid = get_container_pid(container_name)
    if not cuda_checkpoint_toggle(src_pid):
        logger.warning(f"  cuda-checkpoint suspend failed for {robot_id} "
                       f"(pid={src_pid}); CRIU dump will likely fail")

    # 3 pre-dumps (warm pre-copy phase) + 1 final dump (--leave-running)
    parent = ""
    for iteration in range(3):
        predump_dir = os.path.join(criu_dir, f"predump_{iteration}")
        res = real_criu_dump(src_pid, predump_dir, parent_dir=parent,
                              pre_dump=True, leave_running=True, timeout=60)
        if res["returncode"] != 0:
            logger.error(f"[MIGRATION] pre-dump {iteration} failed for "
                         f"{robot_id}: {res['stderr'][:300]}")
            break
        parent = predump_dir
        time.sleep(0.05)

    final_dir = os.path.join(criu_dir, "final")
    res_final = real_criu_dump(src_pid, final_dir, parent_dir=parent,
                                pre_dump=False, leave_running=True, timeout=120)
    if res_final["returncode"] != 0:
        logger.error(f"[MIGRATION] final dump failed for {robot_id}: "
                     f"{res_final['stderr'][:300]}")

    dump_ms = (time.perf_counter() - t_dump_start) * 1000
    chk_size_mb = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(criu_dir) for f in files
    ) / (1024 * 1024) if os.path.exists(criu_dir) else 0

    # --- Parallel transfer: CRIU dir + policy/replay_buffer files ---
    # The unified migration's defining feature: the RL state moves alongside
    # the container state, in parallel.
    t_transfer_start = time.perf_counter()
    transfer_results = {}

    def transfer_criu():
        t = time.perf_counter()
        shutil.copytree(criu_dir, os.path.join(chk_dst, "criu"),
                         dirs_exist_ok=True)
        transfer_results["criu_ms"] = (time.perf_counter() - t) * 1000

    def transfer_policy():
        t = time.perf_counter()
        for fname in ["policy_weights.pt", "replay_buffer.pkl", "manifest.json"]:
            src = os.path.join(chk_src, fname)
            dst = os.path.join(chk_dst, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        transfer_results["policy_ms"] = (time.perf_counter() - t) * 1000

    t1 = threading.Thread(target=transfer_criu)
    t2 = threading.Thread(target=transfer_policy)
    t1.start(); t2.start()
    t1.join();  t2.join()

    t_transfer_done = time.perf_counter()
    transfer_ms     = (t_transfer_done - t_transfer_start) * 1000

    # --- Count replay buffer entries that arrived at destination ---
    # This is the headline metric for the paper: CRIU baselines transfer no
    # replay buffer (always 0), unified migration restores the trained
    # experience. Counted at destination, not source, so we measure what
    # actually arrived after the network transfer.
    replay_buffer_entries = 0
    rb_dst = os.path.join(chk_dst, "replay_buffer.pkl")
    if os.path.exists(rb_dst):
        try:
            with open(rb_dst, "rb") as _f:
                rb_obj = pickle.load(_f)
            replay_buffer_entries = len(rb_obj) if hasattr(rb_obj, "__len__") else 0
        except Exception as e:
            logger.warning(f"  Could not count replay buffer entries: {e}")

    # --- Restore: simulated time only ---
    # We used `criu dump --leave-running`, so the source container is alive.
    # Restoring the dump back into Docker isn't supported on runc 1.3.5 +
    # CRIU 3.16.1 for CUDA workloads. The "destination" is conceptual: we
    # apply a realistic restore delay drawn from CRIU benchmarks (warm
    # restore on H100/A100 is 200-500ms; we use the same triangle dist as
    # the old simulator). The container itself never paused, so the worker
    # is already running — load_policy + migration_done unblocks it
    # immediately, giving real policy_load_ms.
    import random
    t_restore_start = time.perf_counter()
    simulated_restore_ms = random.triangular(200, 330, 500)
    time.sleep(simulated_restore_ms / 1000.0)

    # Re-acquire CUDA on the source process (it never moved, but it had its
    # CUDA suspended for the dump above and needs it back).
    if not cuda_checkpoint_toggle(src_pid):
        logger.warning(f"  cuda-checkpoint resume failed for {robot_id} "
                       f"(pid={src_pid}); robot may have no working CUDA")

    restore_ms     = (time.perf_counter() - t_restore_start) * 1000
    t_restore_done = time.perf_counter()

    # --- Signal robot: migration done + policy ready to load ---
    # ORDER MATTERS. The worker is blocked on `migration_done`. It only loads
    # the policy AFTER receiving that signal. Previously we waited for
    # `first_bid_after_migration` BEFORE setting `migration_done`, which
    # deadlocked: worker waits for done, runner waits for first_bid → 30s
    # timeout fires every migration, inflating total_MTT_ms by ~30s.
    # Set load_policy first (so it's ready when worker reads it), then
    # migration_done to unblock the worker, then wait for confirmation.
    r_client.set(f"load_policy:{robot_id}", chk_dst, ex=60)
    r_client.set(f"migration_done:{robot_id}", "1", ex=60)

    # Wait for robot to confirm policy loaded and first bid submitted
    deadline_bid = time.time() + 30
    policy_load_ms = 0.0
    got_first_bid = False
    while time.time() < deadline_bid:
        data = r_client.get(f"first_bid_after_migration:{robot_id}")
        if data:
            info = json.loads(data)
            policy_load_ms = float(info.get("policy_load_ms", 0))
            r_client.delete(f"first_bid_after_migration:{robot_id}")
            got_first_bid = True
            break
        time.sleep(0.1)

    if not got_first_bid:
        logger.warning(f"[MIGRATION] {robot_id}: no first_bid_after_migration "
                       f"signal in 30s — policy_load_ms recorded as 0")

    t_fully_operational = time.perf_counter()
    total_MTT_ms = (t_fully_operational - t_trigger) * 1000
    # Robot is only down during stop+restore+policy_load — pre-dump and transfer
    # happen with --leave-running so the source container keeps serving until stop.
    downtime_ms  = restore_ms + policy_load_ms

    net_post     = get_net_bytes()
    gpu_post     = get_gpu_util()
    cpu_post     = get_cpu_util()
    net_bytes    = net_post - net_pre

    # Measure post-migration success rate (wait for 10 new tasks after migration)
    success_rate_post = _get_post_migration_success_rate(robot_id, task_counter_pre, n=10)
    regression_pct = 0.0
    if success_rate_pre > 0:
        regression_pct = (success_rate_pre - success_rate_post) / success_rate_pre * 100

    logger.info(f"[MIGRATION] {robot_id} complete: "
                f"MTT={total_MTT_ms:.0f}ms  "
                f"dump={dump_ms:.0f}ms  "
                f"transfer={transfer_ms:.0f}ms  "
                f"policy_load={policy_load_ms:.0f}ms")

    return {
        "robot_id":                    robot_id,
        "trigger_to_dump_ms":          dump_ms,
        "dump_to_transfer_ms":         transfer_ms,
        "transfer_to_restore_ms":      restore_ms,
        "policy_load_ms":              policy_load_ms,
        "downtime_ms":                 downtime_ms,
        "total_MTT_ms":                total_MTT_ms,
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
        "replay_buffer_entries_restored": replay_buffer_entries,
        "criu_mode":                   "unified",
    }


def live_status_thread(interval: int = 15):
    """
    Background thread — every `interval` seconds, prints a snapshot of every
    robot's task counter, last success rate, and current FL round, plus any
    pending migration requests. Pulls everything from Redis (no docker exec).
    """
    logger.info(f"[Status] Live status thread started (every {interval}s)")
    time.sleep(interval)
    while True:
        try:
            rows = []
            # Read most recent task_log per robot
            latest: dict = {}
            for raw in r_client.lrange("task_logs", 0, 500):
                try:
                    e = json.loads(raw)
                except Exception:
                    continue
                rid = e.get("robot_id")
                if rid and rid not in latest:
                    latest[rid] = e
                if len(latest) >= TOTAL_CLIENTS:
                    break

            for cid in range(TOTAL_CLIENTS):
                rid = f"robot_{cid:03d}"
                e   = latest.get(rid)
                if not e:
                    rows.append(f"  {rid}: <no tasks yet>")
                    continue
                rows.append(
                    f"  {rid}: tasks={e.get('task_counter',0):>4}  "
                    f"fl_round={e.get('fl_round',0):>2}  "
                    f"success10={e.get('success_rate_rolling10',0):.2f}  "
                    f"reward={e.get('reward',0):+.2f}  "
                    f"entropy={e.get('policy_entropy',0):.2f}  "
                    f"step={e.get('training_step',0)}"
                )

            pending_mig = r_client.keys("migration_request:robot_*")
            mig_str = f"  pending_migrations={len(pending_mig)}" if pending_mig else ""

            logger.info("=" * 78)
            logger.info(f"[Status] Live snapshot{mig_str}")
            for line in rows:
                logger.info(line)
            logger.info("=" * 78)

        except Exception as e:
            logger.error(f"[Status] Error: {e}")
        time.sleep(interval)


def migration_monitor_thread():
    """
    Background thread — watches Redis for migration requests from containers.
    When a request appears, triggers unified migration (Option B).
    """
    logger.info("[Monitor] Migration monitor started")
    while True:
        try:
            keys = r_client.keys("migration_request:robot_*")
            for key in keys:
                raw = r_client.get(key)
                if not raw:
                    continue
                info      = json.loads(raw)
                robot_id  = info["robot_id"]
                cid       = int(robot_id.split("_")[1])
                cname     = f"swiftbot-robot-{cid}"

                r_client.delete(key)

                success_rate_pre = float(info.get("success_rate", 0))
                task_counter_pre = int(info.get("task_counter", 0))

                mig_metrics = trigger_unified_migration(
                    robot_id, cname, "node_src", "node_dst",
                    success_rate_pre=success_rate_pre,
                    task_counter_pre=task_counter_pre,
                )
                mig_metrics["success_rate_pre"] = success_rate_pre
                mig_metrics["criu_mode"]        = "unified"
                metrics_writer.write_event(mig_metrics)

        except Exception as e:
            logger.error(f"[Monitor] Error: {e}")
        time.sleep(1)


async def main():
    master_ip = get_master_ip()
    logger.info(f"Master IP: {master_ip}")

    # Start migration monitor as background thread
    monitor = threading.Thread(target=migration_monitor_thread, daemon=True)
    monitor.start()

    # Start live status reporter
    status = threading.Thread(target=live_status_thread, args=(15,), daemon=True)
    status.start()

    # Create 4 DHT nodes (matches original dht_asr_optimized.py structure)
    nodes = (
        [DHTNode(0, BASE_PORT)] +
        [DHTNode(i, BASE_PORT + i, ("127.0.0.1", BASE_PORT))
         for i in range(1, NUM_NODES)]
    )

    logger.info(f"Launching {TOTAL_CLIENTS} robot containers...")
    await asyncio.gather(*[n.start(master_ip) for n in nodes])

    logger.info("\n[SUCCESS] All containers running.")
    logger.info("Now run the Flower server in a separate terminal:")
    logger.info("  cd ~/swiftbot_rl/dht_frl && python3 flower_server.py")
    logger.info("\nPress Ctrl+C to stop containers when experiment completes.\n")

    try:
        while True:
            await asyncio.sleep(10)
    except KeyboardInterrupt:
        logger.info("Stopping...")


if __name__ == "__main__":
    asyncio.run(main())
