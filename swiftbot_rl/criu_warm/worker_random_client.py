"""
Random policy worker — CRIU warm baseline (same as criu_cold worker).
No PPO, no FedAvg, no Flower. Just executes tasks with random bidding.
Logs task results to Redis for metric collection.
"""
import os, sys, time, json, signal, logging, argparse
import numpy as np, psutil, redis, torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

shutdown_requested = False
def signal_handler(s, f):
    global shutdown_requested
    shutdown_requested = True
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

sys.path.insert(0, "/app/robot")
from task_generator import SyntheticTaskGenerator

TOTAL_TASKS           = 1000
FORCED_MIGRATION_TASKS = set(range(200, 1050, 20))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id",      type=int, required=True)
    parser.add_argument("--container-type", type=str, default="gpu_specialist")
    args = parser.parse_args()

    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    robot_id   = f"robot_{args.client_id:03d}"
    r          = redis.Redis(host=REDIS_HOST, decode_responses=True)
    task_gen   = SyntheticTaskGenerator(args.container_type, seed=args.client_id * 100)
    success_hist = []

    logger.info(f"[{robot_id}] Random policy worker started (CRIU warm baseline)")

    for task_counter in range(TOTAL_TASKS):
        if shutdown_requested:
            break

        if task_counter in FORCED_MIGRATION_TASKS:
            sr = sum(success_hist[-10:]) / max(len(success_hist[-10:]), 1)
            r.set(f"migration_request:{robot_id}", json.dumps({
                "robot_id":     robot_id,
                "success_rate": sr,
                "task_counter": task_counter,
            }), ex=30)
            deadline = time.time() + 60
            while time.time() < deadline:
                if r.get(f"migration_done:{robot_id}"):
                    r.delete(f"migration_done:{robot_id}")
                    break
                time.sleep(0.5)

        task   = task_gen.generate()
        bid    = np.random.uniform(0, 1)   # RANDOM POLICY — no learning
        result = task_gen.execute(task)
        success = result["status"] == "success"
        success_hist.append(1 if success else 0)

        sr = sum(success_hist[-10:]) / max(len(success_hist[-10:]), 1)
        # task_counter+1 = count of completed tasks (matches DHT+FRL post-increment semantics)
        r.lpush("task_logs", json.dumps({
            "robot_id":               robot_id,
            "task_counter":           task_counter + 1,
            "fl_round":               0,
            "task_type":              task["task_type"],
            "complexity":             task["complexity"],
            "bid_value":              round(float(bid), 4),
            "reward":                 1.0 if success else -1.0,
            "status":                 result["status"],
            "exec_latency_ms":        round(result.get("latency_ms", 0), 2),
            "deadline_ms":            task["deadline_ms"],
            "success_rate_rolling10": round(sr, 4),
            "policy_entropy":         1.0,
            "training_step":          0,
        }))
        r.ltrim("task_logs", 0, 99999)
        r.setex(f"robot_load:{robot_id}", 30, json.dumps({
            "robot_id": robot_id,
            "cpu_util": psutil.cpu_percent() / 100.0,
            "task_count": task_counter,
        }))

        if task_counter % 100 == 0:
            logger.info(f"[{robot_id}] Tasks: {task_counter}/{TOTAL_TASKS} "
                        f"success_rate={sr:.3f}")

    r.set(f"robot_done:{robot_id}", "1")
    logger.info(f"[{robot_id}] Complete.")
