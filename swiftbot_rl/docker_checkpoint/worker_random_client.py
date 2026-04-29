"""
Random policy worker — Docker Checkpoint baseline.
Identical to criu_cold/criu_warm worker. Kept as a separate file so each
condition's directory is self-contained.
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

TOTAL_TASKS = 1000
_MIGRATION_SCHEDULE = [200, 400, 600, 800, 950]
def forced_migration_tasks_for(client_id: int) -> set:
    offset = client_id * 25
    return {t + offset for t in _MIGRATION_SCHEDULE}


def _reward_for(status: str) -> float:
    return {"success": 1.0, "timeout": -0.5,
            "declined": -0.2, "failed": -1.0}.get(status, -1.0)


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
    forced_migration_tasks = forced_migration_tasks_for(args.client_id)

    logger.info(f"[{robot_id}] Random policy worker started (Docker Checkpoint baseline)")

    for task_counter in range(TOTAL_TASKS):
        if shutdown_requested:
            break

        if task_counter in forced_migration_tasks:
            sr = sum(success_hist[-10:]) / max(len(success_hist[-10:]), 1)
            r.set(f"migration_request:{robot_id}", json.dumps({
                "robot_id":     robot_id,
                "success_rate": sr,
                "task_counter": task_counter,
            }), ex=600)
            deadline = time.time() + 600
            while time.time() < deadline:
                if r.get(f"migration_done:{robot_id}"):
                    r.delete(f"migration_done:{robot_id}")
                    break
                time.sleep(0.5)

        task = task_gen.generate()
        bid  = np.random.uniform(0, 1)
        result  = task_gen.execute(task, bid=float(bid), bid_threshold=0.5)
        success = result["status"] == "success"
        success_hist.append(1 if success else 0)

        sr = sum(success_hist[-10:]) / max(len(success_hist[-10:]), 1)
        r.lpush("task_logs", json.dumps({
            "robot_id":               robot_id,
            "task_counter":           task_counter + 1,
            "fl_round":               0,
            "task_type":              task["task_type"],
            "complexity":             task["complexity"],
            "bid_value":              round(float(bid), 4),
            "reward":                 _reward_for(result["status"]),
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
            logger.info(f"[{robot_id}] Tasks: {task_counter}/{TOTAL_TASKS} success_rate={sr:.3f}")

    r.set(f"robot_done:{robot_id}", "1")
    logger.info(f"[{robot_id}] Complete.")
