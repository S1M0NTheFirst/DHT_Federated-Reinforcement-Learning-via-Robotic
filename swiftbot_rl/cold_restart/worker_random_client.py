"""
Random policy worker — Cold Restart baseline.
Identical to other baseline workers. Cold restart kills + relaunches the
container with NO state preservation, so on restart the worker starts
fresh from task_counter=0. The CSV will show two "lives" per migration —
the runner stitches them via robot_id when computing success_rate_post.
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


def _reward_for(status):
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

    # Cold restart resumes task_counter from where the previous life left
    # off — the runner stores it in redis before killing the container.
    start_counter = int(r.get(f"resume_counter:{robot_id}") or 0)
    if start_counter:
        logger.info(f"[{robot_id}] Resuming after cold restart at task {start_counter} "
                    f"(NO state preserved — random policy fresh)")
        r.delete(f"resume_counter:{robot_id}")
    else:
        logger.info(f"[{robot_id}] Random policy worker started (Cold Restart baseline)")

    for task_counter in range(start_counter, TOTAL_TASKS):
        if shutdown_requested:
            break

        if task_counter in forced_migration_tasks:
            sr = sum(success_hist[-10:]) / max(len(success_hist[-10:]), 1)
            # Tell runner where to resume after the kill+restart.
            r.set(f"resume_counter:{robot_id}", task_counter + 1, ex=600)
            r.set(f"migration_request:{robot_id}", json.dumps({
                "robot_id":     robot_id,
                "success_rate": sr,
                "task_counter": task_counter,
            }), ex=600)
            # The runner will SIGKILL us. We just exit cleanly first if
            # given the chance — either way, all in-RAM state (success_hist,
            # numpy RNG state) is lost on restart. That's the whole point.
            deadline = time.time() + 600
            while time.time() < deadline:
                if r.get(f"migration_done:{robot_id}"):
                    r.delete(f"migration_done:{robot_id}")
                    break
                time.sleep(0.5)
            # If we make it past the wait without being killed (shouldn't),
            # break out so we don't double-execute this task.
            break

        task = task_gen.generate()
        bid  = np.random.uniform(0, 1)
        result = task_gen.execute(task, bid=float(bid), bid_threshold=0.5)
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

    if not r.get(f"resume_counter:{robot_id}"):
        # No pending restart → we're truly done.
        r.set(f"robot_done:{robot_id}", "1")
        logger.info(f"[{robot_id}] Complete.")
