"""
Cluster-local worker with application-level checkpointing — REAL PPO, no
federation.

Used by Conditions C (app-cold), D (app-warm), and reused by F (concurrent)
and G (failure injection). This is the apples-to-apples baseline for the DHT
worker (Condition A): it runs the SAME RobotPPOAgent / BidPolicyMLP and the
SAME task workload, the ONLY differences being:

  - migration transport is application-level torch checkpoint + rsync
    (vs A's DHT bundle), and
  - there is NO FedAvg / DHT weight sharing — each robot trains in isolation.

This makes the migration payload the same small policy (~70 KB) for every
condition, so the latency / bandwidth comparison is clean (per advisor
feedback). It also lets the recovery-curve metric mean something: because the
policy actually learns, a stale/lost restore visibly regresses success rate.

Behavior vs the previous version (which used a 17 MB synthetic blob + random
bids): the model is now load-bearing. Bids come from the policy; rewards drive
PPO updates; the checkpoint is the policy + optimizer + replay buffer.

Env vars that govern behavior:
  WORKER_PRETRAINED_PATH    — load this RobotPPOAgent checkpoint at fresh start
  APP_CHECKPOINT_PATH       — where to dump on migration request (cond C/D)
  APP_RESTORE_FROM          — load this file at startup (after migration)
  WARM_CHECKPOINT_PATH      — periodic snapshot path (cond D)
  WARM_CHECKPOINT_INTERVAL  — task interval between warm snapshots (cond D)
  TOTAL_TASKS               — total task budget per robot (default 1200)
  MIGRATION_OFFSET          — task-counter offset per client (default 10)
  REDIS_HOST / REDIS_PORT   — Redis location

CLI args (compat with launch_robot in cluster_runner.py):
  --client-id <int>         (required)
  --num-clients <int>       (unused, kept for compatibility)
  --container-type <str>    (gpu_specialist | cpu_specialist)
"""
import os, sys, time, json, signal, pickle, logging, argparse
import numpy as np, psutil, redis, torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

shutdown_requested = False
def _on_signal(s, f):
    global shutdown_requested
    shutdown_requested = True
signal.signal(signal.SIGTERM, _on_signal)
signal.signal(signal.SIGINT, _on_signal)

# Robot modules (policy/sensor/task_generator) live in swiftbot_rl and are
# bound into the container at /robot_lib (also reachable via /app/robot). Add
# both so this worker runs inside the container AND when smoke-tested locally.
for _p in ("/robot_lib", "/app/robot",
           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "swiftbot_rl", "dht_frl", "robot")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
from policy import RobotPPOAgent          # noqa: E402
from sensor import RobotSensor            # noqa: E402
from task_generator import SyntheticTaskGenerator  # noqa: E402

STATE_DIM           = 15
TOTAL_TASKS         = int(os.environ.get("TOTAL_TASKS", "1200"))
_MIGRATION_SCHEDULE = [200, 400, 600, 800, 950]
_MIGRATION_OFFSET   = int(os.environ.get("MIGRATION_OFFSET", "10"))

WORKER_PRETRAINED_PATH   = os.environ.get("WORKER_PRETRAINED_PATH")
APP_CHECKPOINT_PATH      = os.environ.get("APP_CHECKPOINT_PATH")
APP_RESTORE_FROM         = os.environ.get("APP_RESTORE_FROM")
WARM_CHECKPOINT_PATH     = os.environ.get("WARM_CHECKPOINT_PATH")
WARM_CHECKPOINT_INTERVAL = int(os.environ.get("WARM_CHECKPOINT_INTERVAL", "0"))


def _forced_tasks_for(client_id: int) -> set:
    return {t + client_id * _MIGRATION_OFFSET for t in _MIGRATION_SCHEDULE}


def compute_reward(result: dict) -> float:
    """Same reward shaping as the DHT worker (swiftbot_rl/dht_frl), so the two
    workers optimise the identical objective."""
    status = result["status"]
    if status == "success":
        lat = result.get("latency_ms", 1e9)
        dl  = result.get("deadline_ms", 1e9)
        return +1.5 if lat <= dl else +0.5
    if status == "timeout":
        return -0.3
    if status == "declined":
        return -0.4
    return -1.0


def _save_state(agent: RobotPPOAgent, sensor: RobotSensor,
                task_counter: int, success_hist: list, path: str):
    """Serialize the full PPO agent state. Small (~70 KB) — policy + optimizer
    + a bounded replay tail + counters/RNG."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "policy_state_dict":    agent.policy.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "training_step":        agent.training_step,
        "replay_tail":          agent.replay_buffer.tail(1000),
        "task_counter":         task_counter,
        "success_hist":         success_hist,
        "gpu_success_history":  sensor._gpu_success_history,
        "cpu_success_history":  sensor._cpu_success_history,
        "np_rng_state":         np.random.get_state(),
        "torch_rng_state":      torch.get_rng_state(),
    }, tmp)
    os.replace(tmp, path)  # atomic — readers see old or new, never partial


def _restore_state(agent: RobotPPOAgent, sensor: RobotSensor, path: str) -> dict:
    # weights_only=False: the checkpoint carries numpy RNG state (ndarray in a
    # tuple) which PyTorch 2.6+ rejects under the weights_only=True default.
    # The file is ours and trusted.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    agent.policy.load_state_dict(ckpt["policy_state_dict"])
    agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    agent.training_step = ckpt.get("training_step", 0)
    if ckpt.get("replay_tail"):
        agent.replay_buffer.load_tail(ckpt["replay_tail"])
    sensor._gpu_success_history = ckpt.get("gpu_success_history", [])
    sensor._cpu_success_history = ckpt.get("cpu_success_history", [])
    np.random.set_state(ckpt["np_rng_state"])
    torch.set_rng_state(ckpt["torch_rng_state"])
    return ckpt


def _load_pretrained(agent: RobotPPOAgent, path: str) -> bool:
    """Load the shared pretrained policy as the starting point. Same format as
    RobotPPOAgent.save_checkpoint (policy + optimizer + training_step)."""
    try:
        agent.load_checkpoint(path)
        return True
    except Exception as e:
        logger.warning(f"pretrained load from {path} failed: {e!r} — "
                       f"starting from random init")
        return False


def _publish(r, key: str, payload: dict, ttl: int = 600):
    r.set(key, json.dumps(payload), ex=ttl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id",      type=int, required=True)
    parser.add_argument("--num-clients",    type=int, default=20)
    parser.add_argument("--container-type", type=str, default="gpu_specialist")
    args = parser.parse_args()

    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    robot_id   = f"robot_{args.client_id:03d}"
    r          = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

    agent    = RobotPPOAgent(state_dim=STATE_DIM, robot_id=robot_id)
    sensor   = RobotSensor(robot_id=robot_id)
    task_gen = SyntheticTaskGenerator(args.container_type, seed=args.client_id * 100)
    forced   = _forced_tasks_for(args.client_id)

    success_hist = []
    start_counter = int(r.get(f"resume_counter:{robot_id}") or 0)

    # --- Initialize or restore --------------------------------------------
    if APP_RESTORE_FROM and os.path.exists(APP_RESTORE_FROM):
        t0 = time.perf_counter()
        ckpt = _restore_state(agent, sensor, APP_RESTORE_FROM)
        success_hist = ckpt.get("success_hist", [])
        # The saved task_counter IS the migration task — resume at the NEXT one
        # or the for-loop re-fires the migration immediately (infinite loop).
        start_counter = ckpt.get("task_counter", start_counter) + 1
        load_ms = (time.perf_counter() - t0) * 1000
        size_mb = os.path.getsize(APP_RESTORE_FROM) / (1024 * 1024)
        _publish(r, f"app_restore_done:{robot_id}",
                 {"load_ms": load_ms, "size_mb": size_mb,
                  "task_counter": start_counter,
                  "replay_buffer_entries": len(agent.replay_buffer)})
        logger.info(f"[{robot_id}] Restored from {APP_RESTORE_FROM} "
                    f"({load_ms:.1f}ms, {size_mb:.2f}MB) → resuming at task {start_counter}")
        r.delete(f"resume_counter:{robot_id}")
    elif start_counter:
        logger.info(f"[{robot_id}] Resuming via resume_counter at task {start_counter} "
                    f"(no app-state file)")
        if WORKER_PRETRAINED_PATH and os.path.exists(WORKER_PRETRAINED_PATH):
            _load_pretrained(agent, WORKER_PRETRAINED_PATH)
        r.delete(f"resume_counter:{robot_id}")
    else:
        loaded = False
        if WORKER_PRETRAINED_PATH and os.path.exists(WORKER_PRETRAINED_PATH):
            loaded = _load_pretrained(agent, WORKER_PRETRAINED_PATH)
        logger.info(f"[{robot_id}] Started fresh "
                    f"(pretrained={'yes' if loaded else 'no'}, "
                    f"app_ckpt={APP_CHECKPOINT_PATH or 'off'}, "
                    f"warm={WARM_CHECKPOINT_PATH or 'off'}/{WARM_CHECKPOINT_INTERVAL})")

    # --- Main task loop ----------------------------------------------------
    for task_counter in range(start_counter, TOTAL_TASKS):
        if shutdown_requested:
            break

        # Forced migration: dump app-state (if enabled) and sit waiting.
        if task_counter in forced:
            sr = sum(success_hist[-10:]) / max(len(success_hist[-10:]), 1)
            r.set(f"resume_counter:{robot_id}", task_counter + 1, ex=600)

            if APP_CHECKPOINT_PATH:
                t0 = time.perf_counter()
                _save_state(agent, sensor, task_counter, success_hist,
                            APP_CHECKPOINT_PATH)
                save_ms = (time.perf_counter() - t0) * 1000
                size_mb = os.path.getsize(APP_CHECKPOINT_PATH) / (1024 * 1024)
                _publish(r, f"app_checkpoint_done:{robot_id}",
                         {"save_ms": save_ms, "size_mb": size_mb,
                          "path": APP_CHECKPOINT_PATH,
                          "task_counter": task_counter,
                          "replay_buffer_entries": len(agent.replay_buffer)})
                logger.info(f"[{robot_id}] App-checkpoint saved "
                            f"({save_ms:.1f}ms, {size_mb:.3f}MB)")

            # 1h TTL: a migration wave can take >10 min to drain through the
            # runner's serial trigger_fn; a shorter TTL silently expired
            # tail-of-queue requests, stranding the worker in the sleep loop.
            r.set(f"migration_request:{robot_id}", json.dumps({
                "robot_id":     robot_id,
                "success_rate": sr,
                "task_counter": task_counter,
            }), ex=3600)
            logger.info(f"[{robot_id}] migration requested at task {task_counter}, "
                        f"sleeping until runner kills me")
            while True:
                time.sleep(5)

        # --- Normal task ---------------------------------------------------
        task  = task_gen.generate()
        state = sensor.read(task)
        bid   = agent.get_bid(state)
        result = task_gen.execute(task, bid=float(bid), bid_threshold=0.5)
        reward = compute_reward(result)
        next_state = sensor.read(task)
        agent.record_reward(reward, next_state)           # drives PPO updates
        sensor.record_outcome(task["task_type"], result["status"] == "success")

        success = result["status"] == "success"
        success_hist.append(1 if success else 0)
        if len(success_hist) > 1000:
            success_hist = success_hist[-1000:]

        sr = sum(success_hist[-10:]) / max(len(success_hist[-10:]), 1)
        r.lpush("task_logs", json.dumps({
            "robot_id":               robot_id,
            "task_counter":           task_counter + 1,
            "fl_round":               0,
            "task_type":              task["task_type"],
            "complexity":             task["complexity"],
            "bid_value":              round(float(bid), 4),
            "reward":                 round(reward, 4),
            "status":                 result["status"],
            "exec_latency_ms":        round(result.get("latency_ms", 0), 2),
            "deadline_ms":            task["deadline_ms"],
            "success_rate_rolling10": round(sr, 4),
            "policy_entropy":         round(agent.get_entropy(), 4),
            "training_step":          agent.training_step,
        }))
        r.ltrim("task_logs", 0, 99999)
        r.setex(f"robot_load:{robot_id}", 30, json.dumps({
            "robot_id": robot_id,
            "cpu_util": psutil.cpu_percent() / 100.0,
            "task_count": task_counter,
        }))

        # Periodic warm snapshot (Condition D).
        if (WARM_CHECKPOINT_PATH and WARM_CHECKPOINT_INTERVAL > 0
                and task_counter > 0
                and task_counter % WARM_CHECKPOINT_INTERVAL == 0):
            try:
                _save_state(agent, sensor, task_counter, success_hist,
                            WARM_CHECKPOINT_PATH)
            except Exception as e:
                logger.warning(f"[{robot_id}] warm snapshot failed: {e}")

        if task_counter % 25 == 0:
            logger.info(f"[{robot_id}] Tasks: {task_counter}/{TOTAL_TASKS} "
                        f"success_rate={sr:.3f} step={agent.training_step}")

    if not r.get(f"resume_counter:{robot_id}"):
        r.set(f"robot_done:{robot_id}", "1")
        logger.info(f"[{robot_id}] Complete.")


if __name__ == "__main__":
    main()
