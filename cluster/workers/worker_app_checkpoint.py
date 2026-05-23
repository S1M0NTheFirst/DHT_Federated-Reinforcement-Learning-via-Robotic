"""
Cluster-local worker with application-level checkpointing.

Used by Conditions C (app-cold) and D (app-warm) on the cluster. Replaces
the kernel-CRIU mechanism with PyTorch-level state save/load, which works
without root and without kernel features unavailable on shared HPCs.

Behavior identical to swiftbot_rl/cold_restart/worker_random_client.py
(random-policy bidding, forced migrations at offset_per_client + schedule)
*plus*:

  - Maintains a synthetic torch state (model + optimizer + replay buffer +
    RNG state) sized to ~20 MB serialized — matches a small PPO agent.
    The model is NEVER used to make bids (bids stay random); it exists
    purely so the dump/transfer/restore numbers reflect realistic state size.

  - On migration request: torch.save's that state to APP_CHECKPOINT_PATH
    (default /checkpoints/state.pt), then publishes timing+size back to
    Redis under app_checkpoint_done:<robot> before sleeping.

  - On startup with APP_RESTORE_FROM set: torch.load's the file, restores
    counters/RNG, publishes load_ms back to Redis under app_restore_done:<robot>.

  - If WARM_CHECKPOINT_PATH + WARM_CHECKPOINT_INTERVAL are set, the worker
    writes a periodic snapshot every N tasks during normal operation. The
    runner's pre-copy thread rsyncs that snapshot between nodes so the
    migration-time delta is small (Condition D).

The workstation cold_restart worker is unchanged — this is a separate file
so swiftbot_rl/ stays frozen.

Env vars that govern behavior:
  APP_CHECKPOINT_PATH       — where to dump on migration request (cond C/D)
  APP_RESTORE_FROM          — load this file at startup (cond C/D after migration)
  WARM_CHECKPOINT_PATH      — periodic snapshot path (cond D)
  WARM_CHECKPOINT_INTERVAL  — task interval between warm snapshots (cond D)
  TOTAL_TASKS               — total task budget per robot (default 1200)
  MIGRATION_OFFSET          — task-counter offset per client (default 10)
  REDIS_HOST / REDIS_PORT   — Redis location

CLI args (compat with launch_robot in cluster_runner.py):
  --client-id <int>         (required)
  --num-clients <int>       (currently unused, kept for compatibility)
  --container-type <str>    (passed to SyntheticTaskGenerator; default gpu_specialist)
"""
import os, sys, time, json, signal, logging, argparse
import numpy as np, psutil, redis, torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

shutdown_requested = False
def _on_signal(s, f):
    global shutdown_requested
    shutdown_requested = True
signal.signal(signal.SIGTERM, _on_signal)
signal.signal(signal.SIGINT, _on_signal)

# /app is the swiftbot_rl bind — we still need its task_generator.
sys.path.insert(0, "/app/robot")
from task_generator import SyntheticTaskGenerator   # noqa: E402

TOTAL_TASKS         = int(os.environ.get("TOTAL_TASKS", "1200"))
_MIGRATION_SCHEDULE = [200, 400, 600, 800, 950]
_MIGRATION_OFFSET   = int(os.environ.get("MIGRATION_OFFSET", "10"))

APP_CHECKPOINT_PATH      = os.environ.get("APP_CHECKPOINT_PATH")
APP_RESTORE_FROM         = os.environ.get("APP_RESTORE_FROM")
WARM_CHECKPOINT_PATH     = os.environ.get("WARM_CHECKPOINT_PATH")
WARM_CHECKPOINT_INTERVAL = int(os.environ.get("WARM_CHECKPOINT_INTERVAL", "0"))


def _forced_tasks_for(client_id: int) -> set:
    return {t + client_id * _MIGRATION_OFFSET for t in _MIGRATION_SCHEDULE}


def _reward_for(status):
    return {"success": 1.0, "timeout": -0.5,
            "declined": -0.2, "failed": -1.0}.get(status, -1.0)


def _build_synthetic_state(client_id: int):
    """Build the synthetic (model, optimizer, replay_buffer) state.
    Sized so the serialized checkpoint is ~20 MB (matches a small PPO).
    """
    torch.manual_seed(client_id * 7919)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 32),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Take one step so Adam's exp_avg/exp_avg_sq state is populated (otherwise
    # state_dict() returns the empty initial slot for these moments).
    dummy = torch.randn(4, 64)
    loss = model(dummy).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # ~4 MB filler tensor representing a typical short rollout buffer.
    replay_buffer = {"obs": torch.randn(1024, 1024), "tasks": []}
    return model, optimizer, replay_buffer


def _save_state(model, optimizer, replay_buffer, task_counter, success_hist, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "model":           model.state_dict(),
        "optimizer":       optimizer.state_dict(),
        "replay_buffer":   replay_buffer,
        "task_counter":    task_counter,
        "success_hist":    success_hist,
        "np_rng_state":    np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }, tmp)
    os.replace(tmp, path)  # atomic — readers either see old or new, never partial


def _load_state(path):
    # weights_only=False: our checkpoint contains numpy RNG state (a tuple
    # with numpy.ndarray inside), which PyTorch 2.6+ rejects under the new
    # weights_only=True default. The file is ours and trusted, so explicitly
    # opt out — otherwise restore fails with UnpicklingError and the worker
    # dies before publishing app_restore_done.
    return torch.load(path, map_location="cpu", weights_only=False)


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
    task_gen   = SyntheticTaskGenerator(args.container_type, seed=args.client_id * 100)
    forced     = _forced_tasks_for(args.client_id)

    # --- Initialize or restore synthetic state -----------------------------
    model, optimizer, replay_buffer = _build_synthetic_state(args.client_id)
    success_hist = []
    start_counter = int(r.get(f"resume_counter:{robot_id}") or 0)

    if APP_RESTORE_FROM and os.path.exists(APP_RESTORE_FROM):
        t0 = time.perf_counter()
        ckpt = _load_state(APP_RESTORE_FROM)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        replay_buffer = ckpt.get("replay_buffer", replay_buffer)
        np.random.set_state(ckpt["np_rng_state"])
        torch.set_rng_state(ckpt["torch_rng_state"])
        success_hist = ckpt.get("success_hist", [])
        # The saved task_counter IS the migration task — we already triggered
        # migration at that step. If we resume at the same value, the next
        # iteration of the for-loop sees `task_counter in forced` again and
        # fires another migration immediately, creating an infinite loop.
        # Advance past it so we continue with the NEXT real task.
        start_counter = ckpt.get("task_counter", start_counter) + 1
        load_ms = (time.perf_counter() - t0) * 1000
        size_mb = os.path.getsize(APP_RESTORE_FROM) / (1024 * 1024)
        _publish(r, f"app_restore_done:{robot_id}",
                 {"load_ms": load_ms, "size_mb": size_mb,
                  "task_counter": start_counter,
                  "replay_buffer_entries": len(replay_buffer.get("tasks", []))})
        logger.info(f"[{robot_id}] Restored from {APP_RESTORE_FROM} "
                    f"({load_ms:.1f}ms, {size_mb:.2f}MB) → resuming at task {start_counter}")
        r.delete(f"resume_counter:{robot_id}")
    elif start_counter:
        logger.info(f"[{robot_id}] Resuming via resume_counter at task {start_counter} "
                    f"(no app-state file)")
        r.delete(f"resume_counter:{robot_id}")
    else:
        logger.info(f"[{robot_id}] Started fresh "
                    f"(app_ckpt={APP_CHECKPOINT_PATH or 'off'}, "
                    f"warm={WARM_CHECKPOINT_PATH or 'off'}/{WARM_CHECKPOINT_INTERVAL})")

    # --- Main task loop -----------------------------------------------------
    for task_counter in range(start_counter, TOTAL_TASKS):
        if shutdown_requested:
            break

        # Forced migration: dump app-state (if enabled) and sit waiting.
        if task_counter in forced:
            sr = sum(success_hist[-10:]) / max(len(success_hist[-10:]), 1)
            r.set(f"resume_counter:{robot_id}", task_counter + 1, ex=600)

            if APP_CHECKPOINT_PATH:
                t0 = time.perf_counter()
                _save_state(model, optimizer, replay_buffer,
                            task_counter, success_hist, APP_CHECKPOINT_PATH)
                save_ms = (time.perf_counter() - t0) * 1000
                size_mb = os.path.getsize(APP_CHECKPOINT_PATH) / (1024 * 1024)
                _publish(r, f"app_checkpoint_done:{robot_id}",
                         {"save_ms": save_ms, "size_mb": size_mb,
                          "path": APP_CHECKPOINT_PATH,
                          "task_counter": task_counter,
                          "replay_buffer_entries": len(replay_buffer.get("tasks", []))})
                logger.info(f"[{robot_id}] App-checkpoint saved "
                            f"({save_ms:.1f}ms, {size_mb:.2f}MB)")

            # 1h TTL: when many robots request migration in the same wave
            # the runner's serial trigger_fn (≈90s each) can take 15+ minutes
            # to drain the queue. With ex=600 (10 min), tail-of-queue requests
            # silently expired before being processed and the worker sat in
            # the sleep-forever loop below with no one to kill it.
            r.set(f"migration_request:{robot_id}", json.dumps({
                "robot_id":     robot_id,
                "success_rate": sr,
                "task_counter": task_counter,
            }), ex=3600)
            logger.info(f"[{robot_id}] migration requested at task {task_counter}, "
                        f"sleeping until runner kills me")
            while True:
                time.sleep(5)

        # --- Normal task ----------------------------------------------------
        task = task_gen.generate()
        bid  = np.random.uniform(0, 1)
        result = task_gen.execute(task, bid=float(bid), bid_threshold=0.5)
        success = result["status"] == "success"
        success_hist.append(1 if success else 0)
        if len(success_hist) > 1000:
            success_hist = success_hist[-1000:]

        # Append to replay buffer (capped — keeps state size bounded).
        rb_tasks = replay_buffer.get("tasks", [])
        rb_tasks.append({
            "bid": float(bid),
            "reward": _reward_for(result["status"]),
            "status": result["status"],
        })
        if len(rb_tasks) > 256:
            rb_tasks = rb_tasks[-256:]
        replay_buffer["tasks"] = rb_tasks

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

        # Periodic warm snapshot (Condition D).
        if (WARM_CHECKPOINT_PATH and WARM_CHECKPOINT_INTERVAL > 0
                and task_counter > 0
                and task_counter % WARM_CHECKPOINT_INTERVAL == 0):
            try:
                _save_state(model, optimizer, replay_buffer,
                            task_counter, success_hist, WARM_CHECKPOINT_PATH)
            except Exception as e:
                logger.warning(f"[{robot_id}] warm snapshot failed: {e}")

        if task_counter % 100 == 0:
            logger.info(f"[{robot_id}] Tasks: {task_counter}/{TOTAL_TASKS} "
                        f"success_rate={sr:.3f}")

    if not r.get(f"resume_counter:{robot_id}"):
        r.set(f"robot_done:{robot_id}", "1")
        logger.info(f"[{robot_id}] Complete.")


if __name__ == "__main__":
    main()
