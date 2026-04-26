"""
Robot worker client — runs INSIDE each Docker container.
Based on worker_client_asr_optimized.py structure.
KEEP: Flower client interface, get_parameters/set_parameters,
      fit/evaluate signatures, retry logic, psutil tracking, signal handling.
REPLACE: SimpleASR → BidPolicyMLP, LibriSpeech → SyntheticTaskGenerator,
         CTC training → PPO update, ASR metrics → robot task metrics.
"""
import os, sys, gc, time, signal, logging, argparse, pickle
import psutil, numpy as np, torch
import traceback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"

import flwr as fl
from collections import OrderedDict
import redis, json

# Real GPU utilization (% busy) — replaces torch.cuda.memory_allocated which
# only reports allocated MB, not utilization. pynvml may not be available
# inside every container, so guard the import.
try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _GPU_UTIL_OK = True
except Exception:
    _GPU_HANDLE = None
    _GPU_UTIL_OK = False


def _gpu_util_pct() -> float:
    """0-100 GPU busy %. Falls back to 0 if pynvml unavailable."""
    if not _GPU_UTIL_OK:
        return 0.0
    try:
        return float(pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE).gpu)
    except Exception:
        return 0.0

sys.path.insert(0, "/app/robot")
from policy       import RobotPPOAgent, BidPolicyMLP
from sensor       import RobotSensor
from task_generator import SyntheticTaskGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

DEVICE        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_RETRIES   = 10
RETRY_DELAY   = 5
STATE_DIM     = 15
TASKS_PER_ROUND = 20        # tasks each robot does between FL sync rounds
TOTAL_ROUNDS    = 50        # matches server N_ROUNDS
# Forced migration delayed to task 200 — gives PPO ~5 updates + ~10 FL rounds first
FORCED_MIGRATION_TASKS = set(range(200, 1050, 20))

shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    logger.info(f"Signal {signum} — graceful shutdown")
    shutdown_requested = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def compute_reward(result: dict) -> float:
    """
    Reward shaping:
      success  : +1.0   (accepted and finished under deadline)
      timeout  : -0.5   (accepted but missed deadline — overcommit)
      declined : -0.2   (declined task — small penalty so policy still tries)
      failed   : -1.0   (exception / crash)
    The asymmetric penalty (decline < timeout) discourages always-decline
    while still letting the policy reject when it would otherwise time out.
    """
    status = result["status"]
    if status == "success":
        lat = result.get("latency_ms", 1e9)
        dl  = result.get("deadline_ms", 1e9)
        return +1.0 if lat <= dl else +0.3
    elif status == "timeout":
        return -0.5
    elif status == "declined":
        return -0.2
    else:
        return -1.0


class RobotClient(fl.client.NumPyClient):
    """
    Flower client wrapping a PPO robot agent.
    fit() = run TASKS_PER_ROUND tasks and do PPO updates
    evaluate() = return migration metrics + current success rate
    """

    def __init__(self, agent: RobotPPOAgent, sensor: RobotSensor,
                 task_gen: SyntheticTaskGenerator,
                 client_id: int, redis_client: redis.Redis):
        self.agent      = agent
        self.sensor     = sensor
        self.task_gen   = task_gen
        self.client_id  = client_id
        self.robot_id   = f"robot_{client_id:03d}"
        self.r          = redis_client
        self.task_counter  = 0
        self.success_hist  = []
        self.net_start     = psutil.net_io_counters().bytes_sent + \
                             psutil.net_io_counters().bytes_recv

    def get_parameters(self, config):
        return [p.cpu().numpy() for p in self.agent.policy.state_dict().values()]

    def set_parameters(self, params):
        keys = self.agent.policy.state_dict().keys()
        state_dict = OrderedDict({
            k: torch.tensor(v) for k, v in zip(keys, params)
        })
        self.agent.policy.load_state_dict(state_dict, strict=True)

    def fit(self, params, config):
        self.set_parameters(params)
        fl_round = int(config.get("round", 0))
        t_start  = time.time()

        rewards_this_round = []
        for _ in range(TASKS_PER_ROUND):
            if shutdown_requested:
                break
            task = self.task_gen.generate()
            state = self.sensor.read(task)

            # Check forced migration signal from DHT orchestrator
            if self.task_counter in FORCED_MIGRATION_TASKS:
                success_rate = (sum(self.success_hist[-10:]) /
                                max(len(self.success_hist[-10:]), 1))

                # 1. Save policy + replay buffer so CRIU captures them atomically
                chk_dir = os.path.join("/checkpoints", self.robot_id)
                os.makedirs(chk_dir, exist_ok=True)
                torch.save(self.agent.policy.state_dict(),
                           os.path.join(chk_dir, "policy_weights.pt"))
                with open(os.path.join(chk_dir, "replay_buffer.pkl"), "wb") as _f:
                    pickle.dump(self.agent.replay_buffer.tail(1000), _f)
                with open(os.path.join(chk_dir, "manifest.json"), "w") as _f:
                    json.dump({
                        "robot_id":           self.robot_id,
                        "policy_version":     self.agent.training_step,
                        "success_rate_pre":   round(success_rate, 4),
                    }, _f)

                # 2. Tell host: policy is on disk, CRIU can run now
                self.r.set(f"ready_for_criu:{self.robot_id}", "1", ex=60)

                # 3. Request migration (runner picks this up and runs CRIU)
                self.r.set(
                    f"migration_request:{self.robot_id}",
                    json.dumps({
                        "robot_id":     self.robot_id,
                        "success_rate": success_rate,
                        "task_counter": self.task_counter,
                        "fl_round":     fl_round,
                        "trigger":      "forced_experiment_event",
                    }),
                    ex=30
                )

                # 4. Wait for migration to complete (CRIU captures+restores here)
                timeout = time.time() + 120
                while time.time() < timeout:
                    done = self.r.get(f"migration_done:{self.robot_id}")
                    if done:
                        self.r.delete(f"migration_done:{self.robot_id}")
                        break
                    time.sleep(0.5)

                # 5. Load policy from destination checkpoint dir
                t_load = time.perf_counter()
                load_dir = self.r.get(f"load_policy:{self.robot_id}")
                if load_dir:
                    wpath = os.path.join(load_dir, "policy_weights.pt")
                    if os.path.exists(wpath):
                        state = torch.load(wpath, map_location=DEVICE)
                        self.agent.policy.load_state_dict(state, strict=True)
                        logger.info(f"[{self.robot_id}] Policy loaded from {load_dir}")
                    self.r.delete(f"load_policy:{self.robot_id}")
                policy_load_ms = (time.perf_counter() - t_load) * 1000

                # 6. Confirm first bid ready — runner measures policy_load_ms from this
                self.r.set(
                    f"first_bid_after_migration:{self.robot_id}",
                    json.dumps({"policy_load_ms": round(policy_load_ms, 2)}),
                    ex=60
                )

            # Get bid; bid gates whether task is actually executed
            bid    = self.agent.get_bid(state)
            result = self.task_gen.execute(task, bid=bid, bid_threshold=0.5)
            reward = compute_reward(result)

            next_state = self.sensor.read(task)
            self.agent.record_reward(reward, next_state)
            self.sensor.record_outcome(task["task_type"], result["status"] == "success")
            self.success_hist.append(1 if result["status"] == "success" else 0)
            rewards_this_round.append(reward)
            self.task_counter += 1

            # Log to Redis for collection
            sr = sum(self.success_hist[-10:]) / max(len(self.success_hist[-10:]), 1)
            self.r.lpush("task_logs", json.dumps({
                "robot_id":              self.robot_id,
                "task_counter":          self.task_counter,
                "fl_round":              fl_round,
                "task_type":             task["task_type"],
                "complexity":            task["complexity"],
                "duration_s":            task["duration_s"],
                "bid_value":             round(bid, 4),
                "reward":                round(reward, 4),
                "status":                result["status"],
                "exec_latency_ms":       round(result.get("latency_ms", 0), 2),
                "deadline_ms":           task["deadline_ms"],
                "success_rate_rolling10": round(sr, 4),
                "policy_entropy":        round(self.agent.get_entropy(), 4),
                "training_step":         self.agent.training_step,
            }))
            self.r.ltrim("task_logs", 0, 99999)

            # Publish load for DHT migration monitoring
            import psutil as _ps
            self.r.setex(f"robot_load:{self.robot_id}", 30, json.dumps({
                "robot_id":   self.robot_id,
                "cpu_util":   _ps.cpu_percent() / 100.0,
                "task_count": self.task_counter,
            }))

        # Hardware metrics — gpu_usage is now real % utilization (pynvml),
        # not allocated MB. Old metric was stuck at ~8 MB regardless of load
        # so the fl_hardware.csv chart was a flat line.
        cpu_usage = psutil.cpu_percent()
        gpu_usage = _gpu_util_pct()
        net_now    = psutil.net_io_counters()
        net_mb     = (net_now.bytes_sent + net_now.bytes_recv - self.net_start) / (1024 * 1024)
        self.net_start = net_now.bytes_sent + net_now.bytes_recv

        mean_reward  = float(np.mean(rewards_this_round)) if rewards_this_round else 0.0
        success_rate = sum(self.success_hist[-20:]) / max(len(self.success_hist[-20:]), 1)
        # Real PPO total_loss from the last optimizer step (was a fake
        # proxy `1 - success_rate` that read 0 once tasks were succeeding).
        train_loss   = float(self.agent.last_loss)
        train_time   = float(time.time() - t_start)

        logger.info(f"[{self.robot_id}] Round {fl_round}: "
                    f"tasks={self.task_counter} success={success_rate:.3f} "
                    f"reward={mean_reward:+.3f} loss={train_loss:.4f} "
                    f"entropy={self.agent.get_entropy():.3f} "
                    f"time={train_time:.1f}s")

        return self.get_parameters({}), self.task_counter, {
            "train_loss":     train_loss,
            "mean_reward":    float(mean_reward),
            "success_rate":   float(success_rate),
            "policy_entropy": float(self.agent.get_entropy()),
            "train_time":     train_time,
            "cpu_usage":      float(cpu_usage),
            "gpu_usage":      float(gpu_usage),
            "network_mb":     float(net_mb),
        }

    def evaluate(self, params, config):
        self.set_parameters(params)
        sr = sum(self.success_hist[-20:]) / max(len(self.success_hist[-20:]), 1)
        loss = max(0.0, 1.0 - sr)
        return float(loss), max(self.task_counter, 1), {
            "accuracy":        float(sr),
            "loss":            float(loss),
            "eval_time":       0.1,
            "success_rate":    float(sr),
            "policy_entropy":  float(self.agent.get_entropy()),
            "cpu_usage":       float(psutil.cpu_percent()),
            "gpu_usage":       _gpu_util_pct(),
        }


def cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id",    type=int, required=True)
    parser.add_argument("--num-clients",  type=int, default=8)
    parser.add_argument("--container-type", type=str,
                        choices=["gpu_specialist", "cpu_specialist"],
                        default="gpu_specialist")
    args = parser.parse_args()

    SERVER    = os.getenv("MASTER_ADDRESS", "127.0.0.1:8080")
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

    logger.info(f"[robot_{args.client_id:03d}] Starting — "
                f"server={SERVER} type={args.container_type}")

    r         = redis.Redis(host=REDIS_HOST, decode_responses=True)
    agent     = RobotPPOAgent(robot_id=f"robot_{args.client_id:03d}")
    sensor    = RobotSensor(robot_id=f"robot_{args.client_id:03d}")
    task_gen  = SyntheticTaskGenerator(container_type=args.container_type,
                                        seed=args.client_id * 100)
    client    = RobotClient(agent, sensor, task_gen, args.client_id, r)

    retry = 0
    while retry < MAX_RETRIES and not shutdown_requested:
        try:
            fl.client.start_numpy_client(
                server_address=SERVER,
                client=client
            )
            logger.info(f"[robot_{args.client_id:03d}] All FL rounds complete")
            break
        except KeyboardInterrupt:
            shutdown_requested = True
            break
        except Exception as e:
            retry += 1
            logger.warning(f"Connection failed ({e}). Retry {retry}/{MAX_RETRIES} "
                           f"in {RETRY_DELAY}s...")
            cleanup()
            time.sleep(RETRY_DELAY)

    cleanup()
    logger.info(f"[robot_{args.client_id:03d}] Shutdown complete")
