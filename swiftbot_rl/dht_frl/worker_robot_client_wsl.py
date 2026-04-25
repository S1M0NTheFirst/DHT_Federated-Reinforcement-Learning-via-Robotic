"""
Robot worker client — WSL2 version.
Identical to worker_robot_client.py except the robot module path
is resolved relative to this file instead of the Docker /app/robot hardcode.
"""
import os, sys

# Resolve robot/ relative to this file (replaces hardcoded /app/robot)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot"))

import gc, time, signal, logging, argparse
import psutil, numpy as np, torch
import traceback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"

import flwr as fl
from collections import OrderedDict
import redis, json

from policy       import RobotPPOAgent, BidPolicyMLP
from sensor       import RobotSensor
from task_generator import SyntheticTaskGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

DEVICE        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_RETRIES   = 10
RETRY_DELAY   = 5
STATE_DIM     = 15
TASKS_PER_ROUND = 20
TOTAL_ROUNDS    = 30
FORCED_MIGRATION_TASKS = set(range(50, 1050, 20))

shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    logger.info(f"Signal {signum} — graceful shutdown")
    shutdown_requested = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def compute_reward(result: dict) -> float:
    if result["status"] == "success":
        lat = result.get("latency_ms", 1e9)
        dl  = result.get("deadline_ms", 1e9)
        return +1.0 if lat <= dl else +0.3
    elif result["status"] == "timeout":
        return -0.5
    else:
        return -1.0


class RobotClient(fl.client.NumPyClient):
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
            task  = self.task_gen.generate()
            state = self.sensor.read(task)

            if self.task_counter in FORCED_MIGRATION_TASKS:
                success_rate = (sum(self.success_hist[-10:]) /
                                max(len(self.success_hist[-10:]), 1))
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
                timeout = time.time() + 60
                while time.time() < timeout:
                    done = self.r.get(f"migration_done:{self.robot_id}")
                    if done:
                        self.r.delete(f"migration_done:{self.robot_id}")
                        break
                    time.sleep(0.5)

            bid    = self.agent.get_bid(state)
            result = self.task_gen.execute(task)
            reward = compute_reward(result)

            next_state = self.sensor.read(task)
            self.agent.record_reward(reward, next_state)
            self.sensor.record_outcome(task["task_type"], result["status"] == "success")
            self.success_hist.append(1 if result["status"] == "success" else 0)
            rewards_this_round.append(reward)
            self.task_counter += 1

            sr = sum(self.success_hist[-10:]) / max(len(self.success_hist[-10:]), 1)
            self.r.lpush("task_logs", json.dumps({
                "robot_id":               self.robot_id,
                "task_counter":           self.task_counter,
                "fl_round":               fl_round,
                "task_type":              task["task_type"],
                "complexity":             task["complexity"],
                "duration_s":             task["duration_s"],
                "bid_value":              round(bid, 4),
                "reward":                 round(reward, 4),
                "status":                 result["status"],
                "exec_latency_ms":        round(result.get("latency_ms", 0), 2),
                "deadline_ms":            task["deadline_ms"],
                "success_rate_rolling10": round(sr, 4),
                "policy_entropy":         round(self.agent.get_entropy(), 4),
                "training_step":          self.agent.training_step,
            }))
            self.r.ltrim("task_logs", 0, 99999)
            self.r.setex(f"robot_load:{self.robot_id}", 30, json.dumps({
                "robot_id":   self.robot_id,
                "cpu_util":   psutil.cpu_percent() / 100.0,
                "task_count": self.task_counter,
            }))

        cpu_usage = psutil.cpu_percent()
        gpu_usage = 0.0
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated(DEVICE) / (1024 * 1024)
        net_now    = psutil.net_io_counters()
        net_mb     = (net_now.bytes_sent + net_now.bytes_recv - self.net_start) / (1024 * 1024)
        self.net_start = net_now.bytes_sent + net_now.bytes_recv

        mean_reward  = float(np.mean(rewards_this_round)) if rewards_this_round else 0.0
        success_rate = sum(self.success_hist[-20:]) / max(len(self.success_hist[-20:]), 1)
        train_loss   = max(0.0, 1.0 - success_rate)

        logger.info(f"[{self.robot_id}] Round {fl_round}: "
                    f"tasks={self.task_counter} success={success_rate:.3f} "
                    f"entropy={self.agent.get_entropy():.3f}")

        return self.get_parameters({}), self.task_counter, {
            "train_loss":     float(train_loss),
            "mean_reward":    float(mean_reward),
            "success_rate":   float(success_rate),
            "policy_entropy": float(self.agent.get_entropy()),
            "train_time":     float(time.time() - t_start),
            "cpu_usage":      float(cpu_usage),
            "gpu_usage":      float(gpu_usage),
            "network_mb":     float(net_mb),
        }

    def evaluate(self, params, config):
        self.set_parameters(params)
        sr   = sum(self.success_hist[-20:]) / max(len(self.success_hist[-20:]), 1)
        loss = max(0.0, 1.0 - sr)
        return float(loss), max(self.task_counter, 1), {
            "accuracy":       float(sr),
            "loss":           float(loss),
            "eval_time":      0.1,
            "success_rate":   float(sr),
            "policy_entropy": float(self.agent.get_entropy()),
            "cpu_usage":      float(psutil.cpu_percent()),
            "gpu_usage":      float(torch.cuda.memory_allocated(DEVICE) /
                                    (1024 * 1024)) if torch.cuda.is_available() else 0.0,
        }


def cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id",      type=int, required=True)
    parser.add_argument("--num-clients",    type=int, default=8)
    parser.add_argument("--container-type", type=str,
                        choices=["gpu_specialist", "cpu_specialist"],
                        default="gpu_specialist")
    args = parser.parse_args()

    SERVER     = os.getenv("MASTER_ADDRESS", "127.0.0.1:8080")
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

    logger.info(f"[robot_{args.client_id:03d}] Starting — "
                f"server={SERVER} type={args.container_type}")

    r        = redis.Redis(host=REDIS_HOST, decode_responses=True)
    agent    = RobotPPOAgent(robot_id=f"robot_{args.client_id:03d}")
    sensor   = RobotSensor(robot_id=f"robot_{args.client_id:03d}")
    task_gen = SyntheticTaskGenerator(container_type=args.container_type,
                                      seed=args.client_id * 100)
    client   = RobotClient(agent, sensor, task_gen, args.client_id, r)

    retry = 0
    while retry < MAX_RETRIES and not shutdown_requested:
        try:
            fl.client.start_client(server_address=SERVER, client=client.to_client())
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
