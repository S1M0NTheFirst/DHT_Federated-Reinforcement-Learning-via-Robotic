"""
task2 Flower server — FedAvg over the ONLINE-SAC actor weights, shared by ALL 6
conditions. Differences from task1's condition_A server:

  - Seeds initial_parameters from a shared SAC Actor at a fixed seed so every
    condition + every robot starts from the SAME global policy (PLAN: shared
    start for fair comparison). No pretrained file needed — online learning
    starts from a common random actor.
  - Writes fl_convergence.csv + fl_{network,hardware,latency}.csv for EVERY
    condition (task1 only wrote these for condition A).
  - PERSISTS task_logs.csv from Redis at the end (the per-round eval-return time
    series that was MISSING in task1). Every condition, including cold_restart.

Env (set by run.sh / runner): N_CLIENTS, N_ROUNDS, FLOWER_BIND, FL_RESULT_DIR,
REDIS_HOST, REDIS_PORT, SHARED_SEED.
"""
import csv
import json
import logging
import os
import signal
import sys
import time
from typing import Dict, List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import flwr as fl
import numpy as np
import psutil
import torch
from flwr.common import Metrics, ndarrays_to_parameters

# task2/worker on path for the Actor shape used to seed the global model.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "worker"))
from sac import Actor  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("task2_server")
logging.getLogger("flwr").setLevel(logging.WARNING)

N_CLIENTS      = int(os.environ.get("N_CLIENTS", "20"))
N_ROUNDS       = int(os.environ.get("N_ROUNDS", "150"))
SERVER_ADDRESS = os.environ.get("FLOWER_BIND", "0.0.0.0:8470")
RESULT_DIR     = os.environ.get("FL_RESULT_DIR",
                                os.path.join(os.path.dirname(__file__), "results"))
SHARED_SEED    = int(os.environ.get("SHARED_SEED", "12345"))
REDIS_HOST     = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT     = int(os.environ.get("REDIS_PORT", "6379"))

round_start_time = 0.0
round_start_net = 0.0
_round_counter = 0


def _sig(signum, frame):
    logger.info(f"Signal {signum} received")


signal.signal(signal.SIGTERM, _sig)
signal.signal(signal.SIGINT, _sig)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global _round_counter
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    reward = sum(n * m.get("mean_reward", 0) for n, m in metrics) / total
    success = sum(n * m.get("success_rate", 0) for n, m in metrics) / total
    loss = sum(n * m.get("train_loss", 0) for n, m in metrics) / total
    entropy = sum(n * m.get("policy_entropy", 0) for n, m in metrics) / total
    cpu = sum(m.get("cpu_usage", 0) for _, m in metrics) / len(metrics)
    gpu = sum(m.get("gpu_usage", 0) for _, m in metrics) / len(metrics)
    net = sum(m.get("network_mb", 0) for _, m in metrics) / len(metrics)
    ttime = max(m.get("train_time", 0) for _, m in metrics)

    total_latency = time.time() - round_start_time
    net_now = psutil.net_io_counters()
    server_net = ((net_now.bytes_sent + net_now.bytes_recv)
                  - round_start_net) / (1024 * 1024)

    _round_counter += 1
    phase = "FIT " if _round_counter % 2 == 1 else "EVAL"
    fl_round = (_round_counter + 1) // 2
    logger.info(f"[ROUND {fl_round:>3}/{N_ROUNDS} | {phase}] "
                f"clients={len(metrics)}/{N_CLIENTS} "
                f"eval_return~{reward:+.1f} success={success:.3f} "
                f"loss={loss:.3f} cpu={cpu:.0f}% "
                f"net={max(net, server_net):.2f}MB lat={total_latency:.1f}s")
    return {
        "mean_reward": round(reward, 4),
        "success_rate": round(success, 4),
        "train_loss": round(loss, 6),
        "policy_entropy": round(entropy, 4),
        "cpu_usage": round(cpu, 2),
        "gpu_usage": round(gpu, 2),
        "network_mb": round(max(net, server_net), 3),
        "train_time": round(ttime, 2),
        "total_latency": round(total_latency, 2),
    }


def _initial_actor_arrays():
    """Shared global actor at a fixed seed — same start for all conditions."""
    torch.manual_seed(SHARED_SEED)
    np.random.seed(SHARED_SEED)
    model = Actor()
    logger.info(f"Seeded global actor from fixed seed {SHARED_SEED}")
    return [v.cpu().numpy() for v in model.state_dict().values()]


def save_results(history):
    os.makedirs(RESULT_DIR, exist_ok=True)
    merged = {**history.metrics_distributed, **history.metrics_distributed_fit}
    rounds = sorted({r for k in merged for r, _ in merged[k]})
    cols = ["train_loss", "mean_reward", "success_rate", "policy_entropy",
            "cpu_usage", "gpu_usage", "network_mb", "train_time", "total_latency"]
    rows = []
    for rd in rounds:
        row = {"round": rd}
        for k in cols:
            row[k] = next((v for rn, v in merged.get(k, []) if rn == rd), 0)
        rows.append(row)

    def _write(path, fields):
        with open(os.path.join(RESULT_DIR, path), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, 0) for k in fields})

    _write("fl_convergence.csv", ["round"] + cols)
    _write("fl_network.csv", ["round", "network_mb"])
    _write("fl_hardware.csv", ["round", "cpu_usage", "gpu_usage"])
    _write("fl_latency.csv", ["round", "total_latency", "train_time"])
    logger.info(f"fl_*.csv written to {RESULT_DIR}/")


def persist_task_logs():
    """Drain Redis `task_logs` (per-round eval rows pushed by every worker) to
    task_logs.csv — the time-series file that was MISSING in task1."""
    try:
        import redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        raw = r.lrange("task_logs", 0, -1)
    except Exception as e:
        logger.warning(f"could not read task_logs from redis: {e!r}")
        return
    rows = []
    for item in raw:
        try:
            rows.append(json.loads(item))
        except Exception:
            continue
    if not rows:
        logger.warning("task_logs empty in redis — nothing to persist")
        return
    fields = ["robot_id", "fl_round", "training_step", "reward",
              "success_rate_rolling10", "policy_entropy", "status",
              "eval_return", "eval_episode_len", "eval_success"]
    path = os.path.join(RESULT_DIR, "task_logs.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in sorted(rows, key=lambda e: (e.get("robot_id", ""),
                                               e.get("fl_round", 0))):
            w.writerow({k: row.get(k, "") for k in fields})
    logger.info(f"task_logs.csv persisted ({len(rows)} rows) to {path}")


def run():
    global round_start_time, round_start_net
    params = ndarrays_to_parameters(_initial_actor_arrays())

    def config_fn(server_round: int) -> Dict:
        global round_start_time, round_start_net
        round_start_time = time.time()
        net = psutil.net_io_counters()
        round_start_net = net.bytes_sent + net.bytes_recv
        return {"round": server_round}

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=N_CLIENTS, min_evaluate_clients=N_CLIENTS,
        min_available_clients=N_CLIENTS,
        initial_parameters=params,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=config_fn,
        on_evaluate_config_fn=config_fn,
    )
    logger.info(f"Waiting for {N_CLIENTS} online-SAC clients...")
    return fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy=strategy,
    )


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    logger.info(f"task2 Flower server — {N_CLIENTS} clients × {N_ROUNDS} rounds "
                f"→ {RESULT_DIR}")
    time.sleep(15)
    try:
        history = run()
        save_results(history)
    finally:
        persist_task_logs()
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    main()
