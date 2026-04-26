"""
Flower server for DHT+FRL condition.
Runs FedAvg only (no FedProx, no FedAdam).
Aggregates BidPolicyMLP weights from 8 robot containers.
Based on server_asr_optimized.py structure — same pattern, different model.
"""
import os
import sys
import time
import signal
import logging
import psutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"

import flwr as fl
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from flwr.common import Metrics, ndarrays_to_parameters
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robot"))
from policy import BidPolicyMLP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logging.getLogger("flwr").setLevel(logging.WARNING)

# --- CONFIG ---
N_CLIENTS       = 8
N_ROUNDS        = 50     # 50 × 20 = 1000 tasks per worker, matches baseline TOTAL_TASKS
SERVER_ADDRESS  = "0.0.0.0:8080"
RESULT_DIR      = os.path.join(os.path.dirname(__file__), "results")
STATE_DIM       = 15

shutdown_requested = False
round_start_time   = 0.0
round_start_net    = 0.0


def signal_handler(signum, frame):
    global shutdown_requested
    logger.info(f"Signal {signum} received — shutting down")
    shutdown_requested = True


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


_round_counter = 0


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global _round_counter
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}

    reward        = sum(n * m.get("mean_reward", 0)        for n, m in metrics) / total
    success_rate  = sum(n * m.get("success_rate", 0)       for n, m in metrics) / total
    train_loss    = sum(n * m.get("train_loss", 0)         for n, m in metrics) / total
    policy_entropy= sum(n * m.get("policy_entropy", 0)     for n, m in metrics) / total

    cpu_usage  = sum(m.get("cpu_usage", 0)  for _, m in metrics) / len(metrics)
    gpu_usage  = sum(m.get("gpu_usage", 0)  for _, m in metrics) / len(metrics)
    net_client = sum(m.get("network_mb", 0) for _, m in metrics) / len(metrics)
    train_time = max(m.get("train_time", 0) for _, m in metrics)

    total_latency = time.time() - round_start_time
    net_now       = psutil.net_io_counters()
    server_net    = ((net_now.bytes_sent + net_now.bytes_recv) - round_start_net) / (1024 * 1024)

    _round_counter += 1
    phase = "FIT " if _round_counter % 2 == 1 else "EVAL"
    fl_round = (_round_counter + 1) // 2
    logger.info(
        f"[ROUND {fl_round:>2}/{N_ROUNDS} | {phase}] "
        f"clients={len(metrics)}/{N_CLIENTS}  "
        f"success={success_rate:.3f}  reward={reward:+.3f}  "
        f"loss={train_loss:.4f}  entropy={policy_entropy:.3f}  "
        f"cpu={cpu_usage:.1f}%  gpu={gpu_usage:.1f}%  "
        f"net={max(net_client, server_net):.2f}MB  "
        f"latency={total_latency:.1f}s"
    )

    return {
        "mean_reward":    round(reward, 4),
        "success_rate":   round(success_rate, 4),
        "train_loss":     round(train_loss, 6),
        "policy_entropy": round(policy_entropy, 4),
        "cpu_usage":      round(cpu_usage, 2),
        "gpu_usage":      round(gpu_usage, 2),
        "network_mb":     round(max(net_client, server_net), 3),
        "train_time":     round(train_time, 2),
        "total_latency":  round(total_latency, 2),
    }


def save_results(history):
    """Save FL convergence metrics to CSV and generate graphs."""
    os.makedirs(RESULT_DIR, exist_ok=True)

    fit_data  = history.metrics_distributed_fit
    eval_data = history.metrics_distributed

    rounds = sorted({r for k in {**fit_data, **eval_data} for r, _ in {**fit_data, **eval_data}[k]})
    rows = []
    for r in rounds:
        row = {"round": r}
        for k in ["train_loss", "mean_reward", "success_rate", "policy_entropy",
                  "cpu_usage", "gpu_usage", "network_mb", "train_time", "total_latency"]:
            val = next((v for rn, v in {**fit_data, **eval_data}.get(k, []) if rn == r), 0)
            row[k] = val
        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(f"{RESULT_DIR}/fl_convergence.csv", index=False)
    df[["round", "total_latency", "train_time"]].to_csv(
        f"{RESULT_DIR}/fl_latency.csv", index=False
    )
    df[["round", "cpu_usage", "gpu_usage"]].to_csv(
        f"{RESULT_DIR}/fl_hardware.csv", index=False
    )
    df[["round", "network_mb"]].to_csv(
        f"{RESULT_DIR}/fl_network.csv", index=False
    )
    logger.info(f"CSVs saved to {RESULT_DIR}/")

    _plot(df, "round", ["success_rate", "mean_reward"],
          "Policy performance vs FL round",
          f"{RESULT_DIR}/graph_policy_performance.png")
    _plot(df, "round", ["train_loss"],
          "Training loss vs FL round",
          f"{RESULT_DIR}/graph_train_loss.png")
    _plot(df, "round", ["cpu_usage", "gpu_usage"],
          "Hardware utilization vs FL round",
          f"{RESULT_DIR}/graph_hardware.png")
    _plot(df, "round", ["network_mb"],
          "Network traffic vs FL round",
          f"{RESULT_DIR}/graph_network.png")
    logger.info(f"Graphs saved to {RESULT_DIR}/")


def _plot(df, x_col, y_cols, title, save_path):
    if df.empty:
        return
    plt.figure(figsize=(10, 5))
    for y in y_cols:
        if y in df.columns:
            plt.plot(df[x_col], df[y], marker="o", markersize=3, label=y, linewidth=1.5)
    plt.title(title)
    plt.xlabel(x_col)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_fedavg():
    global round_start_time, round_start_net
    model  = BidPolicyMLP(state_dim=STATE_DIM)
    params = ndarrays_to_parameters([v.cpu().numpy() for v in model.state_dict().values()])

    def config_fn(server_round: int) -> Dict:
        global round_start_time, round_start_net
        round_start_time = time.time()
        net = psutil.net_io_counters()
        round_start_net = net.bytes_sent + net.bytes_recv
        logger.info(f">>> Starting FL round {server_round}/{N_ROUNDS} — dispatching to {N_CLIENTS} clients...")
        return {"local_epochs": 1, "round": server_round}

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=N_CLIENTS,
        min_evaluate_clients=N_CLIENTS,
        min_available_clients=N_CLIENTS,
        initial_parameters=params,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=config_fn,
    )

    logger.info(f"Waiting for {N_CLIENTS} robot clients to connect...")
    history = fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy=strategy,
    )
    return history


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    logger.info(f"Flower server starting — FedAvg only — {N_CLIENTS} clients — {N_ROUNDS} rounds")
    logger.info("Waiting 15s for containers to start...")
    time.sleep(15)

    try:
        history = run_fedavg()
        save_results(history)
        logger.info("FedAvg complete. Results saved.")
    except KeyboardInterrupt:
        logger.info("Server interrupted")
    finally:
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    main()
