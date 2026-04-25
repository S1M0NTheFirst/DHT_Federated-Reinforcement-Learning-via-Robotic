# server_asr_optimized.py
import os
import sys
import signal

# --- SILENCE LOGS ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["PYTHONUNBUFFERED"] = "1"

import flwr as fl
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from flwr.common import Metrics, ndarrays_to_parameters
import logging
import time
import traceback
import gc
import psutil # <--- ADDED

# --- CONFIGURATION ---
N_CLIENTS = 8
N_ROUNDS = 2         # <--- CHANGED to 2
EPOCHS_PER_ROUND = 1 # <--- CHANGED to 1
N_CLASSES = 29
# Allow the server to start even if not all clients are perfectly ready immediately
MIN_AVAILABLE_CLIENTS = N_CLIENTS 
SERVER_ADDRESS = "0.0.0.0:8080"
RESULT_DIR = "result_dht" # <--- NEW FOLDER

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logging.getLogger('flwr').setLevel(logging.WARNING)

# Shutdown flag
shutdown_requested = False

# --- GLOBAL TIMING & NETWORK ---
round_start_time = 0.0
round_start_net = 0.0

def signal_handler(signum, frame):
    # --- FIX IS HERE: Added 'global' keyword ---
    global shutdown_requested
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

class SimpleASR(nn.Module):
    def __init__(self, in_feat=80, rnn_h=256, n_class=N_CLASSES, down_factor=4, dropout=0.1):
        super(SimpleASR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True), 
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 32, 3, 2, 1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True), 
            nn.Dropout2d(dropout)
        )
        self.rnn_input_size = 32 * (in_feat // down_factor)
        self.rnn = nn.GRU(
            self.rnn_input_size, rnn_h, 2, 
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.fc = nn.Linear(rnn_h * 2, n_class)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1, x.size(3)).transpose(1, 2)
        x, _ = self.rnn(x)
        return self.fc(x)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics: return {}
    total_examples = sum([num_examples for num_examples, _ in metrics])
    if total_examples == 0: return {}
    
    # 1. Base Metrics (Weighted Average)
    acc = sum([num * m.get("accuracy", 0) for num, m in metrics]) / total_examples
    loss = sum([num * m.get("loss", 0) for num, m in metrics]) / total_examples
    train_loss = sum([num * m.get("train_loss", 0) for num, m in metrics]) / total_examples

    # 2. Hardware / Network (Simple Average across clients)
    cpu = sum([m.get("cpu_usage", 0) for _, m in metrics]) / len(metrics)
    gpu = sum([m.get("gpu_usage", 0) for _, m in metrics]) / len(metrics)
    client_net_mb = sum([m.get("network_mb", 0) for _, m in metrics]) / len(metrics)
    
    # 3. Latency 
    train_time = max([m.get("train_time", 0) for _, m in metrics])
    eval_time = max([m.get("eval_time", 0) for _, m in metrics])
    total_latency = time.time() - round_start_time

    # 4. Server-Side Network Delta (Real System Traffic)
    net_now = psutil.net_io_counters()
    server_net_delta = ((net_now.bytes_sent + net_now.bytes_recv) - round_start_net) / (1024 * 1024)
    # Use max of client reported or server observed to get a sense of activity
    final_net_mb = max(client_net_mb, server_net_delta)

    return {
        "accuracy": acc, "loss": loss, "train_loss": train_loss,
        "cpu_usage": cpu, "gpu_usage": gpu,
        "train_time": train_time, "eval_time": eval_time, 
        "total_latency": total_latency,
        "network_mb": final_net_mb
    }

def save_results(method: str, history):
    try:
        os.makedirs(RESULT_DIR, exist_ok=True)
        
        combined = {}
        fit_data = history.metrics_distributed_fit
        eval_data = history.metrics_distributed
        
        # Get all rounds present
        round_set = set()
        for k in fit_data.keys():
            for r, v in fit_data[k]: round_set.add(r)
        for k in eval_data.keys():
            for r, v in eval_data[k]: round_set.add(r)
        
        rounds = sorted(list(round_set))

        for r in rounds:
            combined[r] = {"Round": r, "Method": method}
            
            # Extract from Fit (Train)
            for k in ["train_time", "cpu_usage", "gpu_usage", "network_mb", "train_loss"]:
                val = next((v for rnd, v in fit_data.get(k, []) if rnd == r), 0)
                combined[r][k] = val

            # Extract from Eval
            for k in ["accuracy", "loss", "eval_time", "total_latency"]:
                val = next((v for rnd, v in eval_data.get(k, []) if rnd == r), 0)
                if k == "loss": combined[r]["global_loss"] = val
                else: combined[r][k] = val

        df_new = pd.DataFrame(list(combined.values()))
        
        # --- SAVE CSVs EXACTLY LIKE LOCAL ---
        # 1. Main Results
        _save_csv(df_new, f"{RESULT_DIR}/all_dht_results.csv", ["Method", "Round", "accuracy", "global_loss", "train_loss"])
        # 2. Latency
        _save_csv(df_new, f"{RESULT_DIR}/dht_latency.csv", ["Method", "Round", "total_latency", "train_time", "eval_time"])
        # 3. Hardware
        _save_csv(df_new, f"{RESULT_DIR}/dht_hardware.csv", ["Method", "Round", "cpu_usage", "gpu_usage"])
        # 4. Network
        _save_csv(df_new, f"{RESULT_DIR}/dht_network.csv", ["Method", "Round", "network_mb"])

        # --- GENERATE THE 4 GRAPHS ---
        _plot_dual_axis(f"{RESULT_DIR}/all_dht_results.csv", "accuracy", "global_loss", "Global Accuracy vs Global Loss", f"{RESULT_DIR}/graph_accuracy_loss.png")
        _plot_multi_line(f"{RESULT_DIR}/dht_latency.csv", ["total_latency", "train_time", "eval_time"], "Latency Breakdown (s)", f"{RESULT_DIR}/graph_latency.png")
        _plot_multi_line(f"{RESULT_DIR}/dht_hardware.csv", ["cpu_usage", "gpu_usage"], "Hardware Usage (%)", f"{RESULT_DIR}/graph_hardware.png")
        _plot_multi_line(f"{RESULT_DIR}/dht_network.csv", ["network_mb"], "Network Traffic (MB)", f"{RESULT_DIR}/graph_network.png")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        import traceback
        traceback.print_exc()

def _save_csv(new_df, filename, cols):
    for c in cols:
        if c not in new_df.columns: new_df[c] = 0   
    if os.path.exists(filename):
        old_df = pd.read_csv(filename)
        merged = pd.concat([old_df, new_df[cols]], ignore_index=True)
        # Drop duplicates to prevent accumulating same run data if restarted
        merged = merged.drop_duplicates(subset=['Method', 'Round'], keep='last')
        merged.to_csv(filename, index=False)
    else:
        new_df[cols].to_csv(filename, index=False)
    logger.info(f"Saved {filename}")

def _plot_dual_axis(csv, y1, y2, title, save_path):
    if not os.path.exists(csv): return
    df = pd.read_csv(csv)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    for method in df['Method'].unique():
        sub = df[df['Method'] == method]
        if not sub.empty:
            ax1.plot(sub['Round'], sub[y1], marker='o', label=f"{method} Acc")
    ax2 = ax1.twinx()
    for method in df['Method'].unique():
        sub = df[df['Method'] == method]
        if not sub.empty:
            ax2.plot(sub['Round'], sub[y2], marker='x', linestyle='--', label=f"{method} Loss")
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy', color='b')
    ax2.set_ylabel('Global Loss', color='r')
    plt.title(title)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()

def _plot_multi_line(csv, y_cols, title, save_path):
    if not os.path.exists(csv): return
    df = pd.read_csv(csv)
    plt.figure(figsize=(10, 6))
    for method in df['Method'].unique():
        sub = df[df['Method'] == method]
        for y in y_cols:
            if y in sub.columns and not sub[y].empty:
                plt.plot(sub['Round'], sub[y], marker='.', label=f"{method} {y}")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def run_method(method: str):
    logger.info(f"\n{'='*60}\nSTARTING METHOD: {method}\n{'='*60}")
    
    model = SimpleASR()
    params = ndarrays_to_parameters([val.cpu().numpy() for val in model.state_dict().values()])
    
    def config_fn(server_round: int) -> Dict:
        # Start Global Timer/Network Counter for the round
        global round_start_time, round_start_net
        round_start_time = time.time()
        net = psutil.net_io_counters()
        round_start_net = net.bytes_sent + net.bytes_recv
        
        return {
            "local_epochs": EPOCHS_PER_ROUND,
            "proximal_mu": 0.1 if method == "FedProx" else 0.0, # Increased Mu slightly to ensure effect
            "round": server_round
        }
    
    strategy_args = {
        "fraction_fit": 1.0, "fraction_evaluate": 1.0,
        "min_fit_clients": N_CLIENTS, "min_evaluate_clients": N_CLIENTS,
        "min_available_clients": MIN_AVAILABLE_CLIENTS,
        "initial_parameters": params,
        "evaluate_metrics_aggregation_fn": weighted_average,
        "fit_metrics_aggregation_fn": weighted_average,
        "on_fit_config_fn": config_fn,
    }
    
    if method == "FedProx":
        strategy = fl.server.strategy.FedProx(proximal_mu=0.1, **strategy_args)
    elif method == "FedAdam":
        strategy = fl.server.strategy.FedAdam(eta=1e-1, eta_l=1e-1, beta_1=0.9, beta_2=0.99, tau=1e-9, **strategy_args)
    else:
        strategy = fl.server.strategy.FedAvg(**strategy_args)
    
    logger.info(f"Waiting for {MIN_AVAILABLE_CLIENTS} clients to connect...")
    
    history = fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy=strategy
    )
    return history

def main():
    try:
        os.makedirs(RESULT_DIR, exist_ok=True)
        logger.info(f"Server starting on {SERVER_ADDRESS}")
        
        # Initial wait for clients
        logger.info("Waiting 10s for clients to initialize...")
        time.sleep(10)
        
        methods = ["FedAvg", "FedProx", "FedAdam"]
        
        for i, method in enumerate(methods):
            if shutdown_requested: break
            
            if i > 0:
                logger.info("Waiting 20s for clients to reset...")
                time.sleep(20)
                
            try:
                history = run_method(method)
                save_results(method, history)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error running {method}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Fatal server error: {e}")
    finally:
        logger.info("Server shutting down")

if __name__ == "__main__":
    main()