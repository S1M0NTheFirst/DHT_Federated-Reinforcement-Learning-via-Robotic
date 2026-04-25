"""
Shared metrics collector — used by all three experiment conditions.
Writes migration event metrics to CSV files in the condition's results/ folder.
"""
import csv
import os
import time
import json
import threading
import psutil
import redis

try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False
    _GPU_HANDLE = None


def get_gpu_util() -> float:
    if not GPU_AVAILABLE:
        return 0.0
    try:
        return pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE).gpu / 100.0
    except Exception:
        return 0.0


def get_cpu_util() -> float:
    return psutil.cpu_percent(interval=0.1) / 100.0


def get_net_bytes() -> int:
    c = psutil.net_io_counters()
    return c.bytes_sent + c.bytes_recv


class MigrationMetricsWriter:
    """
    Writes one CSV row per migration event.
    Each row captures timing, resource usage, and RL performance metrics.
    """

    FIELDNAMES = [
        "condition",          # dht_frl | criu_cold | criu_warm
        "robot_id",
        "migration_event_id", # sequential counter per condition
        "timestamp",
        # --- Timing breakdown ---
        "trigger_to_dump_ms",      # migration trigger → CRIU checkpoint done
        "dump_to_transfer_ms",     # CRIU done → transfer complete at destination
        "transfer_to_restore_ms",  # transfer done → container running at destination
        "policy_load_ms",          # container running → policy loaded in memory (0 for CRIU baselines)
        "downtime_ms",             # trigger → first bid at destination (what robot loses)
        "total_MTT_ms",            # trigger → fully operational
        # --- RL performance ---
        "success_rate_pre",        # rolling success rate 10 tasks before migration
        "success_rate_post",       # rolling success rate 10 tasks after migration
        "regression_pct",          # (pre-post)/pre*100
        "fl_rounds_to_recover",    # FL rounds until within 5% of pre-migration rate
        "replay_buffer_entries_restored",  # 0 for CRIU baselines
        # --- Resource usage during migration window ---
        "gpu_util_pre_migration",
        "gpu_util_during_migration",
        "gpu_util_post_migration",
        "cpu_util_pre_migration",
        "cpu_util_during_migration",
        "cpu_util_post_migration",
        "network_bytes_transferred",
        # --- CRIU-specific ---
        "checkpoint_size_mb",
        "criu_mode",               # cold | precopy | unified
    ]

    def __init__(self, condition: str, results_dir: str):
        self.condition = condition
        self.results_dir = results_dir
        self.csv_path = os.path.join(results_dir, "migration_events.csv")
        self._lock = threading.Lock()
        self._event_counter = 0
        os.makedirs(results_dir, exist_ok=True)

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def write_event(self, metrics: dict):
        """Write one migration event row to CSV."""
        with self._lock:
            self._event_counter += 1
            row = {field: metrics.get(field, 0) for field in self.FIELDNAMES}
            row["condition"] = self.condition
            row["migration_event_id"] = self._event_counter
            row["timestamp"] = time.time()
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writerow(row)


class TaskMetricsWriter:
    """Writes one CSV row per task execution — tracks success rate over time."""

    FIELDNAMES = [
        "condition", "robot_id", "task_counter", "fl_round",
        "task_type", "complexity", "duration_s",
        "bid_value", "reward", "status",
        "exec_latency_ms", "deadline_ms",
        "success_rate_rolling10",
        "gpu_util", "cpu_util",
        "policy_entropy",
        "training_step",
        "timestamp",
    ]

    def __init__(self, condition: str, results_dir: str):
        self.condition = condition
        self.csv_path = os.path.join(results_dir, "task_logs.csv")
        self._lock = threading.Lock()
        os.makedirs(results_dir, exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def write_task(self, metrics: dict):
        with self._lock:
            row = {field: metrics.get(field, 0) for field in self.FIELDNAMES}
            row["condition"] = self.condition
            row["timestamp"] = time.time()
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writerow(row)
