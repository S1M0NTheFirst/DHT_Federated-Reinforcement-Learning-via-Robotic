# SwiftBot-RL v2 — Full Agent Implementation Guide
## DHT + FRL Migration vs CRIU Migration Experiment

---

## FULL PROJECT PROMPT — READ THIS COMPLETELY BEFORE TOUCHING ANY FILE

You are implementing a systems research experiment called **SwiftBot-RL** for an academic paper targeting ATC 2026. The goal is to prove that a **DHT-coordinated Federated Reinforcement Learning (FRL) migration system** outperforms **CRIU-based container migration** (both cold and warm/pre-copy variants) when robot agents need to migrate between overloaded nodes.

### What This Project Is

Eight Docker containers act as robot agents. Each robot processes synthetic computational tasks (heavy GPU matrix operations simulating robot perception, heavy CPU eigendecomposition simulating robot planning). Each robot has a local PPO policy that learns to bid on tasks — deciding whether to accept or decline an incoming task based on its current system load. Policy weights are aggregated across all 8 robots using FedAvg via a Flower server. When a robot's node becomes overloaded (CPU or GPU > 85%), a migration event fires.

**The central claim of the paper:** When migration happens, the DHT+FRL system transfers the container checkpoint + policy weights + replay buffer together as one unified atomic unit, so the robot arrives at its destination with its full intelligence intact and can bid competently immediately. The CRIU baselines (cold and warm/pre-copy) only transfer container runtime state — the robot wakes up with no learned policy and no experience, and must relearn from scratch. This causes measurable performance regression that our DHT+FRL system avoids.

### What the Three Conditions Are

**Condition A — DHT+FRL (YOUR SYSTEM):**
- 8 containers, each running a PPO policy trained with FedAvg (Flower server)
- Migration triggered by DHT orchestrator on the HOST (Option B — not from inside the container)
- When migration fires: container saves policy_weights.pt + replay_buffer.pkl to /checkpoints/, then CRIU checkpoints the container from the host, then all three components transfer in parallel to destination
- Robot at destination: container restores, policy already waiting in /checkpoints/, replay buffer restores — robot bids on its first task immediately
- **Has FRL: YES**

**Condition B — CRIU Cold (BASELINE):**
- 8 containers, each running a RANDOM policy (no PPO, no FedAvg, no Flower)
- Migration triggered by CRIU runner script on the HOST
- When migration fires: container fully stops, CRIU dumps everything sequentially, transfer, restore
- Robot at destination: wakes with empty state, random bidding
- **Has FRL: NO**

**Condition C — CRIU Pre-copy / Warm (BASELINE):**
- 8 containers, each running a RANDOM policy (no PPO, no FedAvg, no Flower)
- Migration triggered by CRIU runner script on the HOST  
- When migration fires: CRIU iteratively pre-dumps dirty pages while container keeps running, brief final delta pause only
- Robot at destination: wakes with empty state despite shorter pause time, random bidding
- **Has FRL: NO**

### Why FRL Is Only in Condition A

CRIU is the state of the art for container migration. No published paper combines CRIU with FRL policy migration. Adding FedAvg to CRIU baselines would give them an unfair advantage that does not exist in real systems and would obscure the migration contribution. The comparison is: our full system (DHT + FRL + unified migration) vs the best available migration tool (CRIU) used as it actually exists in practice.

### Architecture Source

This project reuses the existing DHT + Flower FL architecture from:
- `dht_asr_optimized.py` — Kademlia DHT overlay, 4 nodes × 2 containers = 8 total. KEEP the entire DHT/Kademlia structure, node bootstrap, Docker container launch loop, IP detection logic, asyncio pattern.
- `worker_client_asr_optimized.py` — Flower client (fl.client.NumPyClient). KEEP: get_parameters(), set_parameters(), fit(), evaluate() signatures, retry/reconnect logic, psutil hardware tracking, signal handling. REPLACE: SimpleASR model → BidPolicyMLP, LibriSpeech dataset → SyntheticTaskGenerator, CTC training → PPO update, ASR metrics → migration metrics.
- `server_asr_optimized.py` — Flower server. KEEP: entire server structure, weighted_average(), strategy selection, run_method() loop, CSV/graph saving pattern. CHANGE: model class, metrics collected, methods list to ["FedAvg"] only.
- `Dockerfile.optimized` — KEEP: base CUDA image, flwr, psutil, torch. REMOVE: torchaudio, jiwer, libsndfile, ffmpeg, sox. ADD: criu, stable-baselines3, gymnasium.

### What to Measure and Track

At EVERY migration event, log these metrics to CSV:
- `downtime_ms` — time from migration trigger to robot's first bid at destination
- `total_MTT_ms` — total migration time (trigger → fully operational)
- `container_dump_ms` — CRIU checkpoint creation time
- `transfer_ms` — time to copy checkpoint to destination
- `policy_load_ms` — time to load policy_weights.pt into memory (YOUR NEW METRIC)
- `success_rate_pre` — task success rate in 10 tasks before migration
- `success_rate_post` — task success rate in 10 tasks after migration
- `regression_pct` — (pre - post) / pre × 100
- `fl_rounds_to_recover` — FL rounds until success rate returns within 5% of pre-migration
- `gpu_util_during_migration` — GPU utilization during the migration window
- `cpu_util_during_migration` — CPU utilization during the migration window
- `network_bytes_transferred` — total bytes moved during migration
- `replay_buffer_entries_restored` — how many experience tuples arrived at destination

### Hardware

AMD Ryzen 9 7900X (12-core, 4.70 GHz) · NVIDIA GeForce RTX 4080 (16 GB VRAM) · 32 GB RAM · Ubuntu 22.04 LTS (bare metal — required for CRIU)

### Important CRIU Note

The good news is that as of 2024, NVIDIA and the CRIU developers finally solved this. To make Federated Learning migration work, we can try two specific pieces of software acting in tandem:

1. NVIDIA's cuda-checkpoint utility
NVIDIA recently released a standalone tool specifically designed to freeze CUDA execution.

    How it works: When triggered, cuda-checkpoint locks all active CUDA APIs, waits for current matrix operations to finish, and then takes everything currently sitting in your RTX 4080's VRAM and copies it back over the PCIe bus into your system's regular host RAM. Finally, it tells the GPU to release the resources.

    Requirements: It requires NVIDIA Display Driver version 550 or higher (which you should already have installed on your fresh Ubuntu setup).

2. The criu-cuda-plugin
CRIU version 4.0+ introduced official support for a dynamic library plugin that talks directly to NVIDIA's tool.

    How it works: When you execute a CRIU dump command, the criu-cuda-plugin acts as the middleman. It tells cuda-checkpoint to move the GPU memory into the host RAM first. Once the VRAM is safely sitting in the standard Linux memory space, CRIU does its normal job of writing that host memory to a checkpoint file on your disk.

    The Restore: When you migrate the container to a new node, CRIU restores the host memory, and the plugin tells the NVIDIA driver to re-acquire the GPU and push the data back into the VRAM.

---

## PROJECT FOLDER STRUCTURE

```
~/swiftbot_rl/
├── dht_frl/                          ← YOUR SYSTEM (Condition A)
│   ├── Dockerfile                    ← robot container image with PPO + FedAvg
│   ├── dht_frl_runner.py             ← orchestrator: DHT + migration trigger (Option B)
│   ├── worker_robot_client.py        ← Flower client with PPO policy
│   ├── flower_server.py              ← Flower server running FedAvg only
│   ├── robot/
│   │   ├── sensor.py                 ← 15-dim state vector reader
│   │   ├── policy.py                 ← BidPolicyMLP + ReplayBuffer + PPO update
│   │   ├── task_generator.py         ← SyntheticTaskGenerator (GPU/CPU tasks)
│   │   └── checkpoint_manager.py    ← UnifiedCheckpointManager (pack/transfer/restore)
│   └── results/                      ← CSVs and graphs saved here
│
├── criu_cold/                        ← CRIU COLD BASELINE (Condition B)
│   ├── Dockerfile                    ← robot container image with random policy only
│   ├── criu_cold_runner.py           ← orchestrator: launches containers + CRIU cold migration
│   ├── worker_random_client.py       ← simple random bidding worker, no Flower, no PPO
│   └── results/                      ← CSVs and graphs saved here
│
├── criu_warm/                        ← CRIU PRE-COPY BASELINE (Condition C)
│   ├── Dockerfile                    ← same as criu_cold (shared image possible)
│   ├── criu_warm_runner.py           ← orchestrator: launches containers + CRIU pre-copy
│   ├── worker_random_client.py       ← same as criu_cold worker
│   └── results/                      ← CSVs and graphs saved here
│
├── evaluation/
│   ├── compare_all.py                ← reads all 3 results dirs, produces comparison figures
│   └── figures/                      ← final paper figures saved here
│
└── shared/
    └── metrics_collector.py          ← shared metrics logging utilities
```

---

## PHASE 0 — Ubuntu System Setup

**Do this once on your Ubuntu bare-metal machine. Do not run experiments on Windows.**

### 0.1 Verify Ubuntu bare-metal

```bash
uname -a
# Must show Linux kernel, NOT microsoft or WSL
# Example: Linux hostname 5.15.0-91-generic #101-Ubuntu SMP x86_64 GNU/Linux
cat /etc/os-release | grep VERSION
# Should show Ubuntu 22.04
```

### 0.2 Install system dependencies

```bash
sudo apt-get update && sudo apt-get upgrade -y

# Install CRIU — this is critical, must be on the HOST
sudo apt-get install -y criu
criu --version
# Must show: Criuimage (CRIU) of 3.x or higher

# Install Python 3.10+
sudo apt-get install -y python3.10 python3.10-venv python3-pip git wget curl

# Install Redis
sudo apt-get install -y redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server
redis-cli ping
# Must return: PONG

# Install Python packages on host (for runner scripts)
pip3 install docker kademlia asyncio psutil pynvml pandas matplotlib \
             seaborn redis numpy torch flwr
```

### 0.3 Install Docker Engine (NOT Docker Desktop)

```bash
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) \
  signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Enable experimental features — REQUIRED for docker checkpoint
sudo mkdir -p /etc/docker
echo '{"experimental": true}' | sudo tee /etc/docker/daemon.json
sudo systemctl restart docker
sudo usermod -aG docker $USER
# Log out and log back in after this

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 0.4 Verify CRIU works with Docker

```bash
# This test MUST pass before any experiment runs
docker run -d --name criu_test \
  --security-opt seccomp:unconfined \
  alpine sleep 3600

docker checkpoint create \
  --checkpoint-dir=/tmp/criu_verify \
  criu_test chk1

ls /tmp/criu_verify/chk1/
# Must show: core-*.img, dump.log, files.img, fs-*.img, pagemap-*.img, pages-*.img

docker rm -f criu_test
echo "CRIU VERIFICATION PASSED"
```

**If this fails:** CRIU cannot work on this machine. Common causes: WSL2 instead of bare-metal, kernel version too old (needs 5.4+), seccomp not properly disabled. Fix before proceeding.

### 0.5 Create project directory structure

```bash
mkdir -p ~/swiftbot_rl/{dht_frl,criu_cold,criu_warm,evaluation,shared}
mkdir -p ~/swiftbot_rl/dht_frl/{robot,results}
mkdir -p ~/swiftbot_rl/criu_cold/results
mkdir -p ~/swiftbot_rl/criu_warm/results
mkdir -p ~/swiftbot_rl/evaluation/figures
mkdir -p /tmp/swiftbot_checkpoints
chmod 777 /tmp/swiftbot_checkpoints
echo "Directory structure created"
```

**Verification:**
```bash
tree ~/swiftbot_rl --max-depth 3
# Must show all directories listed above
```

---

## PHASE 1 — Shared Components

### 1.1 Create shared metrics collector

**File: `~/swiftbot_rl/shared/metrics_collector.py`**

```python
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

        # Write header if file doesn't exist
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
```

---

## PHASE 2 — DHT + FRL System (Condition A)

### 2.1 Synthetic Task Generator

**File: `~/swiftbot_rl/dht_frl/robot/task_generator.py`**

```python
"""
Synthetic task generator — replaces UCF101/LibriSpeech.
Creates reproducible GPU-heavy and CPU-heavy tasks that simulate
robotic perception and planning workloads.
The datasets (UCF101, LibriSpeech) justify our task types in the paper
motivation section, but we do not actually process them here.
"""
import random
import hashlib
import time
import torch
import numpy as np


class SyntheticTaskGenerator:
    """
    GPU-heavy tasks simulate robot perception (image classification, object detection).
    CPU-heavy tasks simulate robot planning (path computation, sensor fusion).
    complexity (0.0-1.0) controls matrix size → controls actual GPU/CPU load.
    """

    def __init__(self, container_type: str, seed: int = 42):
        """
        container_type: "gpu_specialist" (containers 0-3)
                     or "cpu_specialist" (containers 4-7)
        """
        assert container_type in ("gpu_specialist", "cpu_specialist")
        self.container_type = container_type
        random.seed(seed)
        self._task_counter = 0

    def generate(self) -> dict:
        self._task_counter += 1
        task_id = hashlib.md5(
            f"{self.container_type}_{self._task_counter}_{time.time()}".encode()
        ).hexdigest()[:10]

        if self.container_type == "gpu_specialist":
            task_type = random.choices(
                ["gpu_heavy", "mixed"], weights=[0.85, 0.15]
            )[0]
            complexity = random.uniform(0.6, 1.0)
            duration_s = random.uniform(2.0, 5.0)
        else:
            task_type = random.choices(
                ["cpu_heavy", "mixed"], weights=[0.85, 0.15]
            )[0]
            complexity = random.uniform(0.5, 0.9)
            duration_s = random.uniform(1.0, 3.0)

        return {
            "task_id": task_id,
            "task_type": task_type,
            "complexity": round(complexity, 3),
            "duration_s": round(duration_s, 2),
            "deadline_ms": round(duration_s * 1000 * 1.5, 1),
        }

    def execute(self, task_spec: dict) -> dict:
        """
        Actually runs the computational workload.
        Returns result dict with status and actual latency.
        """
        import time
        t_start = time.perf_counter()
        task_type = task_spec["task_type"]
        complexity = task_spec["complexity"]
        duration_s = task_spec["duration_s"]

        try:
            if task_type == "gpu_heavy":
                self._run_gpu_task(complexity, duration_s)
            elif task_type == "cpu_heavy":
                self._run_cpu_task(complexity, duration_s)
            else:
                self._run_gpu_task(complexity, duration_s / 2)
                self._run_cpu_task(complexity, duration_s / 2)

            latency_ms = (time.perf_counter() - t_start) * 1000
            success = latency_ms <= task_spec["deadline_ms"]
            return {
                "status": "success" if success else "timeout",
                "latency_ms": round(latency_ms, 2),
                "deadline_ms": task_spec["deadline_ms"],
            }
        except Exception as e:
            return {"status": "failed", "error": str(e),
                    "latency_ms": 0, "deadline_ms": task_spec["deadline_ms"]}

    def _run_gpu_task(self, complexity: float, duration_s: float):
        """Large matrix multiply on GPU — simulates perception workload."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        size = int(1000 + complexity * 3000)
        A = torch.randn(size, size, device=device)
        B = torch.randn(size, size, device=device)
        t = time.perf_counter()
        while time.perf_counter() - t < duration_s:
            _ = torch.matmul(A, B)
            if device == "cuda":
                torch.cuda.synchronize()
        del A, B
        if device == "cuda":
            torch.cuda.empty_cache()

    def _run_cpu_task(self, complexity: float, duration_s: float):
        """Eigenvalue decomposition on CPU — simulates planning workload."""
        size = int(500 + complexity * 1500)
        M = np.random.randn(size, size).astype(np.float32)
        t = time.perf_counter()
        while time.perf_counter() - t < duration_s:
            np.linalg.eigvalsh(M)
```

### 2.2 PPO Policy Network and Replay Buffer

**File: `~/swiftbot_rl/dht_frl/robot/policy.py`**

```python
"""
BidPolicyMLP — PPO policy for robot task bidding.
State: 15-dim vector (see sensor.py)
Action: bid confidence [0.0, 1.0]
Trained with simplified PPO via replay buffer.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import pickle
import os


class BidPolicyMLP(nn.Module):
    """Small MLP. Fast inference < 1ms. Fits in < 1MB for policy transfer."""

    def __init__(self, state_dim: int = 15, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, 1),        nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_bid(self, state: np.ndarray) -> float:
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0)
            return float(self.net(t).squeeze())

    def get_entropy(self, state: np.ndarray) -> float:
        """Policy entropy — lower = more confident. Used as a training signal."""
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0)
            p = float(self.net(t).squeeze())
            p = max(1e-6, min(1.0 - 1e-6, p))
            return -(p * np.log(p) + (1 - p) * np.log(1 - p))


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            float(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
        ))

    def sample(self, batch_size: int = 64):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        s, a, r, ns = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(ns)

    def tail(self, n: int = 1000) -> list:
        return list(self.buffer)[-n:]

    def load_tail(self, entries: list):
        for e in entries:
            self.buffer.append(e)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(list(self.buffer), f)

    def load(self, path: str):
        with open(path, "rb") as f:
            entries = pickle.load(f)
        self.buffer = deque(entries, maxlen=self.buffer.maxlen)

    def __len__(self):
        return len(self.buffer)


class RobotPPOAgent:
    """Wraps BidPolicyMLP with PPO training loop."""

    def __init__(self, state_dim: int = 15, lr: float = 3e-4,
                 robot_id: str = "robot_000"):
        self.robot_id = robot_id
        self.policy = BidPolicyMLP(state_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.training_step = 0
        self._last_state = None
        self._last_action = None

    def get_bid(self, state: np.ndarray) -> float:
        bid = self.policy.get_bid(state)
        self._last_state = state.copy()
        self._last_action = bid
        return bid

    def record_reward(self, reward: float, next_state: np.ndarray):
        if self._last_state is not None:
            self.replay_buffer.add(
                self._last_state, self._last_action, reward, next_state
            )
        if len(self.replay_buffer) >= 64 and len(self.replay_buffer) % 32 == 0:
            self._ppo_update()

    def _ppo_update(self, batch_size: int = 64, epochs: int = 4):
        states, actions, rewards, _ = self.replay_buffer.sample(batch_size)
        s_t = torch.FloatTensor(states)
        r_t = torch.FloatTensor(rewards)
        if r_t.std() > 1e-8:
            r_t = (r_t - r_t.mean()) / (r_t.std() + 1e-8)
        for _ in range(epochs):
            bids = self.policy(s_t).squeeze()
            loss = -torch.mean(r_t * bids)
            entropy = -torch.mean(
                bids * torch.log(bids + 1e-8) +
                (1 - bids) * torch.log(1 - bids + 1e-8)
            )
            total_loss = loss - 0.01 * entropy
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        self.training_step += 1

    def get_entropy(self) -> float:
        if len(self.replay_buffer) < 10:
            return 1.0
        states, *_ = self.replay_buffer.sample(32)
        return float(np.mean([
            self.policy.get_entropy(s) for s in states
        ]))

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step": self.training_step,
            "robot_id": self.robot_id,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.training_step = ckpt.get("training_step", 0)

    def get_weights(self) -> dict:
        return {k: v.clone() for k, v in self.policy.state_dict().items()}

    def set_weights(self, weights: dict):
        self.policy.load_state_dict(weights)
```

### 2.3 Sensor (State Vector)

**File: `~/swiftbot_rl/dht_frl/robot/sensor.py`**

```python
"""
RobotSensor — reads system state and packages into 15-dim RL state vector.

Dim 0:  cpu_util (0-1)
Dim 1:  ram_util (0-1)
Dim 2:  gpu_util (0-1)
Dim 3:  gpu_mem_util (0-1)
Dim 4:  active_tasks_normalized (active/10)
Dim 5:  queue_depth_normalized (depth/20)
Dim 6:  task_type_gpu (0 or 1)
Dim 7:  task_type_cpu (0 or 1)
Dim 8:  task_complexity (0-1)
Dim 9:  task_deadline_normalized (deadline_ms/15000)
Dim 10: success_rate_gpu_rolling10 (0-1)
Dim 11: success_rate_cpu_rolling10 (0-1)
Dim 12: warm_container_ready (0 or 1)
Dim 13: policy_warm (0 or 1, 1 after first PPO update)
Dim 14: fl_staleness (rounds_since_sync/20, capped at 1)
"""
import numpy as np
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _GPU_OK = True
except Exception:
    _GPU_OK = False
    _GPU_HANDLE = None


class RobotSensor:
    def __init__(self, robot_id: str):
        self.robot_id = robot_id
        self._active_tasks = 0
        self._queue_depth = 0
        self._gpu_success_history = []
        self._cpu_success_history = []
        self._warm_ready = False
        self._policy_warm = False
        self._fl_staleness = 0

    def read(self, task_spec: dict = None) -> np.ndarray:
        cpu = psutil.cpu_percent(interval=0.05) / 100.0
        ram = psutil.virtual_memory().percent / 100.0
        if _GPU_OK:
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE).gpu / 100.0
            gm = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
            gpu_mem = gm.used / gm.total
        else:
            gpu_util = gpu_mem = 0.0

        t_gpu = t_cpu = complexity = deadline_norm = 0.5
        if task_spec:
            t_gpu = 1.0 if "gpu" in task_spec.get("task_type", "") else 0.0
            t_cpu = 1.0 if "cpu" in task_spec.get("task_type", "") else 0.0
            complexity = float(task_spec.get("complexity", 0.5))
            deadline_norm = min(float(task_spec.get("deadline_ms", 7500)) / 15000, 1.0)

        sr_gpu = (sum(self._gpu_success_history[-10:]) /
                  max(len(self._gpu_success_history[-10:]), 1))
        sr_cpu = (sum(self._cpu_success_history[-10:]) /
                  max(len(self._cpu_success_history[-10:]), 1))

        return np.array([
            cpu, ram, gpu_util, gpu_mem,
            min(self._active_tasks / 10.0, 1.0),
            min(self._queue_depth / 20.0, 1.0),
            t_gpu, t_cpu, complexity, deadline_norm,
            sr_gpu, sr_cpu,
            float(self._warm_ready),
            float(self._policy_warm),
            min(self._fl_staleness / 20.0, 1.0),
        ], dtype=np.float32)

    def record_outcome(self, task_type: str, success: bool):
        h = self._gpu_success_history if "gpu" in task_type else self._cpu_success_history
        h.append(1.0 if success else 0.0)
        if len(h) > 100:
            h.pop(0)

    def update(self, active: int, queue: int, warm: bool, policy: bool, staleness: int):
        self._active_tasks = active
        self._queue_depth = queue
        self._warm_ready = warm
        self._policy_warm = policy
        self._fl_staleness = staleness
```

### 2.4 Unified Checkpoint Manager

**File: `~/swiftbot_rl/dht_frl/robot/checkpoint_manager.py`**

```python
"""
UnifiedCheckpointManager — the core contribution.

Implements Option B: migration is triggered from the DHT orchestrator (host),
not from inside the container.

The "Unified Agent State" = CRIU checkpoint + policy_weights.pt + replay_buffer.pkl
All three transfer in parallel (pipelined) to minimize downtime.

The container signals the host to trigger migration by writing a flag to Redis.
The DHT orchestrator monitors Redis and calls CRIU when the flag appears.
"""
import os
import json
import time
import shutil
import subprocess
import pickle
import torch
import redis
import threading
import logging
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../shared"))
from metrics_collector import get_gpu_util, get_cpu_util, get_net_bytes

logger = logging.getLogger(__name__)


class UnifiedCheckpointManager:

    def __init__(self, robot_id: str, container_name: str,
                 checkpoint_base: str = "/tmp/swiftbot_checkpoints",
                 redis_host: str = "localhost"):
        self.robot_id = robot_id
        self.container_name = container_name
        self.chk_dir = os.path.join(checkpoint_base, robot_id)
        self.r = redis.Redis(host=redis_host, decode_responses=True)
        os.makedirs(self.chk_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # PACK (called by DHT orchestrator after it decides to migrate)
    # ------------------------------------------------------------------ #
    def pack(self, agent, success_rate: float, trigger_reason: str = "load") -> dict:
        """
        Step 1 of migration: save policy + buffer to checkpoint dir.
        CRIU is NOT called here — that happens from the host in the DHT runner.
        Returns timing dict.
        """
        t0 = time.perf_counter()
        gpu_pre = get_gpu_util()
        cpu_pre = get_cpu_util()
        net_pre = get_net_bytes()

        # Save policy weights
        weights_path = os.path.join(self.chk_dir, "policy_weights.pt")
        agent.save_checkpoint(weights_path)

        # Save replay buffer tail
        buffer_path = os.path.join(self.chk_dir, "replay_buffer.pkl")
        tail = agent.replay_buffer.tail(1000)
        with open(buffer_path, "wb") as f:
            pickle.dump(tail, f)

        # Write manifest
        manifest = {
            "robot_id": self.robot_id,
            "container_name": self.container_name,
            "migration_timestamp": time.time(),
            "trigger_reason": trigger_reason,
            "policy_version": agent.training_step,
            "replay_buffer_size": len(agent.replay_buffer),
            "success_rate_premigration": round(success_rate, 4),
            "weights_path": weights_path,
            "buffer_path": buffer_path,
        }
        manifest_path = os.path.join(self.chk_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        t_pack_ms = (time.perf_counter() - t0) * 1000

        # Signal host that policy is saved and ready for CRIU
        self.r.set(f"ready_for_criu:{self.robot_id}", "1", ex=60)

        logger.info(f"[{self.robot_id}] Pack complete in {t_pack_ms:.1f}ms. "
                    f"weights={os.path.getsize(weights_path)/1024:.1f}KB "
                    f"buffer={os.path.getsize(buffer_path)/1024:.1f}KB")

        return {
            "pack_ms": round(t_pack_ms, 2),
            "gpu_pre": gpu_pre,
            "cpu_pre": cpu_pre,
            "net_pre": net_pre,
            "weights_size_kb": os.path.getsize(weights_path) / 1024,
            "buffer_size_kb": os.path.getsize(buffer_path) / 1024,
        }

    # ------------------------------------------------------------------ #
    # RESTORE (called at destination after container is restored)
    # ------------------------------------------------------------------ #
    def restore(self, agent, checkpoint_dir: str) -> dict:
        """
        Step 3 of migration: load policy + replay buffer at destination.
        Container already restored by CRIU at this point.
        """
        t0 = time.perf_counter()

        manifest_path = os.path.join(checkpoint_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Load policy weights
        t_pol_start = time.perf_counter()
        weights_path = os.path.join(checkpoint_dir, "policy_weights.pt")
        if os.path.exists(weights_path):
            agent.load_checkpoint(weights_path)
        policy_load_ms = (time.perf_counter() - t_pol_start) * 1000

        # Load replay buffer
        buffer_path = os.path.join(checkpoint_dir, "replay_buffer.pkl")
        entries_restored = 0
        if os.path.exists(buffer_path):
            with open(buffer_path, "rb") as f:
                tail = pickle.load(f)
            agent.replay_buffer.load_tail(tail)
            entries_restored = len(tail)

        total_restore_ms = (time.perf_counter() - t0) * 1000

        logger.info(f"[{self.robot_id}] Restore complete: "
                    f"policy_load={policy_load_ms:.1f}ms "
                    f"buffer={entries_restored} entries")

        return {
            "policy_load_ms": round(policy_load_ms, 2),
            "replay_buffer_entries_restored": entries_restored,
            "total_restore_ms": round(total_restore_ms, 2),
            "success_rate_premigration": manifest.get("success_rate_premigration", 0),
        }
```

### 2.5 Flower Server (FedAvg only)

**File: `~/swiftbot_rl/dht_frl/flower_server.py`**

```python
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
N_ROUNDS        = 30     # enough rounds for policy to converge
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


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}

    # Weighted averages
    reward        = sum(n * m.get("mean_reward", 0)        for n, m in metrics) / total
    success_rate  = sum(n * m.get("success_rate", 0)       for n, m in metrics) / total
    train_loss    = sum(n * m.get("train_loss", 0)         for n, m in metrics) / total
    policy_entropy= sum(n * m.get("policy_entropy", 0)     for n, m in metrics) / total

    # Simple averages
    cpu_usage  = sum(m.get("cpu_usage", 0)  for _, m in metrics) / len(metrics)
    gpu_usage  = sum(m.get("gpu_usage", 0)  for _, m in metrics) / len(metrics)
    net_client = sum(m.get("network_mb", 0) for _, m in metrics) / len(metrics)
    train_time = max(m.get("train_time", 0) for _, m in metrics)

    total_latency = time.time() - round_start_time
    net_now       = psutil.net_io_counters()
    server_net    = ((net_now.bytes_sent + net_now.bytes_recv) - round_start_net) / (1024 * 1024)

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

    # Save CSVs
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

    # Generate graphs
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
```

### 2.6 Robot Worker Client (runs inside container)

**File: `~/swiftbot_rl/dht_frl/worker_robot_client.py`**

```python
"""
Robot worker client — runs INSIDE each Docker container.
Based on worker_client_asr_optimized.py structure.
KEEP: Flower client interface, get_parameters/set_parameters,
      fit/evaluate signatures, retry logic, psutil tracking, signal handling.
REPLACE: SimpleASR → BidPolicyMLP, LibriSpeech → SyntheticTaskGenerator,
         CTC training → PPO update, ASR metrics → robot task metrics.
"""
import os, sys, gc, time, signal, logging, argparse
import psutil, numpy as np, torch
import traceback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"

import flwr as fl
from collections import OrderedDict
import redis, json

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
TOTAL_ROUNDS    = 30        # matches server N_ROUNDS
# Forced migration at these task counts (50 per experiment = 50 events for stats)
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
                # Wait for migration to complete
                timeout = time.time() + 60
                while time.time() < timeout:
                    done = self.r.get(f"migration_done:{self.robot_id}")
                    if done:
                        self.r.delete(f"migration_done:{self.robot_id}")
                        break
                    time.sleep(0.5)

            # Get bid and execute task
            bid    = self.agent.get_bid(state)
            result = self.task_gen.execute(task)
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

        # Hardware metrics
        cpu_usage = psutil.cpu_percent()
        gpu_usage = 0.0
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated(DEVICE) / (1024 * 1024)
        net_now    = psutil.net_io_counters()
        net_mb     = (net_now.bytes_sent + net_now.bytes_recv - self.net_start) / (1024 * 1024)
        self.net_start = net_now.bytes_sent + net_now.bytes_recv

        mean_reward  = float(np.mean(rewards_this_round)) if rewards_this_round else 0.0
        success_rate = sum(self.success_hist[-20:]) / max(len(self.success_hist[-20:]), 1)
        train_loss   = max(0.0, 1.0 - success_rate)  # proxy loss for server logging

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
        sr = sum(self.success_hist[-20:]) / max(len(self.success_hist[-20:]), 1)
        loss = max(0.0, 1.0 - sr)
        return float(loss), max(self.task_counter, 1), {
            "accuracy":        float(sr),
            "loss":            float(loss),
            "eval_time":       0.1,
            "success_rate":    float(sr),
            "policy_entropy":  float(self.agent.get_entropy()),
            "cpu_usage":       float(psutil.cpu_percent()),
            "gpu_usage":       float(torch.cuda.memory_allocated(DEVICE) /
                                     (1024 * 1024)) if torch.cuda.is_available() else 0.0,
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
            fl.client.start_client(
                server_address=SERVER,
                client=client.to_client()
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
```

### 2.7 DHT FRL Runner (orchestrator with Option B migration)

**File: `~/swiftbot_rl/dht_frl/dht_frl_runner.py`**

```python
"""
DHT + FRL Orchestrator — Condition A runner.
Based on dht_asr_optimized.py. Keeps entire Kademlia DHT structure.
Adds: Option B migration trigger (from host), CRIU calls, metric logging.

Run this from Ubuntu host:
    python3 dht_frl_runner.py
"""
import asyncio, docker, os, sys, time, json, subprocess, shutil, logging, threading
import platform, socket, redis
from kademlia.network import Server
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../shared"))
from metrics_collector import MigrationMetricsWriter, get_gpu_util, get_cpu_util, get_net_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# --- CONFIG ---
DOCKER_IMAGE_NAME    = "swiftbot-robot:latest"
NUM_NODES            = 4
CLIENTS_PER_NODE     = 2
TOTAL_CLIENTS        = NUM_NODES * CLIENTS_PER_NODE   # = 8
BASE_PORT            = 8470
CHECKPOINT_BASE      = "/tmp/swiftbot_checkpoints"
RESULT_DIR           = os.path.join(os.path.dirname(__file__), "results")
REDIS_HOST           = "localhost"
OVERLOAD_THRESHOLD   = 0.85   # 85% CPU or GPU triggers migration

metrics_writer = MigrationMetricsWriter("dht_frl", RESULT_DIR)
r_client       = redis.Redis(host=REDIS_HOST, decode_responses=True)


def get_master_ip() -> str:
    if sys.platform == "win32" or "microsoft" in platform.uname().release.lower():
        return "host.docker.internal"
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip


class DHTNode:
    """Kademlia DHT node — manages 2 robot containers each."""

    def __init__(self, node_id, port, bootstrap=None):
        self.node_id   = node_id
        self.port      = port
        self.bootstrap = bootstrap
        self.server    = Server()
        self.docker    = docker.from_env()
        self.container_names = []

    async def start(self, master_ip):
        await self.server.listen(self.port)
        if self.bootstrap:
            await self.server.bootstrap([self.bootstrap])
        await self.launch_workers(master_ip)

    async def launch_workers(self, master_ip):
        logger.info(f"[Node {self.node_id}] Launching containers...")
        for i in range(CLIENTS_PER_NODE):
            cid   = self.node_id * CLIENTS_PER_NODE + i
            cname = f"swiftbot-robot-{cid}"
            ctype = "gpu_specialist" if cid < 4 else "cpu_specialist"

            try:
                try:
                    c = self.docker.containers.get(cname)
                    env_vars = c.attrs["Config"]["Env"]
                    master_match = any(master_ip in e for e in env_vars
                                      if "MASTER_ADDRESS" in e)
                    if c.status == "running" and master_match:
                        logger.info(f"  {cname} already running correctly")
                        self.container_names.append(cname)
                        continue
                    c.remove(force=True)
                except docker.errors.NotFound:
                    pass

                cmd = (f"python3 /app/worker_robot_client.py "
                       f"--client-id {cid} --num-clients {TOTAL_CLIENTS} "
                       f"--container-type {ctype}")

                os.makedirs(f"{CHECKPOINT_BASE}/robot_{cid:03d}", exist_ok=True)

                self.docker.containers.run(
                    DOCKER_IMAGE_NAME,
                    command=cmd,
                    name=cname,
                    detach=True,
                    tty=True,
                    shm_size="4g",
                    environment={
                        "MASTER_ADDRESS": f"{master_ip}:8080",
                        "REDIS_HOST":     REDIS_HOST,
                        "NVIDIA_VISIBLE_DEVICES": "all",
                        "PYTHONUNBUFFERED": "1",
                    },
                    device_requests=[
                        docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                    ],
                    volumes={
                        CHECKPOINT_BASE: {
                            "bind": "/checkpoints", "mode": "rw"
                        }
                    },
                    security_opt=["seccomp:unconfined"],   # required for CRIU
                    restart_policy={"Name": "on-failure", "MaximumRetryCount": 3},
                    network_mode="host",
                )
                self.container_names.append(cname)
                logger.info(f"  Started {cname} ({ctype})")

            except Exception as e:
                logger.error(f"  Failed to start {cname}: {e}")
            await asyncio.sleep(0.5)


# ------------------------------------------------------------------ #
# OPTION B MIGRATION — triggered from HOST by DHT orchestrator
# ------------------------------------------------------------------ #

def trigger_unified_migration(robot_id: str, container_name: str,
                               source_node: str, dest_node: str):
    """
    Full unified migration sequence for DHT+FRL system.
    Called from host when migration_request Redis key appears.
    """
    logger.info(f"[MIGRATION] Starting unified migration for {robot_id}")

    t_trigger     = time.perf_counter()
    gpu_pre       = get_gpu_util()
    cpu_pre       = get_cpu_util()
    net_pre       = get_net_bytes()

    chk_src = os.path.join(CHECKPOINT_BASE, robot_id)
    chk_dst = os.path.join(CHECKPOINT_BASE, f"{robot_id}_dest")
    os.makedirs(chk_dst, exist_ok=True)

    # --- Wait for container to save policy + buffer ---
    logger.info(f"  Waiting for {robot_id} to save policy...")
    deadline = time.time() + 30
    while time.time() < deadline:
        if r_client.get(f"ready_for_criu:{robot_id}"):
            break
        time.sleep(0.2)

    # --- CRIU checkpoint from HOST (Option B) ---
    t_dump_start = time.perf_counter()
    gpu_during   = get_gpu_util()
    cpu_during   = get_cpu_util()

    criu_dir = os.path.join(chk_src, "criu")
    os.makedirs(criu_dir, exist_ok=True)

    criu_result = subprocess.run([
        "docker", "checkpoint", "create",
        f"--checkpoint-dir={criu_dir}",
        "--leave-running",
        container_name, "migration_chk"
    ], capture_output=True, text=True, timeout=120)

    t_dump_done    = time.perf_counter()
    dump_ms        = (t_dump_done - t_dump_start) * 1000
    criu_ok        = criu_result.returncode == 0

    if not criu_ok:
        logger.warning(f"  CRIU checkpoint warning: {criu_result.stderr[:200]}")

    # Get checkpoint size
    chk_size_mb = 0.0
    if os.path.exists(criu_dir):
        chk_size_mb = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, files in os.walk(criu_dir)
            for f in files
        ) / (1024 * 1024)

    # --- Parallel transfer: CRIU checkpoint + policy + replay buffer ---
    t_transfer_start = time.perf_counter()
    transfer_results = {}

    def transfer_criu():
        t = time.perf_counter()
        dst = os.path.join(chk_dst, "criu")
        if os.path.exists(criu_dir):
            shutil.copytree(criu_dir, dst, dirs_exist_ok=True)
        transfer_results["criu_ms"] = (time.perf_counter() - t) * 1000

    def transfer_policy():
        t = time.perf_counter()
        for fname in ["policy_weights.pt", "replay_buffer.pkl", "manifest.json"]:
            src = os.path.join(chk_src, fname)
            dst = os.path.join(chk_dst, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        transfer_results["policy_ms"] = (time.perf_counter() - t) * 1000

    t1 = threading.Thread(target=transfer_criu)
    t2 = threading.Thread(target=transfer_policy)
    t1.start(); t2.start()
    t1.join();  t2.join()

    t_transfer_done = time.perf_counter()
    transfer_ms     = (t_transfer_done - t_transfer_start) * 1000

    # --- Restore container at destination ---
    t_restore_start = time.perf_counter()

    dest_cname = f"{container_name}_dest"
    restore_result = subprocess.run([
        "docker", "start",
        f"--checkpoint-dir={os.path.join(chk_dst, 'criu')}",
        f"--checkpoint=migration_chk",
        dest_cname
    ], capture_output=True, text=True, timeout=60)

    t_restore_done = time.perf_counter()
    restore_ms     = (t_restore_done - t_restore_start) * 1000

    # --- Signal robot to load policy (policy_load_ms measured inside container) ---
    r_client.set(f"load_policy:{robot_id}", chk_dst, ex=60)

    # Wait for robot to confirm policy loaded and first bid submitted
    deadline_bid = time.time() + 30
    policy_load_ms = 0.0
    while time.time() < deadline_bid:
        data = r_client.get(f"first_bid_after_migration:{robot_id}")
        if data:
            info = json.loads(data)
            policy_load_ms = float(info.get("policy_load_ms", 0))
            r_client.delete(f"first_bid_after_migration:{robot_id}")
            break
        time.sleep(0.1)

    t_fully_operational = time.perf_counter()
    total_MTT_ms = (t_fully_operational - t_trigger) * 1000
    downtime_ms  = total_MTT_ms  # simplified for same-machine simulation

    net_post     = get_net_bytes()
    gpu_post     = get_gpu_util()
    cpu_post     = get_cpu_util()
    net_bytes    = net_post - net_pre

    # Signal migration complete to robot
    r_client.set(f"migration_done:{robot_id}", "1", ex=60)

    logger.info(f"[MIGRATION] {robot_id} complete: "
                f"MTT={total_MTT_ms:.0f}ms  "
                f"dump={dump_ms:.0f}ms  "
                f"transfer={transfer_ms:.0f}ms  "
                f"policy_load={policy_load_ms:.0f}ms")

    # Return timing dict for metrics writer
    return {
        "robot_id":                    robot_id,
        "trigger_to_dump_ms":          dump_ms,
        "dump_to_transfer_ms":         transfer_ms,
        "transfer_to_restore_ms":      restore_ms,
        "policy_load_ms":              policy_load_ms,
        "downtime_ms":                 downtime_ms,
        "total_MTT_ms":                total_MTT_ms,
        "gpu_util_pre_migration":      gpu_pre,
        "gpu_util_during_migration":   gpu_during,
        "gpu_util_post_migration":     gpu_post,
        "cpu_util_pre_migration":      cpu_pre,
        "cpu_util_during_migration":   cpu_during,
        "cpu_util_post_migration":     cpu_post,
        "network_bytes_transferred":   net_bytes,
        "checkpoint_size_mb":          round(chk_size_mb, 2),
        "criu_mode":                   "unified",
    }


def migration_monitor_thread():
    """
    Background thread — watches Redis for migration requests from containers.
    When a request appears, triggers unified migration (Option B).
    """
    logger.info("[Monitor] Migration monitor started")
    while True:
        try:
            keys = r_client.keys("migration_request:robot_*")
            for key in keys:
                raw = r_client.get(key)
                if not raw:
                    continue
                info      = json.loads(raw)
                robot_id  = info["robot_id"]
                cid       = int(robot_id.split("_")[1])
                cname     = f"swiftbot-robot-{cid}"

                r_client.delete(key)

                # Get pre-migration success rate from Redis task logs
                success_rate_pre = float(info.get("success_rate", 0))

                # Trigger migration
                mig_metrics = trigger_unified_migration(
                    robot_id, cname, "node_src", "node_dst"
                )
                mig_metrics["success_rate_pre"] = success_rate_pre
                mig_metrics["criu_mode"]        = "unified"

                # Post-migration: wait 10 tasks and measure regression
                # (Measured in evaluation phase from task_logs CSV)
                metrics_writer.write_event(mig_metrics)

        except Exception as e:
            logger.error(f"[Monitor] Error: {e}")
        time.sleep(1)


async def main():
    master_ip = get_master_ip()
    logger.info(f"Master IP: {master_ip}")

    # Start migration monitor as background thread
    monitor = threading.Thread(target=migration_monitor_thread, daemon=True)
    monitor.start()

    # Create 4 DHT nodes (matches original dht_asr_optimized.py structure)
    nodes = (
        [DHTNode(0, BASE_PORT)] +
        [DHTNode(i, BASE_PORT + i, ("127.0.0.1", BASE_PORT))
         for i in range(1, NUM_NODES)]
    )

    logger.info(f"Launching {TOTAL_CLIENTS} robot containers...")
    await asyncio.gather(*[n.start(master_ip) for n in nodes])

    logger.info("\n[SUCCESS] All containers running.")
    logger.info("Now run the Flower server in a separate terminal:")
    logger.info("  cd ~/swiftbot_rl/dht_frl && python3 flower_server.py")
    logger.info("\nPress Ctrl+C to stop containers when experiment completes.\n")

    try:
        while True:
            await asyncio.sleep(10)
    except KeyboardInterrupt:
        logger.info("Stopping...")


if __name__ == "__main__":
    asyncio.run(main())
```

### 2.8 Dockerfile for DHT+FRL

**File: `~/swiftbot_rl/dht_frl/Dockerfile`**

```dockerfile
# SwiftBot-RL robot container — DHT+FRL condition
# Based on Dockerfile.optimized — REMOVE torchaudio/jiwer/libsndfile/ffmpeg/sox
# ADD criu, stable-baselines3, gymnasium, pynvml, redis

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV GRPC_VERBOSITY=ERROR
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# System packages — ADD criu, REMOVE libsndfile/ffmpeg/sox
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    criu \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# PyTorch (no torchaudio)
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Robot dependencies — REMOVE jiwer/torchaudio, ADD new packages
RUN pip3 install --no-cache-dir \
    flwr==1.5.0 \
    psutil==5.9.8 \
    pynvml==11.5.0 \
    redis==5.0.1 \
    numpy==1.24.4 \
    stable-baselines3==2.2.1 \
    gymnasium==0.29.1 \
    pandas==2.0.3 \
    matplotlib==3.7.2

# Copy robot code
COPY worker_robot_client.py /app/
COPY robot/ /app/robot/

# Create checkpoint directory (host mounts /tmp/swiftbot_checkpoints here)
RUN mkdir -p /checkpoints /app/results

# Healthcheck
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD pgrep -f python3 || exit 1

CMD ["python3", "--version"]
```

---

## PHASE 3 — CRIU Cold Baseline (Condition B)

### 3.1 Random Policy Worker (no PPO, no Flower)

**File: `~/swiftbot_rl/criu_cold/worker_random_client.py`**

```python
"""
Random policy worker — CRIU cold baseline.
No PPO, no FedAvg, no Flower. Just executes tasks with random bidding.
Logs task results to Redis for metric collection.
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

TOTAL_TASKS           = 1000
FORCED_MIGRATION_TASKS = set(range(50, 1050, 20))


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

    logger.info(f"[{robot_id}] Random policy worker started (CRIU cold baseline)")

    for task_counter in range(TOTAL_TASKS):
        if shutdown_requested:
            break

        # Check for forced migration signal
        if task_counter in FORCED_MIGRATION_TASKS:
            sr = sum(success_hist[-10:]) / max(len(success_hist[-10:]), 1)
            r.set(f"migration_request:{robot_id}", json.dumps({
                "robot_id":     robot_id,
                "success_rate": sr,
                "task_counter": task_counter,
            }), ex=30)
            # Wait for migration done signal
            deadline = time.time() + 60
            while time.time() < deadline:
                if r.get(f"migration_done:{robot_id}"):
                    r.delete(f"migration_done:{robot_id}")
                    break
                time.sleep(0.5)

        task   = task_gen.generate()
        bid    = np.random.uniform(0, 1)   # RANDOM POLICY — no learning
        result = task_gen.execute(task)
        success = result["status"] == "success"
        success_hist.append(1 if success else 0)

        sr = sum(success_hist[-10:]) / max(len(success_hist[-10:]), 1)
        r.lpush("task_logs", json.dumps({
            "robot_id":               robot_id,
            "task_counter":           task_counter,
            "fl_round":               0,
            "task_type":              task["task_type"],
            "complexity":             task["complexity"],
            "bid_value":              round(float(bid), 4),
            "reward":                 1.0 if success else -1.0,
            "status":                 result["status"],
            "exec_latency_ms":        round(result.get("latency_ms", 0), 2),
            "deadline_ms":            task["deadline_ms"],
            "success_rate_rolling10": round(sr, 4),
            "policy_entropy":         1.0,   # always max — random policy
            "training_step":          0,
        }))
        r.ltrim("task_logs", 0, 99999)
        r.setex(f"robot_load:{robot_id}", 30, json.dumps({
            "robot_id": robot_id,
            "cpu_util": psutil.cpu_percent() / 100.0,
            "task_count": task_counter,
        }))

        if task_counter % 100 == 0:
            logger.info(f"[{robot_id}] Tasks: {task_counter}/{TOTAL_TASKS} "
                        f"success_rate={sr:.3f}")

    r.set(f"robot_done:{robot_id}", "1")
    logger.info(f"[{robot_id}] Complete.")
```

### 3.2 CRIU Cold Runner

**File: `~/swiftbot_rl/criu_cold/criu_cold_runner.py`**

```python
"""
CRIU Cold Baseline Runner — Condition B.
Launches 8 containers with random policy workers.
Triggers CRIU cold (stop-and-copy) migration on overload.
No FedAvg, no PPO, no Flower server.
"""
import asyncio, docker, os, sys, time, json, subprocess, shutil
import logging, threading, socket, platform, redis
from kademlia.network import Server

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../shared"))
from metrics_collector import MigrationMetricsWriter, get_gpu_util, get_cpu_util, get_net_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

DOCKER_IMAGE_NAME = "swiftbot-robot:latest"
NUM_NODES         = 4
CLIENTS_PER_NODE  = 2
TOTAL_CLIENTS     = 8
BASE_PORT         = 8480   # different port to avoid conflict
CHECKPOINT_BASE   = "/tmp/swiftbot_checkpoints_criu_cold"
RESULT_DIR        = os.path.join(os.path.dirname(__file__), "results")
REDIS_HOST        = "localhost"

metrics_writer = MigrationMetricsWriter("criu_cold", RESULT_DIR)
r_client       = redis.Redis(host=REDIS_HOST, decode_responses=True)


def get_master_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]; s.close(); return ip


class DHTNode:
    def __init__(self, node_id, port, bootstrap=None):
        self.node_id   = node_id
        self.port      = port
        self.bootstrap = bootstrap
        self.server    = Server()
        self.docker    = docker.from_env()

    async def start(self, master_ip):
        await self.server.listen(self.port)
        if self.bootstrap:
            await self.server.bootstrap([self.bootstrap])
        await self.launch_workers(master_ip)

    async def launch_workers(self, master_ip):
        for i in range(CLIENTS_PER_NODE):
            cid   = self.node_id * CLIENTS_PER_NODE + i
            cname = f"swiftbot-criu-cold-{cid}"
            ctype = "gpu_specialist" if cid < 4 else "cpu_specialist"
            try:
                try:
                    self.docker.containers.get(cname).remove(force=True)
                except docker.errors.NotFound:
                    pass
                cmd = (f"python3 /app/worker_random_client.py "
                       f"--client-id {cid} --container-type {ctype}")
                os.makedirs(f"{CHECKPOINT_BASE}/robot_{cid:03d}", exist_ok=True)
                self.docker.containers.run(
                    DOCKER_IMAGE_NAME, command=cmd, name=cname,
                    detach=True, tty=True, shm_size="4g",
                    environment={
                        "REDIS_HOST": REDIS_HOST,
                        "NVIDIA_VISIBLE_DEVICES": "all",
                        "PYTHONUNBUFFERED": "1",
                    },
                    device_requests=[docker.types.DeviceRequest(
                        count=-1, capabilities=[["gpu"]])],
                    volumes={CHECKPOINT_BASE: {"bind": "/checkpoints", "mode": "rw"}},
                    security_opt=["seccomp:unconfined"],
                    network_mode="host",
                )
                logger.info(f"  Started {cname}")
            except Exception as e:
                logger.error(f"  Failed {cname}: {e}")
            await asyncio.sleep(0.5)


def trigger_criu_cold_migration(robot_id: str, container_name: str) -> dict:
    """CRIU cold: fully stop container, dump, transfer, restore."""
    logger.info(f"[CRIU COLD] Migrating {robot_id}")
    t_trigger  = time.perf_counter()
    gpu_pre    = get_gpu_util()
    cpu_pre    = get_cpu_util()
    net_pre    = get_net_bytes()

    chk_src = os.path.join(CHECKPOINT_BASE, robot_id)
    chk_dst = os.path.join(CHECKPOINT_BASE, f"{robot_id}_dest")
    os.makedirs(chk_dst, exist_ok=True)
    criu_dir = os.path.join(chk_src, "criu_cold")
    os.makedirs(criu_dir, exist_ok=True)

    # Step 1: STOP container and dump (cold — no --leave-running)
    t_dump_start = time.perf_counter()
    gpu_during   = get_gpu_util()
    cpu_during   = get_cpu_util()

    result = subprocess.run([
        "docker", "checkpoint", "create",
        f"--checkpoint-dir={criu_dir}",
        container_name, "cold_chk"      # no --leave-running = container stops
    ], capture_output=True, text=True, timeout=120)

    dump_ms = (time.perf_counter() - t_dump_start) * 1000

    # Get checkpoint size
    chk_size_mb = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(criu_dir) for f in files
    ) / (1024 * 1024) if os.path.exists(criu_dir) else 0

    # Step 2: Transfer (sequential — must complete dump first)
    t_xfer = time.perf_counter()
    shutil.copytree(criu_dir, os.path.join(chk_dst, "criu_cold"), dirs_exist_ok=True)
    transfer_ms = (time.perf_counter() - t_xfer) * 1000

    # Step 3: Restore
    t_restore = time.perf_counter()
    subprocess.run([
        "docker", "start",
        f"--checkpoint-dir={os.path.join(chk_dst, 'criu_cold')}",
        f"--checkpoint=cold_chk",
        f"{container_name}_dest"
    ], capture_output=True, timeout=60)
    restore_ms = (time.perf_counter() - t_restore) * 1000

    total_MTT_ms = (time.perf_counter() - t_trigger) * 1000
    net_post     = get_net_bytes()

    r_client.set(f"migration_done:{robot_id}", "1", ex=60)

    logger.info(f"[CRIU COLD] {robot_id}: MTT={total_MTT_ms:.0f}ms "
                f"dump={dump_ms:.0f}ms transfer={transfer_ms:.0f}ms")

    return {
        "robot_id":                    robot_id,
        "trigger_to_dump_ms":          dump_ms,
        "dump_to_transfer_ms":         transfer_ms,
        "transfer_to_restore_ms":      restore_ms,
        "policy_load_ms":              0,       # CRIU cold has no policy to load
        "downtime_ms":                 total_MTT_ms,
        "total_MTT_ms":                total_MTT_ms,
        "gpu_util_pre_migration":      gpu_pre,
        "gpu_util_during_migration":   gpu_during,
        "gpu_util_post_migration":     get_gpu_util(),
        "cpu_util_pre_migration":      cpu_pre,
        "cpu_util_during_migration":   cpu_during,
        "cpu_util_post_migration":     get_cpu_util(),
        "network_bytes_transferred":   net_post - net_pre,
        "checkpoint_size_mb":          round(chk_size_mb, 2),
        "criu_mode":                   "cold",
    }


def migration_monitor_thread():
    logger.info("[Monitor COLD] Started")
    while True:
        try:
            for key in r_client.keys("migration_request:robot_*"):
                raw = r_client.get(key)
                if not raw: continue
                info     = json.loads(raw)
                robot_id = info["robot_id"]
                cid      = int(robot_id.split("_")[1])
                cname    = f"swiftbot-criu-cold-{cid}"
                r_client.delete(key)
                mig = trigger_criu_cold_migration(robot_id, cname)
                mig["success_rate_pre"] = float(info.get("success_rate", 0))
                metrics_writer.write_event(mig)
        except Exception as e:
            logger.error(f"[Monitor] {e}")
        time.sleep(1)


async def main():
    master_ip = get_master_ip()
    os.makedirs(CHECKPOINT_BASE, exist_ok=True)
    threading.Thread(target=migration_monitor_thread, daemon=True).start()
    nodes = (
        [DHTNode(0, BASE_PORT)] +
        [DHTNode(i, BASE_PORT + i, ("127.0.0.1", BASE_PORT)) for i in range(1, NUM_NODES)]
    )
    await asyncio.gather(*[n.start(master_ip) for n in nodes])
    logger.info("\n[CRIU COLD] All containers running. Waiting for experiment to complete...")
    try:
        while True:
            done = sum(1 for i in range(TOTAL_CLIENTS)
                       if r_client.get(f"robot_done:robot_{i:03d}"))
            if done >= TOTAL_CLIENTS:
                logger.info("All robots done.")
                break
            await asyncio.sleep(10)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    asyncio.run(main())
```

### 3.3 CRIU Warm Runner

**File: `~/swiftbot_rl/criu_warm/criu_warm_runner.py`**

```python
"""
CRIU Pre-copy (Warm) Baseline Runner — Condition C.
Same as criu_cold_runner but uses iterative pre-dumps.
Container keeps running during pre-dumps; only final delta pauses.
"""
# This file is identical to criu_cold_runner.py with ONE change:
# In trigger_criu_migration(), use --leave-running and iterate pre-dumps
# before the final dump.
#
# Full implementation: copy criu_cold_runner.py and replace
# trigger_criu_cold_migration with the function below.
# Change all "cold" references to "warm" and BASE_PORT to 8490.

import subprocess, os, time, shutil
from metrics_collector import get_gpu_util, get_cpu_util, get_net_bytes


def trigger_criu_warm_migration(robot_id, container_name,
                                 checkpoint_base, r_client) -> dict:
    """CRIU pre-copy: pre-dump while running, then small final pause."""
    t_trigger = time.perf_counter()
    gpu_pre   = get_gpu_util()
    cpu_pre   = get_cpu_util()
    net_pre   = get_net_bytes()

    chk_src  = os.path.join(checkpoint_base, robot_id)
    chk_dst  = os.path.join(checkpoint_base, f"{robot_id}_dest")
    criu_dir = os.path.join(chk_src, "criu_warm")
    os.makedirs(criu_dir, exist_ok=True)
    os.makedirs(chk_dst, exist_ok=True)

    # Pre-dump iterations (container stays running)
    t_predump_start = time.perf_counter()
    gpu_during      = get_gpu_util()
    cpu_during      = get_cpu_util()

    for iteration in range(3):   # 3 pre-dump rounds
        predump_dir = os.path.join(criu_dir, f"predump_{iteration}")
        os.makedirs(predump_dir, exist_ok=True)
        subprocess.run([
            "docker", "checkpoint", "create",
            f"--checkpoint-dir={predump_dir}",
            "--leave-running",          # container keeps running!
            container_name, f"predump_{iteration}"
        ], capture_output=True, text=True, timeout=60)
        time.sleep(0.05)   # brief pause between iterations

    # Final delta dump (short pause — only dirty pages since last pre-dump)
    t_final_start = time.perf_counter()
    final_dir = os.path.join(criu_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    subprocess.run([
        "docker", "checkpoint", "create",
        f"--checkpoint-dir={final_dir}",
        container_name, "warm_chk"   # no --leave-running = container pauses
    ], capture_output=True, text=True, timeout=120)

    dump_ms = (time.perf_counter() - t_final_start) * 1000   # just the final pause

    # Transfer and restore (same as cold)
    t_xfer = time.perf_counter()
    shutil.copytree(criu_dir, os.path.join(chk_dst, "criu_warm"), dirs_exist_ok=True)
    transfer_ms = (time.perf_counter() - t_xfer) * 1000

    t_restore = time.perf_counter()
    subprocess.run([
        "docker", "start",
        f"--checkpoint-dir={os.path.join(chk_dst, 'criu_warm', 'final')}",
        f"--checkpoint=warm_chk",
        f"{container_name}_dest"
    ], capture_output=True, timeout=60)
    restore_ms = (time.perf_counter() - t_restore) * 1000

    total_MTT_ms = (time.perf_counter() - t_trigger) * 1000

    r_client.set(f"migration_done:{robot_id}", "1", ex=60)

    return {
        "robot_id":                   robot_id,
        "trigger_to_dump_ms":         dump_ms,      # final delta only
        "dump_to_transfer_ms":        transfer_ms,
        "transfer_to_restore_ms":     restore_ms,
        "policy_load_ms":             0,
        "downtime_ms":                dump_ms + restore_ms,   # only final pause counts
        "total_MTT_ms":               total_MTT_ms,
        "gpu_util_pre_migration":     gpu_pre,
        "gpu_util_during_migration":  gpu_during,
        "gpu_util_post_migration":    get_gpu_util(),
        "cpu_util_pre_migration":     cpu_pre,
        "cpu_util_during_migration":  cpu_during,
        "cpu_util_post_migration":    get_cpu_util(),
        "network_bytes_transferred":  get_net_bytes() - net_pre,
        "checkpoint_size_mb":         0,
        "criu_mode":                  "precopy",
    }
```

---

## PHASE 4 — Evaluation and Comparison

### 4.1 Comparison script

**File: `~/swiftbot_rl/evaluation/compare_all.py`**

```python
"""
Final comparison script — reads all 3 results directories and produces
4 paper-ready figures + 1 LaTeX summary table.
Run AFTER all 3 conditions have completed and metrics are collected.

Usage:
  python3 evaluation/compare_all.py
"""
import os, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = {
    "DHT+FRL (ours)":     "dht_frl/results/migration_events.csv",
    "CRIU cold":          "criu_cold/results/migration_events.csv",
    "CRIU warm (pre-copy)": "criu_warm/results/migration_events.csv",
}
COLORS = {
    "DHT+FRL (ours)":       "#1D9E75",
    "CRIU cold":            "#E24B4A",
    "CRIU warm (pre-copy)": "#7F77DD",
}
OUT_DIR = "evaluation/figures"
os.makedirs(OUT_DIR, exist_ok=True)
ROOT    = os.path.join(os.path.dirname(__file__), "..")


def load_all():
    data = {}
    for label, path in RESULTS.items():
        full = os.path.join(ROOT, path)
        if os.path.exists(full):
            data[label] = pd.read_csv(full)
            print(f"Loaded {label}: {len(data[label])} migration events")
        else:
            print(f"WARNING: {full} not found — skipping {label}")
    return data


def fig1_mtt_stacked_bar(data):
    """Figure 1: Migration Total Time breakdown — stacked bar."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x  = np.arange(len(data))
    w  = 0.5
    for i, (label, df) in enumerate(data.items()):
        dump     = df["trigger_to_dump_ms"].mean()
        transfer = df["dump_to_transfer_ms"].mean()
        restore  = df["transfer_to_restore_ms"].mean()
        policy   = df["policy_load_ms"].mean()
        color    = COLORS[label]
        ax.bar(i, dump,     w, label="Dump"          if i == 0 else "", color="#4A90D9")
        ax.bar(i, transfer, w, bottom=dump,           label="Transfer"    if i == 0 else "", color="#F5A623")
        ax.bar(i, restore,  w, bottom=dump+transfer,  label="Restore"     if i == 0 else "", color="#7ED321")
        ax.bar(i, policy,   w, bottom=dump+transfer+restore,
               label="Policy load (new)" if i == 0 else "", color=color, alpha=0.9)
        total = dump + transfer + restore + policy
        ax.text(i, total + 5, f"{total:.0f}ms", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(list(data.keys()), rotation=15, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Fig 1 — Migration Total Time breakdown")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig1_MTT_breakdown.png", dpi=150)
    plt.close(fig)
    print("Saved fig1_MTT_breakdown.png")


def fig2_regression_boxplot(data):
    """Figure 2: Policy regression distribution across 50 migration events."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_data  = [df["regression_pct"].dropna().values for df in data.values()]
    labels     = list(data.keys())
    colors     = [COLORS[l] for l in labels]
    bp = ax.boxplot(plot_data, patch_artist=True, notch=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Policy regression (%)")
    ax.set_title("Fig 2 — Policy regression per migration event")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig2_regression_boxplot.png", dpi=150)
    plt.close(fig)
    print("Saved fig2_regression_boxplot.png")


def fig3_downtime_cdf(data):
    """Figure 3: CDF of downtime across migration events."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, df in data.items():
        dt     = np.sort(df["downtime_ms"].dropna().values)
        cdf    = np.arange(1, len(dt) + 1) / len(dt)
        ax.plot(dt, cdf, label=label, color=COLORS[label], linewidth=2)
    ax.set_xlabel("Downtime (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Fig 3 — CDF of robot downtime during migration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig3_downtime_cdf.png", dpi=150)
    plt.close(fig)
    print("Saved fig3_downtime_cdf.png")


def fig4_gpu_cpu_during_migration(data):
    """Figure 4: GPU and CPU utilization during migration window."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    labels = list(data.keys())
    x = np.arange(len(labels))
    for ax, metric, title in [
        (ax1, "gpu_util_during_migration", "GPU utilization during migration"),
        (ax2, "cpu_util_during_migration", "CPU utilization during migration"),
    ]:
        means  = [data[l][metric].mean() for l in labels]
        stds   = [data[l][metric].std()  for l in labels]
        colors = [COLORS[l] for l in labels]
        ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Utilization (0-1)")
        ax.set_title(title)
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig4_resource_usage.png", dpi=150)
    plt.close(fig)
    print("Saved fig4_resource_usage.png")


def summary_table(data):
    """LaTeX and CSV summary table."""
    rows = []
    for label, df in data.items():
        rows.append({
            "System":                   label,
            "Mean MTT (ms)":            f"{df['total_MTT_ms'].mean():.1f} ± {df['total_MTT_ms'].std():.1f}",
            "Mean downtime (ms)":       f"{df['downtime_ms'].mean():.1f} ± {df['downtime_ms'].std():.1f}",
            "Mean regression (%)":      f"{df['regression_pct'].mean():.1f} ± {df['regression_pct'].std():.1f}",
            "Policy load (ms)":         f"{df['policy_load_ms'].mean():.1f}",
            "GPU during migration":     f"{df['gpu_util_during_migration'].mean():.2f}",
            "CPU during migration":     f"{df['cpu_util_during_migration'].mean():.2f}",
            "Net bytes xferred (MB)":   f"{df['network_bytes_transferred'].mean() / 1e6:.1f}",
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(f"{OUT_DIR}/summary_table.csv", index=False)
    with open(f"{OUT_DIR}/summary_table.tex", "w") as f:
        f.write(summary.to_latex(index=False, escape=False))
    print("\n=== SUMMARY TABLE ===")
    print(summary.to_string(index=False))
    print(f"\nSaved summary_table.csv and summary_table.tex to {OUT_DIR}/")


if __name__ == "__main__":
    data = load_all()
    if not data:
        print("No data found. Run experiments first.")
        exit(1)
    fig1_mtt_stacked_bar(data)
    fig2_regression_boxplot(data)
    fig3_downtime_cdf(data)
    fig4_gpu_cpu_during_migration(data)
    summary_table(data)
    print("\nAll comparison figures saved to evaluation/figures/")
```

---

## PHASE 5 — Build Docker Image

**Do this ONCE after all code is written. Build from `~/swiftbot_rl/dht_frl/`:**

```bash
cd ~/swiftbot_rl/dht_frl
docker build -t swiftbot-robot:latest .
# Takes 5-15 minutes on first run (downloads PyTorch)
docker images | grep swiftbot
# Must show: swiftbot-robot   latest   <image_id>   <size>
```

---

## PHASE 6 — STEP-BY-STEP INSTRUCTIONS TO RUN THE EXPERIMENT

**DO NOT run the experiment automatically. Follow these steps manually one at a time.**

### Step 1 — Verify everything before starting

```bash
# On Ubuntu host:
redis-cli ping                                    # must: PONG
nvidia-smi                                        # must: show RTX 4080
criu --version                                    # must: 3.x
docker images | grep swiftbot                     # must: swiftbot-robot:latest
python3 -c "import flwr, kademlia, redis, torch; print('imports OK')"
ls ~/swiftbot_rl/dht_frl/robot/                  # must show all 4 .py files
ls ~/swiftbot_rl/criu_cold/
ls ~/swiftbot_rl/criu_warm/
```

### Step 2 — CRIU test (must pass before any experiment)

```bash
docker run -d --name criu_smoke --security-opt seccomp:unconfined \
  alpine sleep 3600
docker checkpoint create --checkpoint-dir=/tmp/criu_smoke_test \
  criu_smoke chk1
echo "Exit code: $?"    # must be 0
docker rm -f criu_smoke
```

### Step 3 — Run Experiment A: DHT + FRL (YOUR SYSTEM)

Open THREE terminals on Ubuntu.

**Terminal 1 — Start the Flower server first:**
```bash
cd ~/swiftbot_rl/dht_frl
python3 flower_server.py
# Wait until you see: "Waiting for 8 robot clients to connect..."
```

**Terminal 2 — Start the DHT orchestrator and containers:**
```bash
cd ~/swiftbot_rl/dht_frl
python3 dht_frl_runner.py
# Watch for: "Started swiftbot-robot-0" through "Started swiftbot-robot-7"
# Then: "All containers running."
```

**Terminal 3 — Monitor progress:**
```bash
# Watch task logs accumulate:
watch -n 5 "redis-cli LLEN task_logs"
# Watch migration events:
watch -n 5 "wc -l ~/swiftbot_rl/dht_frl/results/migration_events.csv"
# Watch container status:
watch -n 10 "docker ps --format 'table {{.Names}}\t{{.Status}}'"
```

**Wait for completion:**
```bash
# Experiment completes when Flower server finishes 30 rounds
# You will see in Terminal 1: "FedAvg complete. Results saved."
# Stop containers: Ctrl+C in Terminal 2
```

**Collect results:**
```bash
redis-cli FLUSHDB    # CRITICAL: clear Redis before next experiment
echo "Experiment A complete. Results in ~/swiftbot_rl/dht_frl/results/"
ls ~/swiftbot_rl/dht_frl/results/
```

### Step 4 — Run Experiment B: CRIU Cold Baseline

Open TWO terminals (no Flower server needed — no FRL in CRIU conditions).

**Terminal 1 — Start CRIU cold runner:**
```bash
cd ~/swiftbot_rl/criu_cold
python3 criu_cold_runner.py
# Watch for: "Started swiftbot-criu-cold-0" through "swiftbot-criu-cold-7"
```

**Terminal 2 — Monitor:**
```bash
watch -n 5 "redis-cli LLEN task_logs"
watch -n 5 "wc -l ~/swiftbot_rl/criu_cold/results/migration_events.csv"
```

**Wait for completion:**
```bash
# Completes when you see: "All robots done."
# Stop: Ctrl+C in Terminal 1
redis-cli FLUSHDB    # CRITICAL: clear Redis before next experiment
echo "Experiment B complete."
ls ~/swiftbot_rl/criu_cold/results/
```

### Step 5 — Run Experiment C: CRIU Warm (Pre-copy) Baseline

```bash
cd ~/swiftbot_rl/criu_warm
python3 criu_warm_runner.py
# Same monitoring as Step 4
# Completes when: "All robots done."
# Stop: Ctrl+C
redis-cli FLUSHDB
echo "Experiment C complete."
ls ~/swiftbot_rl/criu_warm/results/
```

### Step 6 — Generate Comparison Figures

```bash
cd ~/swiftbot_rl
python3 evaluation/compare_all.py
```

**Expected output:**
```
Loaded DHT+FRL (ours): 50 migration events
Loaded CRIU cold: 50 migration events
Loaded CRIU warm (pre-copy): 50 migration events
Saved fig1_MTT_breakdown.png
Saved fig2_regression_boxplot.png
Saved fig3_downtime_cdf.png
Saved fig4_resource_usage.png

=== SUMMARY TABLE ===
[table printed here]

Saved summary_table.csv and summary_table.tex to evaluation/figures/
```

**View results:**
```bash
ls ~/swiftbot_rl/evaluation/figures/
# fig1_MTT_breakdown.png    ← your headline migration time result
# fig2_regression_boxplot.png  ← proves policy regression advantage
# fig3_downtime_cdf.png     ← CDF of downtime across all events
# fig4_resource_usage.png   ← GPU/CPU during migration
# summary_table.csv         ← numbers for your paper Table I
# summary_table.tex         ← LaTeX-ready
```

### Step 7 — If Something Goes Wrong

```bash
# Remove all experiment containers:
docker rm -f $(docker ps -aq --filter "name=swiftbot")

# Clear Redis:
redis-cli FLUSHDB

# Check CRIU still works:
docker run -d --name test --security-opt seccomp:unconfined alpine sleep 99
docker checkpoint create --checkpoint-dir=/tmp/t test c1
docker rm -f test

# Check GPU:
nvidia-smi

# Restart Redis:
sudo systemctl restart redis-server
```

---

## VERIFICATION CHECKLIST

After building all files, run these checks before starting any experiment:

```bash
# 1. All files exist
find ~/swiftbot_rl -name "*.py" | sort

# 2. Docker image built
docker images | grep swiftbot-robot

# 3. CRIU works
criu --version && echo "CRIU OK"

# 4. Redis running
redis-cli ping

# 5. GPU visible
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 6. Imports work inside container
docker run --rm swiftbot-robot:latest python3 -c \
  "import flwr, torch, redis, psutil; print('Container imports OK')"

# 7. Results directories exist
ls ~/swiftbot_rl/dht_frl/results/
ls ~/swiftbot_rl/criu_cold/results/
ls ~/swiftbot_rl/criu_warm/results/
```

All 7 checks must pass before starting experiments.
