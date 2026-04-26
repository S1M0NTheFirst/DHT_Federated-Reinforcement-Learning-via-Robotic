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
            # Tight deadline (10% slack). Under contention from other containers
            # the per-iteration loop overhead pushes some accepted tasks past it,
            # giving PPO a real failure signal to learn from.
            "deadline_ms": round(duration_s * 1000 * 1.1, 1),
        }

    def execute(self, task_spec: dict, bid: float = 1.0,
                bid_threshold: float = 0.5) -> dict:
        """
        Run the workload, gated by the policy's bid value.

        bid < bid_threshold → robot DECLINES (status="declined", no work done).
        bid >= bid_threshold → robot accepts and executes the task.
                               success requires latency <= deadline_ms.

        This makes the bid a load-bearing decision: a random policy declines
        ~50% of tasks, while a trained PPO learns when to accept based on
        load. Without this gate, success_rate is trivially 1.0 for everyone.
        """
        # 1. Bid gating — the robot has to commit before doing the work
        if bid < bid_threshold:
            return {
                "status": "declined",
                "latency_ms": 0.0,
                "deadline_ms": task_spec["deadline_ms"],
                "bid": float(bid),
            }

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
                "bid": float(bid),
            }
        except Exception as e:
            return {"status": "failed", "error": str(e),
                    "latency_ms": 0, "deadline_ms": task_spec["deadline_ms"],
                    "bid": float(bid)}

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
