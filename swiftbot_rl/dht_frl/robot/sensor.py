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
