"""
Shared metrics collector — used by all three experiment conditions.
Writes migration event metrics to CSV files in the condition's results/ folder.
"""
import csv
import os
import time
import json
import threading
import subprocess
import logging
import psutil
import redis

_log = logging.getLogger(__name__)


def get_container_pid(container_name: str) -> int:
    """Host PID of a Docker container's main process.

    Needed by `cuda-checkpoint --toggle --pid <pid>` which runs from the host
    but operates on the container's CUDA contexts.
    Returns 0 on failure so callers can skip cleanly.
    """
    try:
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Pid}}", container_name],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip()) if out.returncode == 0 else 0
    except Exception as e:
        _log.warning(f"get_container_pid({container_name}) failed: {e}")
        return 0


def _discover_external_mounts(pid: int) -> list:
    """Return mount points that CRIU will refuse to dump as 'unreachable
    sharing' or 'missing proper root mount'.

    This covers two cases:
    1. Bind mounts with `master:N` propagation (common in NVIDIA runtime).
    2. Bind mounts of individual files or subdirectories (like Docker's
       /etc/hosts, /etc/resolv.conf) where the mount root is not '/'.
    """
    mounts = []
    try:
        with open(f"/proc/{pid}/mountinfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 7:
                    continue
                
                root = parts[3]
                mount_point = parts[4]
                
                # Check for master:N propagation
                has_master = False
                for i in range(6, len(parts)):
                    if parts[i] == "-":
                        break
                    if parts[i].startswith("master:"):
                        has_master = True
                        break
                
                if has_master:
                    mounts.append(mount_point)
                    continue

                # Check for bind mounts of subpaths (e.g. /etc/hosts)
                # If root is not "/" and it's not a standard pseudo-fs submount
                if root != "/" and not mount_point.startswith(("/proc", "/sys", "/dev")):
                    mounts.append(mount_point)

    except FileNotFoundError:
        _log.warning(f"_discover_external_mounts: /proc/{pid}/mountinfo not found")
    except PermissionError:
        _log.warning(f"_discover_external_mounts: cannot read /proc/{pid}/mountinfo")
    except Exception as e:
        _log.warning(f"_discover_external_mounts({pid}) failed: {e}")
    return sorted(set(mounts))


# Old name kept as alias for any external caller; same return shape.
_discover_nvidia_mounts = _discover_external_mounts


def real_criu_dump(pid: int, images_dir: str, parent_dir: str = "",
                    leave_running: bool = True, pre_dump: bool = False,
                    timeout: int = 120) -> dict:
    """Direct `criu dump` (or `criu pre-dump`) on a host PID.
    """
    os.makedirs(images_dir, exist_ok=True)
    subcmd = "pre-dump" if pre_dump else "dump"
    
    # Create a relative 'parent' symlink manually. CRIU natively looks for 
    # a 'parent' symlink to find the previous image. By making it relative, 
    # we ensure the entire checkpoint folder is portable and can be copied
    # by shutil.copytree(symlinks=True) without pointing back to the source.
    if parent_dir and os.path.exists(parent_dir):
        parent_symlink = os.path.join(images_dir, "parent")
        if os.path.lexists(parent_symlink):
            os.remove(parent_symlink)
        rel_target = os.path.relpath(parent_dir, images_dir)
        os.symlink(rel_target, parent_symlink)

    # CRIU 3.x can't dump processes that hold CUDA mmaps. CRIU 4.0+ ships a
    # cuda plugin that talks to NVIDIA's cuda-checkpoint API and handles
    # those mappings transparently. If the user has built 4.0+ via
    # install_criu_cuda.sh they should set CRIU_BIN to the new binary.
    criu_bin = os.environ.get("CRIU_BIN", "criu")
    # Skip the internal sudo when the runner is already root. Some shell
    # environments propagate PR_SET_NO_NEW_PRIVS (snap, sandboxed shells,
    # bwrap), which makes `sudo -n` fail with "The 'no new privileges' flag
    # is set". Running the whole runner under sudo avoids that path
    # entirely; this branch keeps the cmd clean when that's the case.
    sudo_prefix = [] if os.geteuid() == 0 else ["sudo", "-n"]
    cmd = [
        *sudo_prefix, criu_bin, subcmd,
        "--tree", str(pid),
        "--images-dir", images_dir,
        "--tcp-established",
        "--shell-job",
        "--ext-unix-sk",
        "--manage-cgroups=soft",
    ]

    # Auto-load the cuda plugin when running 4.0+. CRIU walks `--libdir`
    # and dlopen's every .so it finds (cuda_plugin.so, amdgpu_plugin.so,
    # …). Default libdir is /usr/lib/criu/, but install_criu_cuda.sh puts
    # the plugin under /usr/local/lib/criu/ to avoid colliding with the
    # apt-installed CRIU 3.16.1's directory. Override via CRIU_LIBDIR.
    if os.environ.get("CRIU_USE_CUDA_PLUGIN", "0") == "1":
        libdir = os.environ.get("CRIU_LIBDIR", "/usr/local/lib/criu")
        cmd.extend(["--libdir", libdir])

    if parent_dir or pre_dump:
        cmd.append("--track-mem")

    if leave_running and not pre_dump:
        cmd.append("--leave-running")

    # Auto-declare each nvidia bind mount as an external resource.
    for i, mp in enumerate(_discover_nvidia_mounts(pid)):
        cmd.extend(["--external", f"mnt[{mp}]:nv{i}"])

    t0 = time.perf_counter()
    rr = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    dump_ms = (time.perf_counter() - t0) * 1000

    # Fix permissions so the runner (when not root) can read/transfer images.
    if os.geteuid() != 0:
        subprocess.run(["sudo", "-n", "chown", "-R",
                        f"{os.getuid()}:{os.getgid()}", images_dir])

    # Persist full stderr/stdout to a log file alongside the images. The
    # in-memory `stderr` field is truncated for log readability, but the
    # truncation has bitten us before — the real CRIU error often sits past
    # the cutoff behind a wall of harmless `Warn ...` lines.
    try:
        with open(os.path.join(images_dir, "criu.log"), "w") as fh:
            fh.write(f"# cmd: {' '.join(cmd)}\n# rc={rr.returncode}\n")
            fh.write("--- stdout ---\n"); fh.write(rr.stdout or "")
            fh.write("\n--- stderr ---\n"); fh.write(rr.stderr or "")
    except Exception:
        pass

    size_mb = 0.0
    if os.path.exists(images_dir):
        size_mb = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, files in os.walk(images_dir) for f in files
        ) / (1024 * 1024)

    if rr.returncode != 0:
        _log.error(f"criu {subcmd} pid={pid} failed (rc={rr.returncode}): "
                   f"{(rr.stderr or rr.stdout).strip()[:400]}")

    # A pre-dump with 0 modified pages may emit pagemap-*.img but NO pages-*.img.
    # The next dump in the chain then aborts with "No parent image found".
    # Treat such a dump as an invalid parent so the caller breaks the chain.
    valid_parent = True
    if rr.returncode == 0 and pre_dump:
        try:
            files = os.listdir(images_dir)
            has_pagemap = any(f.startswith("pagemap-") and f.endswith(".img") for f in files)
            has_pages   = any(f.startswith("pages-")   and f.endswith(".img") for f in files)
            valid_parent = has_pagemap and has_pages
        except FileNotFoundError:
            valid_parent = False

    # Pull the LAST chunk of stderr — CRIU prints harmless `Warn ...` lines
    # at the top, and the actual error message sits at the bottom.
    stderr_full = (rr.stderr or "").strip()
    stderr_tail = stderr_full[-1500:] if len(stderr_full) > 1500 else stderr_full
    return {
        "dump_ms": dump_ms,
        "returncode": rr.returncode,
        "stderr": stderr_tail,
        "stderr_full": stderr_full,
        "log_path": os.path.join(images_dir, "criu.log"),
        "size_mb": round(size_mb, 2),
        "valid_parent": valid_parent,
    }


def cuda_checkpoint_toggle(pid: int) -> bool:
    """Suspend or resume the CUDA state of a process.

    NVIDIA's cuda-checkpoint with --toggle: the first call releases all CUDA
    resources held by the process (so CRIU can dump it), the second call
    re-acquires them after restore. Requires:
      - cuda-checkpoint binary (path via $CUDA_CHECKPOINT_BIN, default
        `/usr/local/bin/cuda-checkpoint`)
      - NVIDIA driver R550+
      - root privileges (talks to NVIDIA driver, needs ptrace on target)

    Wrapping in `sudo -n` so the runner can stay as a normal user. Add to
    sudoers (visudo):
        simon ALL=(root) NOPASSWD: /usr/local/bin/cuda-checkpoint
    Returns True on success, False otherwise (caller logs and continues;
    CRIU will then fail loudly with the original error).
    """
    if pid <= 0:
        return False
    binary = os.environ.get("CUDA_CHECKPOINT_BIN", "/usr/local/bin/cuda-checkpoint")
    sudo_prefix = [] if os.geteuid() == 0 else ["sudo", "-n"]
    cmd = [*sudo_prefix, binary, "--toggle", "--pid", str(pid)]
    try:
        rr = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if rr.returncode != 0:
            err = (rr.stderr or rr.stdout).strip()[:300]
            _log.error(f"cuda-checkpoint --toggle --pid {pid} failed "
                       f"(rc={rr.returncode}): {err}")
            return False
        return True
    except FileNotFoundError:
        _log.error("sudo or cuda-checkpoint binary not found — install via "
                   "https://github.com/NVIDIA/cuda-checkpoint and add a "
                   "passwordless sudoers entry.")
        return False
    except Exception as e:
        _log.error(f"cuda-checkpoint toggle pid={pid} raised: {e}")
        return False

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
