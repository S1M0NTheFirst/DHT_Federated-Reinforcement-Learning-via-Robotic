"""
Motivation experiment orchestrator — mirrors criu_cold's per-event dump
recipe (`trigger_criu_cold_migration` in criu_cold/criu_cold_runner.py),
but on a single D4RL hopper-medium-v2 agent instead of the bidder workload.

Steps:
  1. Launch one `swiftbot-motivation:latest` container with `--gpus all`,
     same mount/security flags as criu_cold workers.
  2. Wait for the agent to (a) train a few steps, (b) torch.save its
     policy state_dict, and (c) drop a ready marker into the shared
     /checkpoints volume.
  3. `cuda-checkpoint --toggle` on the container PID (suspend CUDA).
  4. `criu dump --leave-running` with nvidia mounts as `--external`
     (same `real_criu_dump` helper criu_cold uses).
  5. `cuda-checkpoint --toggle` again (resume CUDA).
  6. Read the app-level policy size from the shared volume.
  7. Append both numbers to results/motivation.csv.

Run as root (CRIU + cuda-checkpoint need it):
  sudo CRIU_BIN=criu /home/simon/miniconda3/envs/swiftbot/bin/python \
       swiftbot_rl/motivation/run_motivation.py
"""
import argparse, csv, json, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
from metrics_collector import (  # noqa: E402
    real_criu_dump, get_container_pid, cuda_checkpoint_toggle,
)

IMAGE          = "swiftbot-motivation:latest"
CONTAINER_NAME = "swiftbot-motivation-0"
CHECKPOINT_VOL = "/tmp/swiftbot_motivation_vol"      # bind-mounted -> /checkpoints
CRIU_OUT_DIR   = "/tmp/swiftbot_motivation_criu"     # CRIU image output (host)
RESULTS_DIR    = os.path.join(HERE, "results")
CSV_PATH       = os.path.join(RESULTS_DIR, "motivation.csv")


def docker_run_container(steps: int, batch: int) -> str:
    """Start the agent container. Returns container name."""
    # Tear down any previous run.
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME],
                   capture_output=True, text=True)
    os.makedirs(CHECKPOINT_VOL, exist_ok=True)
    # Wipe stale ready marker / policy from a previous run.
    for f in ("hopper_ready.json", "hopper_policy.pt"):
        p = os.path.join(CHECKPOINT_VOL, f)
        if os.path.exists(p):
            os.remove(p)

    cmd = [
        "docker", "run", "-d", "--name", CONTAINER_NAME,
        "--shm-size=4g",
        "-e", "PYTHONUNBUFFERED=1",
        "-e", "NVIDIA_VISIBLE_DEVICES=all",
        "--gpus", "all",
        "-v", f"{CHECKPOINT_VOL}:/checkpoints",
        # Bind-mount the user's minari dataset cache so we don't re-download
        # inside the container. Falls back to in-container download if absent.
        "-v", f"{os.path.expanduser('~/.minari')}:/root/.minari",
        "--security-opt", "seccomp=unconfined",
        "--network", "host",
        IMAGE,
        "python3", "/app/hopper_agent.py",
        "--steps",      str(steps),
        "--batch",      str(batch),
        "--save-path",  "/checkpoints/hopper_policy.pt",
        "--ready-path", "/checkpoints/hopper_ready.json",
    ]
    print(f"[motivation] docker run: {' '.join(cmd)}", flush=True)
    rr = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if rr.returncode != 0:
        raise RuntimeError(f"docker run failed: {rr.stderr}")
    return CONTAINER_NAME


def wait_ready(timeout: int = 900) -> dict:
    """Poll for the agent's ready marker on the shared volume."""
    marker = os.path.join(CHECKPOINT_VOL, "hopper_ready.json")
    deadline = time.time() + timeout
    last_log = 0.0
    while time.time() < deadline:
        if os.path.exists(marker):
            with open(marker) as fh:
                return json.load(fh)
        if time.time() - last_log > 15:
            tail = subprocess.run(
                ["docker", "logs", "--tail", "5", CONTAINER_NAME],
                capture_output=True, text=True,
            )
            print(f"[motivation] waiting for ready marker ... "
                  f"last container log:\n{tail.stdout}{tail.stderr}",
                  flush=True)
            last_log = time.time()
        time.sleep(2.0)
    raise TimeoutError(f"agent did not write ready marker in {timeout}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--job",   default="d4rl_hopper_medium_v2")
    args = ap.parse_args()

    if os.geteuid() != 0:
        print("[motivation] WARNING: not running as root — cuda-checkpoint "
              "and criu dump will likely fail.", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(CRIU_OUT_DIR):
        shutil.rmtree(CRIU_OUT_DIR)
    os.makedirs(CRIU_OUT_DIR, exist_ok=True)

    try:
        docker_run_container(args.steps, args.batch)
        ready = wait_ready()
        policy_path_host = os.path.join(CHECKPOINT_VOL, "hopper_policy.pt")
        policy_kb = os.path.getsize(policy_path_host) / 1024.0
        print(f"[motivation] agent ready (container-pid={ready['pid']}); "
              f"policy state_dict = {policy_kb:.1f} KB", flush=True)

        # Get the *host* PID — ready['pid'] is the in-container PID.
        host_pid = get_container_pid(CONTAINER_NAME)
        if host_pid <= 0:
            raise RuntimeError("could not resolve container host PID")
        print(f"[motivation] container host PID = {host_pid}", flush=True)

        # Let RSS settle after torch.save spikes.
        time.sleep(3.0)

        # Mirror trigger_criu_cold_migration: suspend CUDA, dump, resume CUDA.
        if not cuda_checkpoint_toggle(host_pid):
            print("[motivation] WARNING cuda-checkpoint suspend failed — "
                  "criu dump will likely fail with CUDA mapping errors.",
                  flush=True)

        t0 = time.perf_counter()
        res = real_criu_dump(host_pid, CRIU_OUT_DIR, parent_dir="",
                             pre_dump=False, leave_running=True, timeout=180)
        dump_ms = (time.perf_counter() - t0) * 1000
        criu_mb = res["size_mb"]
        if res["returncode"] != 0:
            print(f"[motivation] WARNING criu rc={res['returncode']}: "
                  f"{res['stderr'][:400]}", flush=True)

        if not cuda_checkpoint_toggle(host_pid):
            print("[motivation] WARNING cuda-checkpoint resume failed.",
                  flush=True)

        ratio = (criu_mb * 1024) / policy_kb if policy_kb else 0.0
        row = {
            "job":                   args.job,
            "criu_cold_size_mb":     round(criu_mb, 2),
            "app_policy_size_kb":    round(policy_kb, 2),
            "ratio_criu_over_app":   round(ratio, 1),
            "criu_returncode":       res["returncode"],
            "dump_ms":               round(dump_ms, 0),
            "container_host_pid":    host_pid,
            "steps":                 args.steps,
            "batch":                 args.batch,
            "timestamp":             time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        write_header = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

        print("[motivation] ============ RESULT ============", flush=True)
        for k, v in row.items():
            print(f"  {k:22s} {v}", flush=True)
        print(f"[motivation] appended row -> {CSV_PATH}", flush=True)
    finally:
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME],
                       capture_output=True, text=True)


if __name__ == "__main__":
    main()
