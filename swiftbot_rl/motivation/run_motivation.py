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
     (same `real_criu_dump` helper criu_cold uses) — timed as dump_ms.
  5. Transfer (copytree) the image set — timed as transfer_ms.
  6. Simulated restore window + `cuda-checkpoint --toggle` to resume CUDA —
     timed as restore_ms. Same triangular distributions as Task 1.
  7. Derive downtime_ms / total_MTT_ms exactly as Task 1's criu_cold /
     criu_warm runners do, read the app-level policy size, and append a row
     to results/motivation.csv.

This reproduces Task 1's full latency breakdown (dump / transfer / restore /
downtime / MTT) on a single agent so Task 2's per-event latency is directly
comparable to the criu_cold and criu_warm migration_events.csv columns.

Run as root (CRIU + cuda-checkpoint need it):
  # cold (default):
  sudo CRIU_BIN=criu /home/simon/miniconda3/envs/swiftbot/bin/python \
       swiftbot_rl/motivation/run_motivation.py --repeat 15
  # warm (pre-copy):
  sudo CRIU_BIN=criu /home/simon/miniconda3/envs/swiftbot/bin/python \
       swiftbot_rl/motivation/run_motivation.py --warm --repeat 15
"""
import argparse, csv, json, os, random, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
from metrics_collector import (  # noqa: E402
    real_criu_dump, get_container_pid, cuda_checkpoint_toggle,
)

IMAGE            = "swiftbot-motivation:latest"
CONTAINER_PREFIX = "swiftbot-motivation"             # per-agent: <prefix>-<i>
CHECKPOINT_BASE  = "/tmp/swiftbot_motivation_vol"    # per-agent: <base>/agent_<i>
CRIU_OUT_BASE    = "/tmp/swiftbot_motivation_criu"   # per-agent: <base>/agent_<i>
RESULTS_DIR      = os.path.join(HERE, "results")
CSV_PATH         = os.path.join(RESULTS_DIR, "motivation.csv")


def _container_name(i: int) -> str:
    return f"{CONTAINER_PREFIX}-{i}"


def _agent_vol(i: int) -> str:
    return os.path.join(CHECKPOINT_BASE, f"agent_{i}")


def _agent_criu_dir(i: int) -> str:
    return os.path.join(CRIU_OUT_BASE, f"agent_{i}")


def docker_run_container(i: int, steps: int, batch: int,
                         keep_training: bool) -> str:
    """Start agent container i with its own /checkpoints volume."""
    name = _container_name(i)
    # Tear down any previous run.
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True)
    vol = _agent_vol(i)
    os.makedirs(vol, exist_ok=True)
    # Wipe stale ready marker / policy from a previous run.
    for f in ("hopper_ready.json", "hopper_policy.pt"):
        p = os.path.join(vol, f)
        if os.path.exists(p):
            os.remove(p)

    cmd = [
        "docker", "run", "-d", "--name", name,
        "--shm-size=4g",
        "-e", "PYTHONUNBUFFERED=1",
        "-e", "NVIDIA_VISIBLE_DEVICES=all",
        "--gpus", "all",
        "-v", f"{vol}:/checkpoints",
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
    if keep_training:
        cmd.append("--keep-training")
    print(f"[motivation] docker run agent {i} -> {name}", flush=True)
    rr = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if rr.returncode != 0:
        raise RuntimeError(f"docker run failed for agent {i}: {rr.stderr}")
    return name


def wait_ready(i: int, timeout: int = 900) -> dict:
    """Poll for agent i's ready marker on its shared volume."""
    name = _container_name(i)
    marker = os.path.join(_agent_vol(i), "hopper_ready.json")
    deadline = time.time() + timeout
    last_log = 0.0
    while time.time() < deadline:
        if os.path.exists(marker):
            with open(marker) as fh:
                return json.load(fh)
        if time.time() - last_log > 15:
            tail = subprocess.run(
                ["docker", "logs", "--tail", "3", name],
                capture_output=True, text=True,
            )
            print(f"[motivation] agent {i}: waiting for ready marker ... "
                  f"last log:\n{tail.stdout}{tail.stderr}", flush=True)
            last_log = time.time()
        time.sleep(2.0)
    raise TimeoutError(f"agent {i} did not write ready marker in {timeout}s")


def _append_row(csv_path: str, row: dict) -> None:
    """Append a result row. If an existing CSV has the older (pre-latency)
    schema, rotate it aside so we start a clean file with the full columns."""
    write_header = not os.path.exists(csv_path)
    if not write_header:
        with open(csv_path) as fh:
            existing_header = fh.readline().strip().split(",")
        if existing_header != list(row.keys()):
            bak = csv_path + ".old_schema"
            os.replace(csv_path, bak)
            print(f"[motivation] old-schema CSV rotated -> {bak}", flush=True)
            write_header = True
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _dir_size_mb(path: str) -> float:
    total = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(path) for f in files
    )
    return total / (1024 * 1024)


# Simulated restore windows — identical to Task 1 (criu_cold/criu_warm
# runners). criu 3.16.1 + runc cannot restore a CUDA dump on this hardware, so
# Task 1 samples a published restore latency instead of timing a real restore.
# Task 2 uses the SAME triangular distributions so downtime/MTT are directly
# comparable to Task 1. (min, mode, max) in ms.
COLD_RESTORE_MS = (600, 1000, 1400)
WARM_RESTORE_MS = (200, 330, 500)


def do_criu_cold(host_pid: int) -> dict:
    """Cold dump + transfer + simulated restore — mirrors
    trigger_criu_cold_migration in criu_cold/criu_cold_runner.py.

    Returns the same latency breakdown Task 1 records:
      size_mb, dump_ms, transfer_ms, restore_ms, downtime_ms, total_MTT_ms.
    """
    if os.path.exists(CRIU_OUT_DIR):
        shutil.rmtree(CRIU_OUT_DIR)
    os.makedirs(CRIU_OUT_DIR, exist_ok=True)
    chk_dst = CRIU_OUT_DIR + "_dest"
    if os.path.exists(chk_dst):
        shutil.rmtree(chk_dst)
    os.makedirs(chk_dst, exist_ok=True)

    t_trigger = time.perf_counter()

    if not cuda_checkpoint_toggle(host_pid):
        print("[motivation] WARNING cuda-checkpoint suspend failed.", flush=True)

    # Step 1: single full dump (real timing).
    t_dump = time.perf_counter()
    res = real_criu_dump(host_pid, CRIU_OUT_DIR, parent_dir="",
                         pre_dump=False, leave_running=True, timeout=180)
    dump_ms = (time.perf_counter() - t_dump) * 1000

    # Step 2: transfer — sequential, must complete the dump first (real timing).
    t_xfer = time.perf_counter()
    shutil.copytree(CRIU_OUT_DIR, os.path.join(chk_dst, "criu_cold"),
                    dirs_exist_ok=True)
    transfer_ms = (time.perf_counter() - t_xfer) * 1000

    # Step 3: simulated restore (cold: 600-1400ms). Resume CUDA inside the
    # restore window, exactly as Task 1 does.
    t_restore = time.perf_counter()
    time.sleep(random.triangular(*COLD_RESTORE_MS) / 1000.0)
    if not cuda_checkpoint_toggle(host_pid):
        print("[motivation] WARNING cuda-checkpoint resume failed.", flush=True)
    restore_ms = (time.perf_counter() - t_restore) * 1000

    total_MTT_ms = (time.perf_counter() - t_trigger) * 1000

    if res["returncode"] != 0:
        print(f"[motivation] WARNING criu rc={res['returncode']}: "
              f"{res['stderr'][:400]}", flush=True)

    return {
        "size_mb":      res["size_mb"],
        "dump_ms":      dump_ms,
        "transfer_ms":  transfer_ms,
        "restore_ms":   restore_ms,
        # Cold: the container is fully stopped for the whole event, so
        # downtime == MTT (same as criu_cold_runner.py).
        "downtime_ms":  total_MTT_ms,
        "total_MTT_ms": total_MTT_ms,
        "returncode":   res["returncode"],
    }


def do_criu_warm(host_pid: int, n_predumps: int = 3) -> dict:
    """N pre-dumps + final delta dump + transfer + simulated restore — mirrors
    trigger_criu_warm_migration in criu_warm/criu_warm_runner.py.

    Total size = sum of all pre-dump dirs + final dir (same as Task 1).
    Downtime counts only the final dump + restore — pre-copy keeps the app
    live during pre-dumps, again matching Task 1.
    """
    warm_root = CRIU_OUT_DIR + "_warm"
    if os.path.exists(warm_root):
        shutil.rmtree(warm_root)
    os.makedirs(warm_root, exist_ok=True)
    chk_dst = CRIU_OUT_DIR + "_warm_dest"
    if os.path.exists(chk_dst):
        shutil.rmtree(chk_dst)
    os.makedirs(chk_dst, exist_ok=True)

    t_trigger = time.perf_counter()

    if not cuda_checkpoint_toggle(host_pid):
        print("[motivation] WARNING cuda-checkpoint suspend failed.", flush=True)

    parent = ""
    for i in range(n_predumps):
        predump_dir = os.path.join(warm_root, f"predump_{i}")
        res = real_criu_dump(host_pid, predump_dir, parent_dir=parent,
                             pre_dump=True, leave_running=True, timeout=60)
        if res["returncode"] != 0:
            print(f"[motivation] pre-dump {i} failed: {res['stderr'][:200]}",
                  flush=True)
            parent = ""
            break
        parent = predump_dir
        print(f"[motivation] pre-dump {i} done "
              f"({_dir_size_mb(predump_dir):.1f} MB)", flush=True)
        time.sleep(0.05)

    # Final delta dump — only dirty pages since the last pre-dump (real timing).
    t_dump = time.perf_counter()
    final_dir = os.path.join(warm_root, "final")
    res_final = real_criu_dump(host_pid, final_dir, parent_dir=parent,
                               pre_dump=False, leave_running=True, timeout=180)
    dump_ms = (time.perf_counter() - t_dump) * 1000

    # Transfer the whole image set (real timing).
    t_xfer = time.perf_counter()
    shutil.copytree(warm_root, os.path.join(chk_dst, "criu_warm"),
                    dirs_exist_ok=True)
    transfer_ms = (time.perf_counter() - t_xfer) * 1000

    # Simulated restore (warm: 200-500ms). Resume CUDA inside the window.
    t_restore = time.perf_counter()
    time.sleep(random.triangular(*WARM_RESTORE_MS) / 1000.0)
    if not cuda_checkpoint_toggle(host_pid):
        print("[motivation] WARNING cuda-checkpoint resume failed.", flush=True)
    restore_ms = (time.perf_counter() - t_restore) * 1000

    total_MTT_ms = (time.perf_counter() - t_trigger) * 1000

    if res_final["returncode"] != 0:
        print(f"[motivation] WARNING final dump rc={res_final['returncode']}: "
              f"{res_final['stderr'][:400]}", flush=True)

    total_mb = _dir_size_mb(warm_root)
    print(f"[motivation] warm total size = {total_mb:.2f} MB "
          f"(all pre-dump dirs + final)", flush=True)
    return {
        "size_mb":      total_mb,
        "dump_ms":      dump_ms,
        "transfer_ms":  transfer_ms,
        "restore_ms":   restore_ms,
        # Warm: only the final stop-and-copy dump + restore freeze the app.
        "downtime_ms":  dump_ms + restore_ms,
        "total_MTT_ms": total_MTT_ms,
        "returncode":   res_final["returncode"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--job",   default="d4rl_hopper_medium_v2")
    ap.add_argument("--warm",  action="store_true",
                    help="Run CRIU warm (pre-copy) instead of cold dump")
    ap.add_argument("--predumps", type=int, default=3,
                    help="Number of pre-dump iterations for --warm (default 3)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="Number of migration events to measure on the same "
                         "running agent (default 1). Use e.g. 10-20 to collect "
                         "a latency distribution comparable to Task 1.")
    args = ap.parse_args()

    if os.geteuid() != 0:
        print("[motivation] WARNING: not running as root — cuda-checkpoint "
              "and criu dump will likely fail.", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        docker_run_container(args.steps, args.batch)
        ready = wait_ready()
        policy_path_host = os.path.join(CHECKPOINT_VOL, "hopper_policy.pt")
        policy_kb = os.path.getsize(policy_path_host) / 1024.0
        print(f"[motivation] agent ready (container-pid={ready['pid']}); "
              f"policy state_dict = {policy_kb:.1f} KB", flush=True)

        host_pid = get_container_pid(CONTAINER_NAME)
        if host_pid <= 0:
            raise RuntimeError("could not resolve container host PID")
        print(f"[motivation] container host PID = {host_pid}", flush=True)

        time.sleep(3.0)  # let RSS settle after torch.save

        criu_mode = "warm" if args.warm else "cold"
        for event in range(args.repeat):
            if args.repeat > 1:
                print(f"[motivation] ---- migration event "
                      f"{event + 1}/{args.repeat} ----", flush=True)
            if args.warm:
                m = do_criu_warm(host_pid, args.predumps)
            else:
                m = do_criu_cold(host_pid)

            criu_mb = m["size_mb"]
            ratio = (criu_mb * 1024) / policy_kb if policy_kb else 0.0
            row = {
                "job":                    args.job,
                "criu_mode":              criu_mode,
                "migration_event_id":     event,
                "criu_size_mb":           round(criu_mb, 2),
                "app_policy_size_kb":     round(policy_kb, 2),
                "ratio_criu_over_app":    round(ratio, 1),
                "criu_returncode":        m["returncode"],
                # Latency breakdown — same column names Task 1 writes to
                # <condition>/results/migration_events.csv, so the two CSVs
                # line up for a direct comparison.
                "trigger_to_dump_ms":     round(m["dump_ms"], 0),
                "dump_to_transfer_ms":    round(m["transfer_ms"], 0),
                "transfer_to_restore_ms": round(m["restore_ms"], 0),
                "dump_ms":                round(m["dump_ms"], 0),
                "downtime_ms":            round(m["downtime_ms"], 0),
                "total_MTT_ms":           round(m["total_MTT_ms"], 0),
                "container_host_pid":     host_pid,
                "steps":                  args.steps,
                "batch":                  args.batch,
                "timestamp":              time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            _append_row(CSV_PATH, row)

            print("[motivation] ============ RESULT ============", flush=True)
            for k, v in row.items():
                print(f"  {k:24s} {v}", flush=True)
            print(f"[motivation] appended row -> {CSV_PATH}", flush=True)
            # Let the agent run a moment so the next event captures fresh state.
            if event + 1 < args.repeat:
                time.sleep(2.0)
    finally:
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME],
                       capture_output=True, text=True)


if __name__ == "__main__":
    main()
