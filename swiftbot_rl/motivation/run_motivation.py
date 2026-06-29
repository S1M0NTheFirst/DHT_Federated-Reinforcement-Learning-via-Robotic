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
  5. Transfer modeled as image_size / bandwidth — recorded as transfer_ms.
  6. Simulated restore window + `cuda-checkpoint --toggle` to resume CUDA —
     timed as restore_ms. Same triangular distributions as Task 1.
  7. Derive downtime_ms / total_MTT_ms exactly as Task 1's criu_cold /
     criu_warm runners do, read the app-level policy size, and append a row
     to results/motivation.csv.

This reproduces Task 1's full latency breakdown (dump / transfer / restore /
downtime / MTT) on a single agent so Task 2's per-event latency is directly
comparable to the criu_cold and criu_warm migration_events.csv columns.

By default 8 agents run concurrently (--agents 8) with 5 forced migration
events each (--events 5) — matching Task 1's 8 robots so each dump is measured
under the same GPU/CPU/disk contention, not in isolation. Agents keep training
during the sweep (use --idle to disable).

Run as root (CRIU + cuda-checkpoint need it):
  # cold (default): 8 agents x 5 events = 40 rows
  sudo CRIU_BIN=criu /home/simon/miniconda3/envs/swiftbot/bin/python \
       swiftbot_rl/motivation/run_motivation.py
  # warm (pre-copy):
  sudo CRIU_BIN=criu /home/simon/miniconda3/envs/swiftbot/bin/python \
       swiftbot_rl/motivation/run_motivation.py --warm
  # quick smoke test (1 agent, 1 event):
  sudo CRIU_BIN=criu /home/simon/miniconda3/envs/swiftbot/bin/python \
       swiftbot_rl/motivation/run_motivation.py --agents 1 --events 1
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

# Default modeled transfer link speed (MB/s). 125 MB/s = 1 Gbps. The checkpoint
# is shipped to the destination node as a byte stream over the network; we model
# transfer = image_size / bandwidth instead of timing a file-by-file copytree.
# copytree of the many-small-file CRIU image measures filesystem syscall
# overhead + local-disk contention (which exploded to 70-180s/agent under 8
# concurrent agents), NOT transfer throughput. Modeling bytes/bandwidth is the
# standard live-migration approach, is contention-independent, and lets Task 1
# and Task 2 be compared on the SAME basis (evaluation/compare_tasks.py applies
# the identical formula to Task 1's recorded checkpoint sizes).
DEFAULT_BANDWIDTH_MBPS = 125.0


def _modeled_transfer_ms(size_mb: float, bandwidth_mbps: float) -> float:
    return (size_mb / bandwidth_mbps) * 1000.0 if bandwidth_mbps > 0 else 0.0


def do_criu_cold(host_pid: int, out_dir: str,
                 bandwidth_mbps: float = DEFAULT_BANDWIDTH_MBPS) -> dict:
    """Cold dump + modeled transfer + simulated restore — mirrors
    trigger_criu_cold_migration in criu_cold/criu_cold_runner.py.

    Returns the same latency breakdown Task 1 records:
      size_mb, dump_ms, transfer_ms, restore_ms, downtime_ms, total_MTT_ms.
    """
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    t_trigger = time.perf_counter()

    if not cuda_checkpoint_toggle(host_pid):
        print("[motivation] WARNING cuda-checkpoint suspend failed.", flush=True)

    # Step 1: single full dump (real timing).
    t_dump = time.perf_counter()
    res = real_criu_dump(host_pid, out_dir, parent_dir="",
                         pre_dump=False, leave_running=True, timeout=180)
    dump_ms = (time.perf_counter() - t_dump) * 1000
    size_mb = res["size_mb"]

    # Step 2: transfer — modeled as shipping the image over a fixed-bandwidth
    # link (size / bandwidth), not a local file copy. See DEFAULT_BANDWIDTH_MBPS.
    transfer_ms = _modeled_transfer_ms(size_mb, bandwidth_mbps)

    # Step 3: simulated restore (cold: 600-1400ms). Resume CUDA inside the
    # restore window, exactly as Task 1 does.
    t_restore = time.perf_counter()
    time.sleep(random.triangular(*COLD_RESTORE_MS) / 1000.0)
    if not cuda_checkpoint_toggle(host_pid):
        print("[motivation] WARNING cuda-checkpoint resume failed.", flush=True)
    restore_ms = (time.perf_counter() - t_restore) * 1000

    # MTT = real wall time (suspend + dump + restore) + modeled transfer.
    total_MTT_ms = (time.perf_counter() - t_trigger) * 1000 + transfer_ms

    if res["returncode"] != 0:
        print(f"[motivation] WARNING criu rc={res['returncode']}: "
              f"{res['stderr'][:400]}", flush=True)

    return {
        "size_mb":      size_mb,
        "dump_ms":      dump_ms,
        "transfer_ms":  transfer_ms,
        "restore_ms":   restore_ms,
        # Cold: the container is fully stopped for the whole event, so
        # downtime == MTT (same as criu_cold_runner.py).
        "downtime_ms":  total_MTT_ms,
        "total_MTT_ms": total_MTT_ms,
        "returncode":   res["returncode"],
    }


def do_criu_warm(host_pid: int, out_dir: str, n_predumps: int = 3,
                 bandwidth_mbps: float = DEFAULT_BANDWIDTH_MBPS) -> dict:
    """N pre-dumps + final delta dump + modeled transfer + simulated restore —
    mirrors trigger_criu_warm_migration in criu_warm/criu_warm_runner.py.

    Total size = sum of all pre-dump dirs + final dir (same as Task 1).
    Downtime counts only the final dump + restore — pre-copy keeps the app
    live during pre-dumps, again matching Task 1.
    """
    warm_root = out_dir + "_warm"
    if os.path.exists(warm_root):
        shutil.rmtree(warm_root)
    os.makedirs(warm_root, exist_ok=True)

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

    total_mb = _dir_size_mb(warm_root)

    # Transfer — modeled as shipping the whole image set over a fixed-bandwidth
    # link (size / bandwidth), not a file-by-file copy. See DEFAULT_BANDWIDTH_MBPS.
    transfer_ms = _modeled_transfer_ms(total_mb, bandwidth_mbps)

    # Simulated restore (warm: 200-500ms). Resume CUDA inside the window.
    t_restore = time.perf_counter()
    time.sleep(random.triangular(*WARM_RESTORE_MS) / 1000.0)
    if not cuda_checkpoint_toggle(host_pid):
        print("[motivation] WARNING cuda-checkpoint resume failed.", flush=True)
    restore_ms = (time.perf_counter() - t_restore) * 1000

    # MTT = real wall time (suspend + pre-dumps + final dump + restore) +
    # modeled transfer.
    total_MTT_ms = (time.perf_counter() - t_trigger) * 1000 + transfer_ms

    if res_final["returncode"] != 0:
        print(f"[motivation] WARNING final dump rc={res_final['returncode']}: "
              f"{res_final['stderr'][:400]}", flush=True)

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


def _measure_event(args, i: int, host_pid: int, policy_kb: float,
                   event: int, criu_mode: str) -> None:
    """Dump agent i once, derive the latency breakdown, append a CSV row."""
    out_dir = _agent_criu_dir(i)
    if args.warm:
        m = do_criu_warm(host_pid, out_dir, args.predumps, args.bandwidth_mbps)
    else:
        m = do_criu_cold(host_pid, out_dir, args.bandwidth_mbps)

    criu_mb = m["size_mb"]
    ratio = (criu_mb * 1024) / policy_kb if policy_kb else 0.0
    row = {
        "job":                    args.job,
        "criu_mode":              criu_mode,
        "robot_id":               f"agent_{i}",
        "migration_event_id":     event,
        "criu_size_mb":           round(criu_mb, 2),
        "app_policy_size_kb":     round(policy_kb, 2),
        "ratio_criu_over_app":    round(ratio, 1),
        "transfer_bandwidth_mbps": args.bandwidth_mbps,
        "criu_returncode":        m["returncode"],
        # Latency breakdown — same column names Task 1 writes to
        # <condition>/results/migration_events.csv, so the two CSVs line up
        # for a direct comparison.
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
    print(f"[motivation] agent_{i} event {event}: "
          f"dump={row['dump_ms']:.0f}ms downtime={row['downtime_ms']:.0f}ms "
          f"MTT={row['total_MTT_ms']:.0f}ms size={row['criu_size_mb']}MB "
          f"rc={row['criu_returncode']} -> {CSV_PATH}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=8,
                    help="Number of concurrent hopper agents (default 8 — "
                         "matches Task 1's 8 robots so dumps run under the "
                         "same GPU/CPU/disk contention).")
    ap.add_argument("--events", type=int, default=5,
                    help="Forced migration events per agent (default 5 — "
                         "matches Task 1). agents*events total CSV rows.")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--job",   default="d4rl_hopper_medium_v2")
    ap.add_argument("--warm",  action="store_true",
                    help="Run CRIU warm (pre-copy) instead of cold dump")
    ap.add_argument("--predumps", type=int, default=3,
                    help="Number of pre-dump iterations for --warm (default 3)")
    ap.add_argument("--bandwidth-mbps", type=float,
                    default=DEFAULT_BANDWIDTH_MBPS,
                    help=f"Modeled transfer link speed in MB/s "
                         f"(default {DEFAULT_BANDWIDTH_MBPS:.0f} = 1 Gbps). "
                         f"transfer_ms = image_size / bandwidth. The same value "
                         f"must be passed to compare_tasks.py so Task 1 and "
                         f"Task 2 use an identical transfer model.")
    ap.add_argument("--idle", action="store_true",
                    help="Let agents idle after warm-up instead of training "
                         "continuously. Default is continuous training so the "
                         "GPU stays loaded during dumps, like Task 1.")
    args = ap.parse_args()

    if os.geteuid() != 0:
        print("[motivation] WARNING: not running as root — cuda-checkpoint "
              "and criu dump will likely fail.", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    keep_training = not args.idle
    criu_mode = "warm" if args.warm else "cold"
    names = [_container_name(i) for i in range(args.agents)]

    try:
        # 1. Launch all agents concurrently so they share the GPU/CPU/disk.
        for i in range(args.agents):
            docker_run_container(i, args.steps, args.batch, keep_training)

        # 2. Wait for every agent to finish warm-up + write its policy, then
        #    resolve each container's host PID.
        policy_kb = {}
        host_pid  = {}
        for i in range(args.agents):
            ready = wait_ready(i)
            policy_kb[i] = os.path.getsize(
                os.path.join(_agent_vol(i), "hopper_policy.pt")) / 1024.0
            pid = get_container_pid(_container_name(i))
            if pid <= 0:
                raise RuntimeError(f"could not resolve host PID for agent {i}")
            host_pid[i] = pid
            print(f"[motivation] agent_{i} ready: host_pid={pid} "
                  f"policy={policy_kb[i]:.1f} KB", flush=True)

        time.sleep(3.0)  # let RSS settle after the torch.save burst

        # 3. Forced migrations: per event round, dump every agent in turn.
        #    All agents stay resident (and training, unless --idle) throughout,
        #    so each dump is measured under the same contention as Task 1.
        for event in range(args.events):
            print(f"[motivation] ======== migration round "
                  f"{event + 1}/{args.events} ({criu_mode}) ========",
                  flush=True)
            for i in range(args.agents):
                try:
                    _measure_event(args, i, host_pid[i], policy_kb[i],
                                   event, criu_mode)
                except Exception as e:
                    # cuda-checkpoint can be brittle with concurrent CUDA
                    # processes; don't let one bad dump abort the whole sweep.
                    print(f"[motivation] ERROR agent_{i} event {event}: {e}",
                          flush=True)
            time.sleep(1.0)

        print(f"[motivation] done — {args.agents} agents x {args.events} "
              f"events appended to {CSV_PATH}", flush=True)
    finally:
        for name in names:
            subprocess.run(["docker", "rm", "-f", name],
                           capture_output=True, text=True)


if __name__ == "__main__":
    main()
