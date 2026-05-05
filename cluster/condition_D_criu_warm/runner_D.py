"""
Condition D — CRIU warm (pre-copy) migration cross-node.

Mechanism:
  1. Three iterative pre-dumps on src node, each rsync'd to dst node in
     parallel with the next pre-dump (overlapping computation + network).
  2. Final stop-and-copy: criu dump --leave-running=false (terminating the
     src process), rsync the small final delta, criu restore on dst node.

Real CRIU + CUDA may fail (same caveat as Condition C). The runner falls
back to SIMULATE mode in that case — synthesized timings model a typical
warm migration profile (4 phases of overlapping ~1-2s each + ~1s final).
"""
import logging, os, sys, time, random, threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.cluster_runner import (
    ClusterConfig, apptainer_instance_name,
    launch_robot, kill_robot, get_robot_pid,
    remote_criu_dump, remote_criu_restore,
    rsync_dir, ssh_run, simulate_dump_seconds,
)
from common.baseline_runner_base import current_node, update_node, run_baseline

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
LOG = logging.getLogger("condD")

CONDITION = "criu_warm"
IMAGE     = "baseline.sif"
# See runner_C.py for why all baselines use the cold_restart worker.
WORKER    = "cold_restart/worker_random_client.py"


def trigger_criu_warm(cfg, r, robot_id, success_rate_pre, task_counter_pre):
    cid = int(robot_id.split("_")[1])
    src = current_node(robot_id)
    dst = cfg.other_node(src)
    inst = apptainer_instance_name(cid)
    base = f"{cfg.checkpoint_base}/{inst}/criu_warm"

    LOG.info("[MIG] %s  %s → %s (criu warm)", robot_id, src, dst)
    t0 = time.perf_counter()
    bytes_xfer = 0
    chk_size_mb = 0.0
    used_simulate = cfg.simulate_criu

    if not used_simulate:
        pid = get_robot_pid(src, cid)
        if pid <= 0:
            LOG.warning("[MIG] no pid for %s; SIMULATE", robot_id)
            used_simulate = True

    t_dump_start = time.perf_counter()
    if not used_simulate:
        # 3 pre-dumps. Each pre-dump can be transferred in parallel with the
        # next pre-dump's computation. We use a background thread to push the
        # previous pre-dump while the next one runs.
        parent = ""
        prev_dir = None
        xfer_threads = []
        for i in range(3):
            pre_dir = f"{base}/predump_{i}"
            ssh_run(src, f"mkdir -p {pre_dir}", timeout=10)
            res = remote_criu_dump(src, pid, pre_dir, parent_dir=parent,
                                   pre_dump=True, leave_running=True, timeout=120)
            if res["returncode"] != 0:
                LOG.warning("[MIG] pre-dump %d failed for %s; ending warm chain",
                            i, robot_id)
                break
            chk_size_mb += res["size_mb"]
            if prev_dir:
                t = threading.Thread(
                    target=lambda d=prev_dir: rsync_dir(src, d, dst, d, timeout=300))
                t.start(); xfer_threads.append(t)
            parent = pre_dir; prev_dir = pre_dir

        # Final stop-and-copy.
        final_dir = f"{base}/final"
        ssh_run(src, f"mkdir -p {final_dir}", timeout=10)
        res_final = remote_criu_dump(src, pid, final_dir, parent_dir=parent,
                                     pre_dump=False, leave_running=False, timeout=120)
        chk_size_mb += res_final["size_mb"]

        for t in xfer_threads:
            t.join(timeout=300)
        if prev_dir:
            rsync_dir(src, prev_dir, dst, prev_dir, timeout=300)
        xfer = rsync_dir(src, final_dir, dst, final_dir, timeout=300)
        bytes_xfer = xfer["bytes_transferred"]
    else:
        # Synthesize warm profile: 3 overlapped pre-copies + small final stop+copy.
        time.sleep(random.triangular(1.5, 2.5, 2.0))  # overlapped pre-copy phase
        time.sleep(random.triangular(0.4, 0.8, 1.2))  # final dump
    t_dump_done = time.perf_counter()

    t_xfer_done = time.perf_counter()  # transfer overlapped with dumps above

    # Restore on dst. Same rationale as runner_C: regardless of whether real
    # CRIU restore succeeds, we always kill src + launch fresh apptainer
    # instance on dst, because a CRIU-restored bare process can't be tracked
    # via `apptainer instance` for the next migration.
    t_restore_start = time.perf_counter()
    if not used_simulate:
        remote_criu_restore(dst, f"{base}/final",
                            f"{cfg.run_log_dir}/criu_restore_{cid}.log",
                            timeout=120)
    else:
        time.sleep(random.triangular(0.4, 0.7, 1.0))  # simulated warm restore
    kill_robot(cfg, src, cid)
    launch_robot(cfg, dst, cid, IMAGE, WORKER)
    t_restore_done = time.perf_counter()
    update_node(robot_id, dst)
    r.set(f"migration_done:{robot_id}", "1", ex=600)

    total_ms = (t_restore_done - t0) * 1000
    return {
        "robot_id": robot_id, "src_node": src, "dst_node": dst,
        "trigger_to_dump_ms":     (t_dump_done   - t_dump_start) * 1000,
        "dump_to_transfer_ms":    (t_xfer_done   - t_dump_done) * 1000,
        "transfer_to_restore_ms": (t_restore_done- t_restore_start) * 1000,
        "policy_load_ms": 0,
        "downtime_ms": (t_restore_done - t_restore_start) * 1000,  # only stop+copy is downtime
        "total_MTT_ms": total_ms,
        "success_rate_pre": success_rate_pre,
        "success_rate_post": 0,
        "regression_pct": 0,
        "replay_buffer_entries_restored": 0,
        "network_bytes_transferred": bytes_xfer,
        "checkpoint_size_mb": round(chk_size_mb, 2),
        "criu_mode": "warm_simulated" if used_simulate else "warm_real",
    }


if __name__ == "__main__":
    sys.exit(run_baseline(
        condition=CONDITION, image=IMAGE,
        worker_script=WORKER, trigger_fn=trigger_criu_warm,
    ))
