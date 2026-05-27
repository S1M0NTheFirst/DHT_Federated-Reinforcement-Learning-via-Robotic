"""
Condition B — Apptainer instance stop+start with state-file replay.

Stand-in for "container-engine checkpoint" on apptainer (which has no native
checkpoint command). Mechanism:
  1. On migration request, the worker has already written a small state file
     to /checkpoints/<inst>/state.json containing task_counter + success_hist.
     (The cold_restart-style worker writes resume_counter to redis; we copy
     that here.)
  2. rsync the state from src node to dst node.
  3. Stop the apptainer instance on src node.
  4. Start a fresh apptainer instance on dst node, with the state file in
     place. The worker on startup reads resume_counter from redis and
     continues from where it left off.

What's preserved: task_counter (via redis). The baseline worker uses a
random policy, so there is no learned policy to ship — for the paper this
quantifies the "container handoff cost" without any model-state preservation,
which is the cleanest baseline distinct from Condition E (no state at all,
also a kill+restart but on the SAME node).
"""
import logging, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.cluster_runner import (
    ClusterConfig, apptainer_instance_name, kill_robot, launch_robot,
    rsync_dir, ssh_run,
)
from common.baseline_runner_base import current_node, update_node, run_baseline

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
LOG = logging.getLogger("condB")

CONDITION = "apptainer_state"
IMAGE     = "baseline.sif"
WORKER    = "cold_restart/worker_random_client.py"


def trigger_state_replay(cfg, r, robot_id, success_rate_pre, task_counter_pre):
    cid = int(robot_id.split("_")[1])
    src = current_node(robot_id)
    dst = cfg.other_node(src)
    src_chk = f"{cfg.checkpoint_base}/{apptainer_instance_name(cid)}"
    dst_chk = src_chk  # same path on dst node — keeps bind-mount mapping consistent

    LOG.info("[MIG] %s  %s → %s (state replay)", robot_id, src, dst)
    t0 = time.perf_counter()

    # 1. (state file already on src — written by worker before sleep). Make
    # sure the dir exists even if worker didn't write anything explicit;
    # task_counter lives in redis (resume_counter:<robot>) which is shared.
    ssh_run(src, f"mkdir -p {src_chk}", timeout=10)
    t_dump = time.perf_counter()

    # 2. rsync state dir cross-node.
    xfer = rsync_dir(src, src_chk, dst, dst_chk, timeout=120)
    t_xfer = time.perf_counter()

    # 3. Stop src apptainer instance.
    kill_robot(cfg, src, cid)
    time.sleep(2)

    # 4. Start fresh instance on dst node. Worker reads resume_counter from
    # redis on startup and continues.
    launch_robot(cfg, dst, cid, IMAGE, WORKER)
    update_node(robot_id, dst)
    t_restore = time.perf_counter()

    # The cold_restart worker, on startup, immediately enters its task loop
    # and starts logging. We mark migration_done so any waiter unblocks.
    r.set(f"migration_done:{robot_id}", "1", ex=600)

    total_ms = (t_restore - t0) * 1000
    return {
        "robot_id": robot_id, "src_node": src, "dst_node": dst,
        "trigger_to_dump_ms":     (t_dump    - t0) * 1000,
        "dump_to_transfer_ms":    (t_xfer    - t_dump) * 1000,
        "transfer_to_restore_ms": (t_restore - t_xfer) * 1000,
        "policy_load_ms": 0,
        "downtime_ms": total_ms,
        "total_MTT_ms": total_ms,
        "success_rate_pre": success_rate_pre,
        "success_rate_post": 0,  # filled in offline by compare script
        "regression_pct": 0,
        "replay_buffer_entries_restored": 0,
        "network_bytes_transferred": xfer["bytes_transferred"],
        "checkpoint_size_mb": 0,
        "criu_mode": "apptainer_state_replay",
    }


if __name__ == "__main__":
    sys.exit(run_baseline(
        condition=CONDITION, image=IMAGE,
        worker_script=WORKER, trigger_fn=trigger_state_replay,
    ))
