"""
task2 condition dht_frl — our system. State-preserving migration by transferring
the unified bundle (sac_state.pt + replay_buffer.pkl + manifest.json) between
nodes. Transport here is rsync of the bundle (same cluster reality as task1
condition A, which admits the worker process isn't relocated — the bundle is
replicated and the live worker reloads it in place). checkpoint_mode=dht_bundle.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# task2's runner_base (top-level module on task2/common); NOT cluster/common,
# which is task1's package and has no runner_base.
from runner_base import (  # noqa: E402
    run_task2, current_node, read_probe_metrics,
)
from common.cluster_runner import (  # noqa: E402
    apptainer_instance_name, rsync_dir, ssh_run,
)

CONDITION = "dht_frl"
CHECKPOINT_MODE = "dht_bundle"


def trigger(cfg, r, robot_id, sr_pre, tc_pre):
    cid = int(robot_id.split("_")[1])
    src = current_node(robot_id)
    dst = cfg.other_node(src)
    inst = apptainer_instance_name(cid)
    # Worker writes to /checkpoints/<robot_id> → host {base}/<inst>/<robot_id>.
    src_bundle = f"{cfg.checkpoint_base}/{inst}/{robot_id}"
    dst_bundle = f"{cfg.checkpoint_base}/{inst}_dest/{robot_id}"

    t0 = time.perf_counter()
    # 1. wait for worker to flush the bundle.
    deadline = time.time() + 60
    while time.time() < deadline:
        if r.get(f"ready_for_criu:{robot_id}"):
            r.delete(f"ready_for_criu:{robot_id}")
            break
        time.sleep(0.2)
    t_dump = time.perf_counter()

    # 2. rsync the bundle across nodes (this is the transport we measure).
    xfer = rsync_dir(src, src_bundle, dst, dst_bundle, timeout=600)
    transfer_ms = xfer["elapsed_ms"]
    bytes_xfer = xfer["bytes_transferred"]

    # 3. bundle size + replay entries (on dst).
    sz = ssh_run(dst, f"du -sb {dst_bundle} 2>/dev/null | awk '{{print $1}}'",
                 timeout=10)
    try:
        chk_mb = int((sz.stdout or "0").strip()) / (1024 * 1024)
    except ValueError:
        chk_mb = 0.0

    # 4. worker stays on src and reloads from its own /checkpoints/<robot_id>
    #    (bundle already present locally where it wrote it). Point load_policy
    #    there and release the worker.
    r.set(f"load_policy:{robot_id}", f"/checkpoints/{robot_id}", ex=600)
    r.set(f"migration_done:{robot_id}", "1", ex=600)

    # 5. read the worker's post-resume probe (behavioral losslessness).
    #    success_rate_post / regression / recovery are NOT computed here (task1's
    #    task-based post_migration_recovery scans task_logs by task_counter,
    #    which the round-based task2 eval rows don't have). Continuity is
    #    computed post-hoc in evaluation2 from task_logs.csv aligned to the
    #    migration fl_round markers. We record the real pre-migration success
    #    and leave post fields as sentinels.
    probe = read_probe_metrics(r, robot_id)
    total_ms = (time.perf_counter() - t0) * 1000

    return {
        "robot_id": robot_id, "src_node": src, "dst_node": dst,
        "trigger_to_dump_ms": (t_dump - t0) * 1000,
        "dump_to_transfer_ms": transfer_ms,
        "transfer_to_restore_ms": 0,
        "policy_load_ms": probe["policy_load_ms"],
        "downtime_ms": transfer_ms + probe["policy_load_ms"],
        "total_MTT_ms": total_ms,
        "success_rate_pre": sr_pre,
        "success_rate_post": -1,
        "regression_pct": -1,
        "fl_rounds_to_recover": -1,
        "replay_buffer_entries_restored": probe["replay_entries_post"],
        "network_bytes_transferred": bytes_xfer,
        "checkpoint_size_mb": round(chk_mb, 3),
        "throughput_post_60s": 0,
        "recovery_tasks_to_pre": -1,
        "policy_action_mse": probe["policy_action_mse"],
        "policy_weight_l2": probe["policy_weight_l2"],
    }


if __name__ == "__main__":
    sys.exit(run_task2(condition=CONDITION, checkpoint_mode=CHECKPOINT_MODE,
                       trigger_fn=trigger))
