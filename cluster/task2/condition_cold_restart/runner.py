"""
task2 condition cold_restart — negative control (Issue #2 floor). No state is
preserved: the worker runs with COLD_RESTART=1, so on resume it DISCARDS the
local bundle (empty replay buffer + reset optimizers/critics), keeping only the
global actor weights it holds. The trigger does NO transport — it just releases
the worker. Expect: eval-return dips and re-climbs; policy_action_mse LARGE.
checkpoint_mode=none.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common.runner_base import (  # noqa: E402
    run_task2, current_node, read_probe_metrics,
)

CONDITION = "cold_restart"
CHECKPOINT_MODE = "none"


def trigger(cfg, r, robot_id, sr_pre, tc_pre):
    src = current_node(robot_id)
    dst = cfg.other_node(src)
    t0 = time.perf_counter()

    # No bundle transport. Consume the ready flag and release the worker; it
    # wipes local state itself (COLD_RESTART=1).
    r.delete(f"ready_for_criu:{robot_id}")
    # Do NOT set load_policy — the worker ignores it under COLD_RESTART anyway.
    r.set(f"migration_done:{robot_id}", "1", ex=600)

    # Continuity computed post-hoc from task_logs.csv (see dht_frl runner note).
    probe = read_probe_metrics(r, robot_id)
    total_ms = (time.perf_counter() - t0) * 1000

    return {
        "robot_id": robot_id, "src_node": src, "dst_node": dst,
        "trigger_to_dump_ms": 0, "dump_to_transfer_ms": 0,
        "transfer_to_restore_ms": 0,
        "policy_load_ms": probe["policy_load_ms"],
        "downtime_ms": total_ms, "total_MTT_ms": total_ms,
        "success_rate_pre": sr_pre,
        "success_rate_post": -1,
        "regression_pct": -1,
        "fl_rounds_to_recover": -1,
        "replay_buffer_entries_restored": probe["replay_entries_post"],
        "network_bytes_transferred": 0, "checkpoint_size_mb": 0,
        "throughput_post_60s": 0,
        "recovery_tasks_to_pre": -1,
        "policy_action_mse": probe["policy_action_mse"],
        "policy_weight_l2": probe["policy_weight_l2"],
    }


if __name__ == "__main__":
    sys.exit(run_task2(condition=CONDITION, checkpoint_mode=CHECKPOINT_MODE,
                       trigger_fn=trigger,
                       initial_extra_env={"COLD_RESTART": "1"}))
