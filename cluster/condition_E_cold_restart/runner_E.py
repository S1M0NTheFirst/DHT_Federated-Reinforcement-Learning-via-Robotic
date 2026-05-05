"""
Condition E — Cold restart cross-node.

Mechanism: kill apptainer instance on src node, launch fresh instance on dst
node. NO state preservation beyond the resume_counter that the worker writes
to redis before sleeping (so the new worker doesn't start from task 0). This
is the floor against which all other conditions are measured.
"""
import logging, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.cluster_runner import (
    ClusterConfig, apptainer_instance_name, kill_robot, launch_robot,
)
from common.baseline_runner_base import current_node, update_node, run_baseline

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
LOG = logging.getLogger("condE")

CONDITION = "cold_restart"
IMAGE     = "baseline.sif"
WORKER    = "cold_restart/worker_random_client.py"


def trigger_cold_restart(cfg, r, robot_id, success_rate_pre, task_counter_pre):
    cid = int(robot_id.split("_")[1])
    src = current_node(robot_id)
    dst = cfg.other_node(src)

    LOG.info("[MIG] %s  %s → %s (cold restart)", robot_id, src, dst)
    t0 = time.perf_counter()

    # Clear stale migration_request/migration_done so the new worker doesn't
    # inherit them and immediately re-trigger. (resume_counter is intentionally
    # left in redis — that's how the new worker knows where to pick up.)
    r.delete(f"migration_request:{robot_id}", f"migration_done:{robot_id}")

    kill_robot(cfg, src, cid)
    t_kill = time.perf_counter()
    time.sleep(2)

    launch_robot(cfg, dst, cid, IMAGE, WORKER)
    update_node(robot_id, dst)
    t_launch = time.perf_counter()

    r.set(f"migration_done:{robot_id}", "1", ex=600)

    total_ms = (t_launch - t0) * 1000
    return {
        "robot_id": robot_id, "src_node": src, "dst_node": dst,
        "trigger_to_dump_ms":     0,
        "dump_to_transfer_ms":    0,
        "transfer_to_restore_ms": (t_launch - t_kill) * 1000,
        "policy_load_ms": 0,
        "downtime_ms": total_ms,
        "total_MTT_ms": total_ms,
        "success_rate_pre": success_rate_pre,
        "success_rate_post": 0,
        "regression_pct": 0,
        "replay_buffer_entries_restored": 0,
        "network_bytes_transferred": 0,
        "checkpoint_size_mb": 0,
        "criu_mode": "cold_restart",
    }


if __name__ == "__main__":
    sys.exit(run_baseline(
        condition=CONDITION, image=IMAGE,
        worker_script=WORKER, trigger_fn=trigger_cold_restart,
    ))
