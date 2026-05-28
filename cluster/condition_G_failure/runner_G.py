"""
Condition G — failure injection during migration.

Reuses the cold app-checkpoint mechanism (Condition C) but deliberately kills
the DESTINATION worker mid-restore, forcing the runner to detect the failure
and retry the migration on a fallback node. Measures the fault-recovery
overhead the checkpoint mechanism pays:

  - re-launch + re-load on a different node, and
  - the extra downtime / success-rate dip the robot suffers.

Contrast for the paper: DHT bundle transfer (Condition A) recovers by
re-pulling weights from the overlay (any peer), so a destination failure costs
~one extra pull; the checkpoint mechanism must restart the whole
kill→launch→torch.load sequence again.

Env:
  FAULT_DELAY_MS   ms after launch before we kill the destination (default 500)
  MIGRATION_OFFSET per-robot offset (default from cluster_config)
"""
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.cluster_runner import (        # noqa: E402
    apptainer_instance_name,
    launch_robot, kill_robot,
    rsync_dir, ssh_run,
    post_migration_recovery,
)
from common.baseline_runner_base import (  # noqa: E402
    current_node, update_node, run_baseline,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
LOG = logging.getLogger("condG")

CONDITION         = "failure_injection"
IMAGE             = "baseline.sif"
WORKER            = "/cluster_app/workers/worker_app_checkpoint.py"
PRETRAINED_INSIDE = "/cluster_app/common/pretrained_policy.pt"
CKPT_INSIDE       = "/checkpoints/state.pt"
FAULT_DELAY_MS    = int(os.environ.get("FAULT_DELAY_MS", "500"))


def _wait_for_redis_key(r, key: str, *, timeout_s: float):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        raw = r.get(key)
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                return None
        time.sleep(0.1)
    return None


def trigger_failure_cold(cfg, r, robot_id, success_rate_pre, task_counter_pre):
    cid  = int(robot_id.split("_")[1])
    src  = current_node(robot_id)
    dst  = cfg.other_node(src)
    inst = apptainer_instance_name(cid)
    chk_dir = f"{cfg.checkpoint_base}/{inst}"

    LOG.info("[MIG] %s  %s → %s (FAULT INJECTION)", robot_id, src, dst)
    t0 = time.perf_counter()

    # 1. Worker has dumped state.pt and posted app_checkpoint_done.
    dump_info = _wait_for_redis_key(r, f"app_checkpoint_done:{robot_id}", timeout_s=30)
    save_ms     = (dump_info or {}).get("save_ms", 0.0)
    chk_size_mb = (dump_info or {}).get("size_mb", 0.0)
    rb_entries  = (dump_info or {}).get("replay_buffer_entries", 0)
    if dump_info:
        r.delete(f"app_checkpoint_done:{robot_id}")

    # 2. rsync the checkpoint dir src → dst. (The src copy stays on disk, so
    # the retry on the fallback node can reuse it without re-dumping.)
    t_xfer_start = time.perf_counter()
    xfer = rsync_dir(src, chk_dir, dst, chk_dir, timeout=600)
    bytes_xfer = xfer["bytes_transferred"]
    t_xfer_done = time.perf_counter()

    # 3. Kill src, launch on dst — the FIRST (doomed) attempt.
    kill_robot(cfg, src, cid)
    r.delete(f"app_restore_done:{robot_id}")
    launch_robot(cfg, dst, cid, IMAGE, WORKER, extra_env={
        "APP_CHECKPOINT_PATH": CKPT_INSIDE,
        "APP_RESTORE_FROM":    CKPT_INSIDE,
    })

    # 4. FAULT: kill the destination worker mid-restore.
    time.sleep(FAULT_DELAY_MS / 1000.0)
    LOG.warning("[FAULT] killing destination %s for %s mid-restore", dst, robot_id)
    kill_robot(cfg, dst, cid)

    # 5. Detect that the doomed attempt did not finish restoring.
    restore_info = _wait_for_redis_key(r, f"app_restore_done:{robot_id}", timeout_s=8)
    fault_confirmed = restore_info is None
    if not fault_confirmed:
        # Restore slipped in before our kill landed — rare, but then there was
        # effectively no fault to recover from.
        LOG.info("[FAULT] %s restored before kill landed; no recovery needed", robot_id)
        r.delete(f"app_restore_done:{robot_id}")

    # 6. RETRY on a fallback node. src is known-alive (we just killed its
    # worker, not the node) and still holds the checkpoint, so retry there.
    retry_count = 0
    load_ms = (restore_info or {}).get("load_ms", 0.0)
    fallback = src
    if fault_confirmed:
        retry_count = 1
        LOG.info("[FAULT] retrying %s on fallback node %s", robot_id, fallback)
        r.delete(f"app_restore_done:{robot_id}")
        kill_robot(cfg, dst, cid)  # belt-and-suspenders: ensure dst is clear
        launch_robot(cfg, fallback, cid, IMAGE, WORKER, extra_env={
            "APP_CHECKPOINT_PATH": CKPT_INSIDE,
            "APP_RESTORE_FROM":    CKPT_INSIDE,
        })
        retry_info = _wait_for_redis_key(r, f"app_restore_done:{robot_id}", timeout_s=120)
        load_ms = (retry_info or {}).get("load_ms", 0.0)
        if retry_info:
            r.delete(f"app_restore_done:{robot_id}")
        landed = fallback
    else:
        landed = dst

    update_node(robot_id, landed)
    r.set(f"migration_done:{robot_id}", "1", ex=600)

    total_ms = (time.perf_counter() - t0) * 1000

    recov = post_migration_recovery(r, robot_id, task_counter_pre, success_rate_pre)
    success_rate_post = recov["success_rate_post"]
    regression = 0.0
    if success_rate_pre > 0:
        regression = (success_rate_pre - success_rate_post) / success_rate_pre * 100

    return {
        "robot_id": robot_id, "src_node": src, "dst_node": landed,
        "trigger_to_dump_ms":     save_ms,
        "dump_to_transfer_ms":    (t_xfer_done - t_xfer_start) * 1000,
        "transfer_to_restore_ms": 0,
        "policy_load_ms":         load_ms,
        "downtime_ms":            total_ms,
        "total_MTT_ms":           total_ms,
        "success_rate_pre":       success_rate_pre,
        "success_rate_post":      success_rate_post,
        "regression_pct":         round(regression, 2),
        "replay_buffer_entries_restored": rb_entries,
        "network_bytes_transferred": bytes_xfer,
        "checkpoint_size_mb":     chk_size_mb,
        "criu_mode":              "app_cold_fault",
        "fault_injected":         1 if fault_confirmed else 0,
        "retry_count":            retry_count,
        "total_recovery_ms":      round(total_ms, 2),
        "throughput_post_60s":    recov["throughput_post_60s"],
        "recovery_tasks_to_pre":  recov["recovery_tasks_to_pre"],
    }


if __name__ == "__main__":
    sys.exit(run_baseline(
        condition=CONDITION,
        image=IMAGE,
        worker_script=WORKER,
        trigger_fn=trigger_failure_cold,
        initial_extra_env={
            "APP_CHECKPOINT_PATH":    CKPT_INSIDE,
            "WORKER_PRETRAINED_PATH": PRETRAINED_INSIDE,
        },
    ))
