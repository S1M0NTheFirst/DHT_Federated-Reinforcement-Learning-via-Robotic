"""
Condition C — application-level COLD checkpoint, cross-node.

Replaces kernel CRIU (which is unavailable on this cluster — see README) with
a torch.save / torch.load based mechanism. The worker (cluster/workers/
worker_app_checkpoint.py) maintains a synthetic ~20 MB PyTorch state and
dumps it on migration request; the runner rsyncs the file to the destination
node and starts a fresh worker there with APP_RESTORE_FROM pointing at it.

Per-event timing:
  trigger_to_dump_ms      = wall time worker spent in torch.save
  dump_to_transfer_ms     = wall time of the cross-node rsync
  transfer_to_restore_ms  = kill_src + launch_dst + worker boots + torch.load
  policy_load_ms          = the worker's reported torch.load wall time
  downtime_ms             = total_MTT_ms (worker is idle the entire time)

CSV criu_mode column is set to "app_cold" — the value used to be
"cold_real"/"cold_simulated" when this ran via CRIU. The folder name still
contains "criu" for git-history continuity.
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
)
from common.baseline_runner_base import (  # noqa: E402
    current_node, update_node, run_baseline,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
LOG = logging.getLogger("condC")

CONDITION = "app_cold_checkpoint"
IMAGE     = "baseline.sif"
WORKER    = "/cluster_app/workers/worker_app_checkpoint.py"

# Path INSIDE the container — every robot has /checkpoints bound to its own
# per-robot host dir, so robots don't collide on this filename.
CKPT_INSIDE = "/checkpoints/state.pt"


def _wait_for_redis_key(r, key: str, *, timeout_s: float):
    """Block until `key` exists in redis or timeout. Returns parsed JSON or None."""
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


def trigger_app_cold(cfg, r, robot_id, success_rate_pre, task_counter_pre):
    cid  = int(robot_id.split("_")[1])
    src  = current_node(robot_id)
    dst  = cfg.other_node(src)
    inst = apptainer_instance_name(cid)
    src_chk_dir = f"{cfg.checkpoint_base}/{inst}"   # = /tmp/swiftbot_<cond>/<inst>
    dst_chk_dir = f"{cfg.checkpoint_base}/{inst}"

    LOG.info("[MIG] %s  %s → %s (app cold)", robot_id, src, dst)
    t0 = time.perf_counter()

    # 1. The worker has already written /checkpoints/state.pt and posted
    # `app_checkpoint_done:<robot>` to Redis right before sleeping. Fetch it.
    t_dump_start = time.perf_counter()
    dump_info = _wait_for_redis_key(r, f"app_checkpoint_done:{robot_id}",
                                    timeout_s=30)
    t_dump_done = time.perf_counter()
    if dump_info is None:
        LOG.warning("[MIG] %s: app_checkpoint_done never arrived; "
                    "proceeding anyway with whatever state.pt is on disk", robot_id)
        save_ms     = 0.0
        chk_size_mb = 0.0
        rb_entries  = 0
    else:
        save_ms     = dump_info.get("save_ms", 0.0)
        chk_size_mb = dump_info.get("size_mb", 0.0)
        rb_entries  = dump_info.get("replay_buffer_entries", 0)
        r.delete(f"app_checkpoint_done:{robot_id}")

    # 2. rsync the per-robot checkpoint dir from src to dst. Only state.pt
    # really matters but the dir may also hold .tmp leftovers — a recursive
    # rsync handles both and any future warm/ subdir.
    t_xfer_start = time.perf_counter()
    xfer = rsync_dir(src, src_chk_dir, dst, dst_chk_dir, timeout=600)
    bytes_xfer = xfer["bytes_transferred"]
    t_xfer_done = time.perf_counter()

    # 3. Kill the src instance, launch a fresh one on dst with APP_RESTORE_FROM.
    t_restore_start = time.perf_counter()
    kill_robot(cfg, src, cid)
    launch_robot(cfg, dst, cid, IMAGE, WORKER, extra_env={
        "APP_CHECKPOINT_PATH": CKPT_INSIDE,
        "APP_RESTORE_FROM":    CKPT_INSIDE,
    })

    # 4. Wait for the new worker to publish app_restore_done.
    restore_info = _wait_for_redis_key(r, f"app_restore_done:{robot_id}",
                                       timeout_s=120)
    t_restore_done = time.perf_counter()
    if restore_info is None:
        LOG.warning("[MIG] %s: app_restore_done never arrived (worker may "
                    "have failed to start on %s)", robot_id, dst)
        load_ms = 0.0
    else:
        load_ms = restore_info.get("load_ms", 0.0)
        r.delete(f"app_restore_done:{robot_id}")

    update_node(robot_id, dst)
    r.set(f"migration_done:{robot_id}", "1", ex=600)

    total_ms = (t_restore_done - t0) * 1000
    return {
        "robot_id": robot_id, "src_node": src, "dst_node": dst,
        "trigger_to_dump_ms":     save_ms,
        "dump_to_transfer_ms":    (t_xfer_done   - t_xfer_start)    * 1000,
        "transfer_to_restore_ms": (t_restore_done - t_restore_start) * 1000,
        "policy_load_ms":         load_ms,
        "downtime_ms":            total_ms,
        "total_MTT_ms":           total_ms,
        "success_rate_pre":       success_rate_pre,
        "success_rate_post":      0,
        "regression_pct":         0,
        "replay_buffer_entries_restored": rb_entries,
        "network_bytes_transferred": bytes_xfer,
        "checkpoint_size_mb":     chk_size_mb,
        "criu_mode":              "app_cold",
    }


if __name__ == "__main__":
    sys.exit(run_baseline(
        condition=CONDITION,
        image=IMAGE,
        worker_script=WORKER,
        trigger_fn=trigger_app_cold,
        # Initial launch: workers have no APP_RESTORE_FROM (fresh start) but
        # do have APP_CHECKPOINT_PATH so they know where to dump on migration.
        initial_extra_env={"APP_CHECKPOINT_PATH": CKPT_INSIDE},
    ))
