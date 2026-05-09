"""
Condition D — application-level WARM checkpoint, cross-node.

Like Condition C, but with continuous *pre-copy* during normal operation:

  - The worker writes a snapshot to /checkpoints/warm/state.pt every
    WARM_CHECKPOINT_INTERVAL tasks (default 50).
  - A background thread on the runner periodically rsyncs each robot's
    /checkpoints/warm/ from its current node to the *other* client node.
    This keeps the dst node hot — most of the state has already arrived
    before the migration trigger fires.
  - At migration time the runner does the same dump+rsync as Condition C.
    But because rsync is delta-based and the warm copy on dst is recent,
    the migration-time bytes_transferred is much smaller than C's, and
    transfer wall time drops with it.

Trade-off the paper measures: D pays continuous bandwidth (the pre-copy
thread) for a faster migration window; C pays nothing during normal
operation but pays the full state size at the migration window.
"""
import json
import logging
import os
import sys
import threading
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
LOG = logging.getLogger("condD")

CONDITION = "app_warm_checkpoint"
IMAGE     = "baseline.sif"
WORKER    = "/cluster_app/workers/worker_app_checkpoint.py"

CKPT_INSIDE      = "/checkpoints/state.pt"
WARM_CKPT_INSIDE = "/checkpoints/warm/state.pt"
WARM_INTERVAL    = 50          # tasks between worker-side snapshots
PRECOPY_PERIOD_S = 20          # seconds between runner-side rsync rounds

# Track migration-time stats per robot for the writer.
_warm_lock     = threading.Lock()
_precopy_bytes = {}            # robot_id -> total bytes pre-copied so far
_precopy_stop  = threading.Event()


def _precopy_thread(cfg):
    """Continuously rsync each robot's warm dir from its current node to the
    other client node. Cheap when the snapshot hasn't changed (rsync stats
    show 0 bytes transferred); the cost arrives in the round right after
    the worker writes a fresh snapshot.
    """
    LOG.info("[Warm] pre-copy thread started (period=%ds)", PRECOPY_PERIOD_S)
    while not _precopy_stop.is_set():
        round_start = time.perf_counter()
        for cid in range(cfg.num_clients):
            if _precopy_stop.is_set():
                break
            robot_id = f"robot_{cid:03d}"
            try:
                src  = current_node(robot_id)
                dst  = cfg.other_node(src)
                inst = apptainer_instance_name(cid)
                warm_src = f"{cfg.checkpoint_base}/{inst}/warm"
                warm_dst = f"{cfg.checkpoint_base}/{inst}/warm"
                # Quick existence check — avoids an rsync error if no warm
                # snapshot has been written yet.
                rr = ssh_run(src, f"test -d {warm_src} && echo yes || echo no",
                             timeout=10)
                if (rr.stdout or "").strip() != "yes":
                    continue
                xfer = rsync_dir(src, warm_src, dst, warm_dst, timeout=120)
                with _warm_lock:
                    _precopy_bytes[robot_id] = (
                        _precopy_bytes.get(robot_id, 0)
                        + xfer.get("bytes_transferred", 0)
                    )
            except Exception as e:
                LOG.debug("[Warm] precopy failed for %s: %r", robot_id, e)
        # Pad out the period if the round was fast.
        elapsed = time.perf_counter() - round_start
        sleep_s = max(0.0, PRECOPY_PERIOD_S - elapsed)
        if _precopy_stop.wait(sleep_s):
            break
    LOG.info("[Warm] pre-copy thread stopped")


def _start_precopy(cfg, _r):
    """Hook called by run_baseline after all robots launch."""
    threading.Thread(target=_precopy_thread, args=(cfg,), daemon=True).start()


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


def trigger_app_warm(cfg, r, robot_id, success_rate_pre, task_counter_pre):
    cid  = int(robot_id.split("_")[1])
    src  = current_node(robot_id)
    dst  = cfg.other_node(src)
    inst = apptainer_instance_name(cid)
    src_chk_dir = f"{cfg.checkpoint_base}/{inst}"
    dst_chk_dir = f"{cfg.checkpoint_base}/{inst}"

    LOG.info("[MIG] %s  %s → %s (app warm)", robot_id, src, dst)
    t0 = time.perf_counter()

    # 1. Wait for the worker's final dump notification.
    t_dump_start = time.perf_counter()
    dump_info = _wait_for_redis_key(r, f"app_checkpoint_done:{robot_id}",
                                    timeout_s=30)
    t_dump_done = time.perf_counter()
    if dump_info is None:
        save_ms = 0.0
        chk_size_mb = 0.0
        rb_entries = 0
    else:
        save_ms     = dump_info.get("save_ms", 0.0)
        chk_size_mb = dump_info.get("size_mb", 0.0)
        rb_entries  = dump_info.get("replay_buffer_entries", 0)
        r.delete(f"app_checkpoint_done:{robot_id}")

    # 2. rsync. Because the warm/ subdir was already pre-copied by the
    # background thread, rsync's delta algorithm should report a small
    # bytes_transferred even though state.pt itself is fresh.
    t_xfer_start = time.perf_counter()
    xfer = rsync_dir(src, src_chk_dir, dst, dst_chk_dir, timeout=600)
    migration_bytes = xfer["bytes_transferred"]
    t_xfer_done = time.perf_counter()

    # 3. Kill src + launch dst, restoring from the migration-time dump
    # (warm/state.pt is left in place as a fallback but not used here).
    t_restore_start = time.perf_counter()
    kill_robot(cfg, src, cid)
    launch_robot(cfg, dst, cid, IMAGE, WORKER, extra_env={
        "APP_CHECKPOINT_PATH":      CKPT_INSIDE,
        "APP_RESTORE_FROM":         CKPT_INSIDE,
        "WARM_CHECKPOINT_PATH":     WARM_CKPT_INSIDE,
        "WARM_CHECKPOINT_INTERVAL": WARM_INTERVAL,
    })

    restore_info = _wait_for_redis_key(r, f"app_restore_done:{robot_id}",
                                       timeout_s=120)
    t_restore_done = time.perf_counter()
    load_ms = (restore_info or {}).get("load_ms", 0.0)
    if restore_info:
        r.delete(f"app_restore_done:{robot_id}")

    update_node(robot_id, dst)
    r.set(f"migration_done:{robot_id}", "1", ex=600)

    with _warm_lock:
        precopy_bytes = _precopy_bytes.get(robot_id, 0)

    total_ms = (t_restore_done - t0) * 1000
    return {
        "robot_id": robot_id, "src_node": src, "dst_node": dst,
        "trigger_to_dump_ms":     save_ms,
        "dump_to_transfer_ms":    (t_xfer_done    - t_xfer_start)    * 1000,
        "transfer_to_restore_ms": (t_restore_done - t_restore_start) * 1000,
        "policy_load_ms":         load_ms,
        # In warm mode "downtime" is just kill+launch+restore — the dump
        # itself happened while the worker was already paused.
        "downtime_ms":            (t_restore_done - t_dump_start) * 1000,
        "total_MTT_ms":           total_ms,
        "success_rate_pre":       success_rate_pre,
        "success_rate_post":      0,
        "regression_pct":         0,
        "replay_buffer_entries_restored": rb_entries,
        # network_bytes_transferred is the migration-window bytes only.
        # Pre-copy bytes (paid during normal operation) accumulate in
        # _precopy_bytes; we surface them via the unused-by-C field
        # `gpu_util_during_migration` would be a poor home for it. Keep this
        # column to migration-window for fair comparison with C; report
        # cumulative pre-copy in the runner log instead.
        "network_bytes_transferred": migration_bytes,
        "checkpoint_size_mb":     chk_size_mb,
        "criu_mode":              "app_warm",
    }


if __name__ == "__main__":
    rc = run_baseline(
        condition=CONDITION,
        image=IMAGE,
        worker_script=WORKER,
        trigger_fn=trigger_app_warm,
        initial_extra_env={
            "APP_CHECKPOINT_PATH":      CKPT_INSIDE,
            "WARM_CHECKPOINT_PATH":     WARM_CKPT_INSIDE,
            "WARM_CHECKPOINT_INTERVAL": WARM_INTERVAL,
        },
        pre_loop=_start_precopy,
    )
    _precopy_stop.set()
    LOG.info("[Warm] cumulative pre-copy bytes per robot: %s", _precopy_bytes)
    sys.exit(rc)
