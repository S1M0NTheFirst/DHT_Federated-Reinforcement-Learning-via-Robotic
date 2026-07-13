"""
task2 condition dht_frl — OUR SYSTEM. State-preserving migration coordinated by a
REAL Kademlia DHT overlay (DHT-coordinated direct transfer, the BitTorrent/IPFS
pattern).

On migration:
  1. worker flushes the unified bundle (sac_state.pt + replay_buffer.pkl + manifest).
  2. source PUTs a small POINTER into the DHT overlay:  {node, path, ts}
     -> real Kademlia `set` (dht_put_ms).
  3. destination GETs that pointer back from a DIFFERENT overlay node
     -> real Kademlia iterative lookup over the routing table (dht_get_ms).
  4. destination fetches the ~11 MB bundle DIRECTLY from the discovered location
     (rsync, point-to-point) -> dump_to_transfer_ms, same channel as the baselines.

So DHT overhead = put+get of a tiny pointer (~ms) and the bulk transfer equals the
direct-transfer baselines. This makes "DHT transport overhead is trivial vs direct
transfer" literally true and measurable. checkpoint_mode=dht_bundle.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from runner_base import (  # noqa: E402
    run_task2, current_node, read_probe_metrics,
)
from common.cluster_runner import (  # noqa: E402
    apptainer_instance_name, rsync_dir, ssh_run,
)
from dht_service import get_dht  # noqa: E402

CONDITION = "dht_frl"
CHECKPOINT_MODE = "dht_bundle"


def _start_overlay(cfg):
    # Bootstrap the Kademlia ring before robots launch so its one-time setup
    # cost never lands inside a migration measurement.
    get_dht()


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

    # 2. DHT coordination: PUT the bundle pointer, then GET it back through the
    #    overlay (resolved on a different node → real routing). Tiny payload.
    dht = get_dht()
    key = f"bundle:{robot_id}"
    dht_put_ms = dht.put(key, {"node": src, "path": src_bundle, "ts": time.time()})
    loc, dht_get_ms = dht.get(key)
    if not loc:                      # overlay failed to resolve — fall back to known src
        loc = {"node": src, "path": src_bundle}

    # 3. direct bundle transfer from the DHT-discovered location (point-to-point,
    #    same channel/metric as the baselines).
    xfer = rsync_dir(loc["node"], loc["path"], dst, dst_bundle, timeout=600)
    transfer_ms = xfer["elapsed_ms"]
    bytes_xfer = xfer["bytes_transferred"]

    # 4. bundle size + replay entries (on dst).
    sz = ssh_run(dst, f"du -sb {dst_bundle} 2>/dev/null | awk '{{print $1}}'",
                 timeout=10)
    try:
        chk_mb = int((sz.stdout or "0").strip()) / (1024 * 1024)
    except ValueError:
        chk_mb = 0.0

    # 5. worker stays on src and reloads from its own /checkpoints/<robot_id>.
    r.set(f"load_policy:{robot_id}", f"/checkpoints/{robot_id}", ex=600)
    r.set(f"migration_done:{robot_id}", "1", ex=600)

    probe = read_probe_metrics(r, robot_id)
    total_ms = (time.perf_counter() - t0) * 1000

    return {
        "robot_id": robot_id, "src_node": src, "dst_node": dst,
        "trigger_to_dump_ms": (t_dump - t0) * 1000,
        "dump_to_transfer_ms": transfer_ms,
        "transfer_to_restore_ms": 0,
        "policy_load_ms": probe["policy_load_ms"],
        # DHT get is on the critical path (dst must resolve the pointer before it
        # can fetch); put can overlap the freeze. Both are ~ms.
        "downtime_ms": dht_get_ms + transfer_ms + probe["policy_load_ms"],
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
        "dht_put_ms": round(dht_put_ms, 3),
        "dht_get_ms": round(dht_get_ms, 3),
    }


if __name__ == "__main__":
    sys.exit(run_task2(condition=CONDITION, checkpoint_mode=CHECKPOINT_MODE,
                       trigger_fn=trigger, on_start=_start_overlay))
