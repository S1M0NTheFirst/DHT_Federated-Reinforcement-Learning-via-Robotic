"""
task2 condition dmtcp — the HEAVY real-checkpoint baseline DHT-FRL must beat
(retires task1's "faked CRIU" criticism).

DMTCP cannot wrap the live FL worker (it jams the worker's gRPC connection to
Flower). So the worker runs NORMALLY (bundle-preserving, like app_cold — giving
a proper continuous learning curve), and at migration the trigger:
  1. rsyncs the bundle cross-node (state preservation / continuity), releases
     the worker immediately (it never waits on the slow DMTCP step), THEN
  2. runs a REAL DMTCP full-process checkpoint of a short-lived torch probe that
     holds the same model+replay footprint (no gRPC) → a genuine ~1.9 GB image;
     measures dump time + image size + cross-node transfer.
Only the FIRST migration per robot is DMTCP-measured (bounds the 1.9 GB×N cost);
later events still preserve state via the bundle. checkpoint_mode=dmtcp.
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from runner_base import (  # noqa: E402
    run_task2, current_node, read_probe_metrics, rsync_transport, bundle_trigger,
)
from common.cluster_runner import (  # noqa: E402
    apptainer_instance_name, rsync_dir, ssh_run,
)

CONDITION = "dmtcp"
CHECKPOINT_MODE = "dmtcp"

_measured = set()          # robots whose heavy DMTCP checkpoint we've already timed
_measured_lock = threading.Lock()
# The monitor now handles migrations CONCURRENTLY. Each heavy DMTCP checkpoint is
# a ~1.9 GB image + cross-node transfer; 20 at once would exhaust disk/network.
# Cap how many run simultaneously (the worker is already released by the bundle
# transport before this runs, so the FL round is never blocked by the wait).
_HEAVY_MAX = int(os.environ.get("DMTCP_HEAVY_CONCURRENCY", "2"))
_heavy_sem = threading.Semaphore(_HEAVY_MAX)


def _dmtcp_measure(cfg, src, dst, cid, robot_id):
    """Launch the torch probe under DMTCP on `src`, checkpoint it into a real
    full-process image, transfer it to `dst`, and return timing/size. Best-effort
    — any failure returns sentinels so the run never breaks."""
    home = os.environ["HOME"]
    img_dir = cfg.img_dir
    cluster_root = os.environ["CLUSTER_ROOT"]
    pylibs = os.path.join(img_dir, "pylibs")
    pylibs2 = os.environ["TASK2_PYLIBS2"]
    conda_base = os.environ.get("CONDA_BASE")
    conda_env = os.environ.get("CONDA_ENV", "base")
    inst = apptainer_instance_name(cid)
    chk = f"{cfg.checkpoint_base}/{inst}"            # host bind -> /checkpoints
    img_host = f"{chk}/dmtcp_img_{robot_id}"         # host path of the image dir
    img_dst = f"{cfg.checkpoint_base}/{inst}_dest/dmtcp_img_{robot_id}"
    coord_port = 7900 + cid

    # One bash script (run inside the container on src) that launches the probe
    # under DMTCP, checkpoints it, and prints DUMP_MS / SIZE_B. Leaves the image
    # on disk for the transfer step below.
    inner = (
        f"export PATH={home}/dmtcp/bin:$PATH; "
        f"export DMTCP_DL_PLUGIN=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        f"OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; "
        f"rm -rf /checkpoints/dmtcp_img_{robot_id}; "
        f"mkdir -p /checkpoints/dmtcp_img_{robot_id}; "
        f"dmtcp_launch --new-coordinator --coord-port {coord_port} "
        f"--ckptdir /checkpoints/dmtcp_img_{robot_id} "
        f"python3 /cluster_app/task2/worker/dmtcp_probe.py "
        f"--bundle /checkpoints/{robot_id} "
        f"--ready /checkpoints/dmtcp_img_{robot_id}/ready "
        f"> /checkpoints/dmtcp_img_{robot_id}/probe.log 2>&1 & "
        f"for i in $(seq 90); do [ -f /checkpoints/dmtcp_img_{robot_id}/ready ] && break; sleep 1; done; "
        f"t0=$(date +%s%N); "
        f"dmtcp_command --coord-port {coord_port} --bcheckpoint; "
        f"t1=$(date +%s%N); "
        f"echo DUMP_MS=$(( (t1-t0)/1000000 )); "
        f"echo SIZE_B=$(du -sb /checkpoints/dmtcp_img_{robot_id} | awk '{{print $1}}'); "
        f"dmtcp_command --coord-port {coord_port} --quit 2>/dev/null || true"
    )
    import shlex
    remote = (
        f"source {conda_base}/bin/activate {conda_env} && "
        f"apptainer exec "
        f"--bind {cluster_root}:/cluster_app "
        f"--bind {chk}:/checkpoints "
        f"--bind {pylibs}:/pylibs --bind {pylibs2}:/pylibs2 "
        f"--bind {home}/dmtcp:{home}/dmtcp "
        f"{img_dir}/robot.sif bash -lc {shlex.quote(inner)}"
    )
    dump_ms, size_mb = -1.0, -1.0
    try:
        rr = ssh_run(src, remote, timeout=600)
        for line in (rr.stdout or "").splitlines():
            if line.startswith("DUMP_MS="):
                dump_ms = float(line.split("=", 1)[1].strip())
            elif line.startswith("SIZE_B="):
                size_mb = int(line.split("=", 1)[1].strip()) / (1024 * 1024)
    except Exception as e:
        print(f"[dmtcp] {robot_id} probe checkpoint failed: {e!r}", flush=True)
        return {"dump_ms": -1, "size_mb": -1, "transfer_ms": -1, "bytes": 0}

    # transfer the heavy image cross-node.
    xfer = rsync_dir(src, img_host, dst, img_dst, timeout=1800)
    # free disk on both nodes — the image is only for measurement.
    ssh_run(src, f"rm -rf {img_host}", timeout=60)
    ssh_run(dst, f"rm -rf {img_dst}", timeout=60)
    return {"dump_ms": dump_ms, "size_mb": size_mb,
            "transfer_ms": xfer["elapsed_ms"],
            "bytes": xfer["bytes_transferred"]}


def trigger(cfg, r, robot_id, sr_pre, tc_pre):
    cid = int(robot_id.split("_")[1])
    src = current_node(robot_id)
    dst = cfg.other_node(src)

    # 1. State-preserving migration (bundle rsync) + release worker + probe —
    #    identical to app_cold, so dmtcp gets a proper continuous learning curve.
    metrics = bundle_trigger(cfg, r, robot_id, sr_pre, transport=rsync_transport)

    # 2. Heavy DMTCP measurement (first migration per robot only, to bound the
    #    1.9 GB × N transfer cost). The worker is already running again by now
    #    (bundle_trigger released it), so this heavy step never blocks the FL
    #    round. A semaphore caps how many run at once across the concurrent
    #    monitor threads.
    with _measured_lock:
        do_measure = robot_id not in _measured
        if do_measure:
            _measured.add(robot_id)
    if do_measure:
        with _heavy_sem:
            m = _dmtcp_measure(cfg, src, dst, cid, robot_id)
        # Overlay the heavy-checkpoint numbers onto the metrics row.
        metrics["trigger_to_dump_ms"] = round(m["dump_ms"], 2)      # DMTCP dump
        metrics["dump_to_transfer_ms"] = round(m["transfer_ms"], 2)  # heavy xfer
        metrics["downtime_ms"] = round(max(m["dump_ms"], 0) + max(m["transfer_ms"], 0), 2)
        metrics["checkpoint_size_mb"] = round(m["size_mb"], 2)       # ~1900 MB
        metrics["network_bytes_transferred"] = m["bytes"]
        print(f"[dmtcp] {robot_id} heavy ckpt: dump={m['dump_ms']:.0f}ms "
              f"size={m['size_mb']:.0f}MB xfer={m['transfer_ms']:.0f}ms", flush=True)
    return metrics


if __name__ == "__main__":
    sys.exit(run_task2(condition=CONDITION, checkpoint_mode=CHECKPOINT_MODE,
                       trigger_fn=trigger))
