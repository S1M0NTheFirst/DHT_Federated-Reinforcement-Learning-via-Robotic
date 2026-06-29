"""
Task 1 vs Task 2 latency comparison — on an identical transfer model.

Task 1 = the criu_cold / criu_warm 8-robot runners.
Task 2 = the single-workload motivation experiment (run_motivation.py),
         now also 8 concurrent agents.

Both tasks recorded a real `criu dump` and a *simulated* restore. They differ
only in how transfer was originally captured: Task 1 timed a file-by-file
`copytree` (which, for the many-small-file warm image under disk contention,
ballooned to tens of seconds and was a filesystem artifact, not throughput).
Task 2's runner already models transfer as image_size / bandwidth.

To compare the two on the SAME basis, this script *re-derives* transfer for
BOTH tasks as image_size / bandwidth (the same formula run_motivation.py uses),
then recomputes MTT by swapping the modeled transfer in for whatever transfer
was originally recorded:

    transfer_model_ms = size_mb / bandwidth_mbps * 1000
    MTT_ms            = recorded_MTT - recorded_transfer + transfer_model_ms

Downtime is left as recorded: for cold it equals MTT (recomputed); for warm it
is dump + restore, which never included transfer, so it is unchanged.

Rows with criu_returncode != 0 are dropped (failed dumps). Compare cold<->cold
and warm<->warm only.

Usage (from swiftbot_rl/):
    python3 evaluation/compare_tasks.py
    python3 evaluation/compare_tasks.py --bandwidth-mbps 125
"""
import argparse, csv, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))     # swiftbot_rl/

DEFAULT_BANDWIDTH_MBPS = 125.0  # 1 Gbps — keep in sync with run_motivation.py

TASK1_COLD = os.path.join(ROOT, "criu_cold", "results", "migration_events.csv")
TASK1_WARM = os.path.join(ROOT, "criu_warm", "results", "migration_events.csv")
TASK2_CSV  = os.path.join(ROOT, "motivation", "results", "motivation.csv")

# Metrics to report, in display order.
METRICS = [
    ("size_mb",      "Checkpoint size (MB)"),
    ("dump_ms",      "Dump (ms)"),
    ("transfer_ms",  "Transfer modeled (ms)"),
    ("downtime_ms",  "Downtime (ms)"),
    ("total_MTT_ms", "MTT (ms)"),
]


def _f(row, *keys):
    """First parseable float among the given column names, else None."""
    for k in keys:
        v = row.get(k, "")
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _is_ok(row):
    rc = _f(row, "criu_returncode")
    return rc is None or rc == 0


def load_rows(path, mode_filter=None):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if mode_filter is not None:
        rows = [r for r in rows if r.get("criu_mode", "") == mode_filter]
    return [r for r in rows if _is_ok(r)]


def load_task2(mode):
    """Task 2 rows for a mode from motivation.csv; if that file has none for
    this mode (e.g. cold not re-run after a schema rotation), fall back to the
    rotated .old_schema file so the comparison still works."""
    rows = load_rows(TASK2_CSV, mode)
    if not rows and os.path.exists(TASK2_CSV + ".old_schema"):
        rows = load_rows(TASK2_CSV + ".old_schema", mode)
        if rows:
            print(f"  (Task 2 {mode}: using {os.path.basename(TASK2_CSV)}"
                  f".old_schema — {len(rows)} rows)")
    return rows


def normalize(rows, mode, bandwidth_mbps):
    """Re-derive transfer + MTT on the shared model. Returns list of dicts."""
    out = []
    for r in rows:
        # Size column differs: Task 1 = checkpoint_size_mb, Task 2 = criu_size_mb.
        size_mb = _f(r, "checkpoint_size_mb", "criu_size_mb")
        dump    = _f(r, "trigger_to_dump_ms", "dump_ms")
        restore = _f(r, "transfer_to_restore_ms", "restore_ms")
        rec_xfer = _f(r, "dump_to_transfer_ms")
        rec_mtt  = _f(r, "total_MTT_ms")
        if size_mb is None or rec_mtt is None:
            continue

        transfer_model = (size_mb / bandwidth_mbps) * 1000.0 if bandwidth_mbps else 0.0
        rec_xfer = rec_xfer if rec_xfer is not None else 0.0
        mtt = rec_mtt - rec_xfer + transfer_model

        if mode == "cold":
            downtime = mtt                      # cold: container stopped whole time
        else:
            downtime = _f(r, "downtime_ms")     # warm: dump + restore, no transfer
            if downtime is None:
                downtime = (dump or 0.0) + (restore or 0.0)

        out.append({
            "size_mb":      size_mb,
            "dump_ms":      dump if dump is not None else 0.0,
            "transfer_ms":  transfer_model,
            "downtime_ms":  downtime,
            "total_MTT_ms": mtt,
        })
    return out


def pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = q / 100.0 * (len(s) - 1)
    lo = int(pos)
    frac = pos - lo
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * frac


def summarize(name, rows):
    print(f"\n{'='*64}\n{name}   (n={len(rows)})\n{'='*64}")
    if not rows:
        print("  (no data — run the experiment first)")
        return
    print(f"  {'metric':24s}{'mean':>12s}{'median':>12s}{'p95':>12s}")
    for key, label in METRICS:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            continue
        print(f"  {label:24s}{st.mean(vals):12.1f}"
              f"{st.median(vals):12.1f}{pct(vals, 95):12.1f}")


def compare(label, t1, t2):
    print(f"\n\n########  {label}  ########")
    summarize("TASK 1", t1)
    summarize("TASK 2", t2)
    if t1 and t2:
        print(f"\n  {'metric':24s}{'Task1 mean':>12s}{'Task2 mean':>12s}{'T2/T1':>10s}")
        for key, lbl in METRICS:
            a = st.mean([r[key] for r in t1])
            b = st.mean([r[key] for r in t2])
            ratio = (b / a) if a else float("nan")
            print(f"  {lbl:24s}{a:12.1f}{b:12.1f}{ratio:10.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandwidth-mbps", type=float, default=DEFAULT_BANDWIDTH_MBPS,
                    help=f"Modeled transfer link speed MB/s "
                         f"(default {DEFAULT_BANDWIDTH_MBPS:.0f} = 1 Gbps). Must "
                         f"match the value used when running run_motivation.py.")
    args = ap.parse_args()
    bw = args.bandwidth_mbps

    print(f"Transfer model: image_size / {bw:.0f} MB/s  "
          f"({bw*8/1000:.2f} Gbps), applied identically to both tasks.")
    print("Restore is simulated in both tasks; failed dumps (rc!=0) dropped.")

    t1_cold = normalize(load_rows(TASK1_COLD), "cold", bw)
    t2_cold = normalize(load_task2("cold"), "cold", bw)
    compare("COLD  (criu_cold  vs  motivation cold)", t1_cold, t2_cold)

    t1_warm = normalize(load_rows(TASK1_WARM), "warm", bw)
    t2_warm = normalize(load_task2("warm"), "warm", bw)
    compare("WARM  (criu_warm  vs  motivation warm)", t1_warm, t2_warm)


if __name__ == "__main__":
    main()
