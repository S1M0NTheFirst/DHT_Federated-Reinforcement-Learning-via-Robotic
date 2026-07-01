"""
Generate IEEE-style publication figures comparing the 5 migration conditions.
Outputs PDF (vector, for LaTeX inclusion) and PNG (for previews) to ./figures/.

Run from the repo root:  python3 cluster/evaluation/make_figures.py
"""
import csv
import os
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- IEEE-style global style ------------------------------------------------
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "axes.linewidth":    0.6,
    "lines.linewidth":   1.0,
    "grid.linewidth":    0.4,
    "grid.alpha":        0.4,
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "figure.dpi":        150,
    "savefig.dpi":       600,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype":      42,   # editable text in PDF
    "ps.fonttype":       42,
})

# Grayscale-friendly palette + hatches so figures survive B&W printing.
# Condition B (Apptainer state) dropped — Apptainer has no real checkpoint
# support; baseline was an approximation that introduced a validity question.
ORDER  = ["A", "C", "D", "E"]
LABELS = {
    "A": "A. DHT+FRL\n(proposed)",
    "C": "C. App cold\ncheckpoint",
    "D": "D. App warm\ncheckpoint",
    "E": "E. Cold\nrestart",
}
COLORS = {
    "A": "#1f3a93",  # deep blue — our method
    "C": "#a04000",
    "D": "#d68910",
    "E": "#117a65",
}
HATCH  = {"A": "", "C": "xx", "D": "\\\\", "E": ".."}

DIRS = {
    "A": "condition_A_dht_frl",
    "C": "condition_C_criu_cold",
    "D": "condition_D_criu_warm",
    "E": "condition_E_cold_restart",
}

ROOT     = Path(__file__).resolve().parent.parent
RESULTS  = ROOT / "results"
OUTDIR   = Path(__file__).resolve().parent / "figures"
OUTDIR.mkdir(exist_ok=True)


def load(cond_key):
    path = RESULTS / DIRS[cond_key] / "migration_events.csv"
    with open(path) as f:
        return list(csv.DictReader(f))


def load_folder(folder_name):
    """Load migration_events.csv from an arbitrary results subdir by name.
    Returns [] if the folder or CSV doesn't exist yet (F/G runs may not have
    been done) so figures degrade gracefully instead of crashing."""
    path = RESULTS / folder_name / "migration_events.csv"
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


DATA = {k: load(k) for k in ORDER}


def col(rows, key, cast=float):
    out = []
    for r in rows:
        v = r.get(key, "")
        if v == "" or v is None:
            continue
        try:
            out.append(cast(v))
        except ValueError:
            pass
    return out


def save(fig, name):
    fig.savefig(OUTDIR / f"{name}.pdf")
    fig.savefig(OUTDIR / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# IEEE single-column width is 3.5 in. Double-column ~7.16 in.
W1 = 3.5      # single column
W2 = 7.16     # double column


# -----------------------------------------------------------------------------
# Fig 1: Migration downtime (median bar + std error bar)
# -----------------------------------------------------------------------------
def fig_mtt_bar():
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    xs = np.arange(len(ORDER))
    meds, means, stds = [], [], []
    for k in ORDER:
        v = col(DATA[k], "total_MTT_ms")
        v = [x / 1000.0 for x in v]   # seconds
        meds.append(statistics.median(v))
        means.append(statistics.mean(v))
        stds.append(statistics.stdev(v) if len(v) > 1 else 0)
    bars = ax.bar(xs, meds,
                  yerr=stds, capsize=2,
                  color=[COLORS[k] for k in ORDER],
                  edgecolor="black", linewidth=0.5,
                  hatch=[HATCH[k] for k in ORDER],
                  error_kw=dict(elinewidth=0.6, ecolor="black"))
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[k] for k in ORDER])
    ax.set_ylabel("Migration downtime (s)")
    ax.set_yscale("log")
    ax.set_ylim(1, 200)
    # value labels above bars
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width()/2, meds[i]*1.15,
                f"{meds[i]:.1f}s", ha="center", va="bottom", fontsize=7)
    ax.set_title("Migration downtime per event (median ± std)")
    save(fig, "fig1_mtt_bar")


# -----------------------------------------------------------------------------
# Fig 2: Network bytes per migration (log scale, dramatic contrast)
# -----------------------------------------------------------------------------
def fig_network_bytes():
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    xs = np.arange(len(ORDER))
    vals = []
    for k in ORDER:
        v = col(DATA[k], "network_bytes_transferred")
        v = [x / (1024 * 1024) for x in v]  # MB
        vals.append(statistics.mean(v) if v else 0)
    # Floor zero to 0.01 so the log bar is visible
    plot_vals = [max(v, 0.01) for v in vals]
    bars = ax.bar(xs, plot_vals,
                  color=[COLORS[k] for k in ORDER],
                  edgecolor="black", linewidth=0.5,
                  hatch=[HATCH[k] for k in ORDER])
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[k] for k in ORDER])
    ax.set_ylabel("Network bytes per migration (MB)")
    ax.set_yscale("log")
    ax.set_ylim(0.005, 100)
    for i, (b, raw) in enumerate(zip(bars, vals)):
        label = f"{raw:.2f} MB" if raw >= 0.01 else "≈0"
        ax.text(b.get_x() + b.get_width()/2, plot_vals[i]*1.4,
                label, ha="center", va="bottom", fontsize=7)
    ax.set_title("Network cost per migration (mean)")
    save(fig, "fig2_network_bytes")


# -----------------------------------------------------------------------------
# Fig 3: CDF of MTT per condition — shows A's tight distribution vs C/D tails
# -----------------------------------------------------------------------------
def fig_mtt_cdf():
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    for k in ORDER:
        v = sorted(col(DATA[k], "total_MTT_ms"))
        v = [x / 1000.0 for x in v]
        if not v:
            continue
        y = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, y,
                label=LABELS[k].replace("\n", " "),
                color=COLORS[k],
                linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("Migration downtime (s)")
    ax.set_ylabel("Empirical CDF")
    ax.set_xlim(1, 200)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax.set_title("CDF of migration downtime")
    save(fig, "fig3_mtt_cdf")


# -----------------------------------------------------------------------------
# Fig 4: Stacked time-breakdown — dump / transfer / restore
# -----------------------------------------------------------------------------
def fig_time_breakdown():
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    xs = np.arange(len(ORDER))
    dump, xfer, rest = [], [], []
    for k in ORDER:
        dump.append(statistics.median(col(DATA[k], "trigger_to_dump_ms"))/1000)
        xfer.append(statistics.median(col(DATA[k], "dump_to_transfer_ms"))/1000)
        rest.append(statistics.median(col(DATA[k], "transfer_to_restore_ms"))/1000)
    bottom = np.zeros(len(ORDER))
    ax.bar(xs, dump, label="Dump (save state)",
           color="#cccccc", edgecolor="black", linewidth=0.5)
    bottom += np.array(dump)
    ax.bar(xs, xfer, bottom=bottom, label="Transfer (cross-node)",
           color="#888888", edgecolor="black", linewidth=0.5, hatch="//")
    bottom += np.array(xfer)
    ax.bar(xs, rest, bottom=bottom, label="Restore (kill+launch+load)",
           color="#444444", edgecolor="black", linewidth=0.5, hatch="xx")
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[k] for k in ORDER])
    ax.set_ylabel("Time (s, median)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    ax.set_title("Where the migration time goes")
    save(fig, "fig4_time_breakdown")


# -----------------------------------------------------------------------------
# Fig 5: Task success rate (pre-migration) — A's learning lead
# -----------------------------------------------------------------------------
def fig_success_rate():
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    xs = np.arange(len(ORDER))
    means, stds = [], []
    for k in ORDER:
        v = col(DATA[k], "success_rate_pre")
        means.append(statistics.mean(v) if v else 0)
        stds.append(statistics.stdev(v) if len(v) > 1 else 0)
    bars = ax.bar(xs, means, yerr=stds, capsize=2,
                  color=[COLORS[k] for k in ORDER],
                  edgecolor="black", linewidth=0.5,
                  hatch=[HATCH[k] for k in ORDER],
                  error_kw=dict(elinewidth=0.6, ecolor="black"))
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[k] for k in ORDER])
    ax.set_ylabel("Task success rate")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.6)
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width()/2, means[i] + 0.04,
                f"{means[i]:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("Task success rate per condition (mean ± std)")
    save(fig, "fig5_success_rate")


# -----------------------------------------------------------------------------
# Fig 6: Box plot of MTT distributions per condition.
# Median + quartiles + whiskers + outliers in one shot. A's tight box next to
# C/D's tall whiskers tells the speed + predictability story without log axes.
# -----------------------------------------------------------------------------
def fig_mtt_boxplot():
    fig, ax = plt.subplots(figsize=(W1, 2.8))
    data = []
    for k in ORDER:
        v = [x / 1000 for x in col(DATA[k], "total_MTT_ms")]
        data.append(v)

    bp = ax.boxplot(
        data,
        labels=[LABELS[k] for k in ORDER],
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(linewidth=0.7),
        capprops=dict(linewidth=0.7),
        flierprops=dict(marker="o", markersize=3,
                        markerfacecolor="black", markeredgecolor="black",
                        alpha=0.55),
    )
    for patch, k in zip(bp["boxes"], ORDER):
        patch.set_facecolor(COLORS[k])
        patch.set_edgecolor("black")
        patch.set_linewidth(0.6)
        patch.set_alpha(0.85)
        patch.set_hatch(HATCH[k])

    ax.set_ylabel("Migration downtime per event (s)")
    ax.set_yscale("log")
    ax.set_ylim(1, 300)
    ax.set_title("Distribution of migration downtime (n≈100 per condition)")

    # Annotate medians on top of each box.
    for i, v in enumerate(data, start=1):
        if not v:
            continue
        med = statistics.median(v)
        ax.text(i, med * 1.08, f"{med:.1f}s",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

    save(fig, "fig6_mtt_boxplot")


# -----------------------------------------------------------------------------
# Fig 7: Fleet-scale projection — extrapolate per-migration cost to N robots.
# Assumes 5 forced migrations per robot (same as the experiment). Plots total
# fleet bandwidth (GB) and total fleet downtime (robot-hours) as fleet size
# grows from 10 → 10 000. Reads as a "scales-to-production" argument.
# -----------------------------------------------------------------------------
def fig_fleet_projection():
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.9))
    fleet_sizes = np.array([10, 100, 1_000, 10_000])
    mig_per_robot = 5

    # Per-condition per-migration cost (mean).
    per_mig_bytes = {}
    per_mig_mtt_s = {}
    for k in ORDER:
        net_mb = [x / (1024 * 1024)
                  for x in col(DATA[k], "network_bytes_transferred")]
        per_mig_bytes[k] = statistics.mean(net_mb) if net_mb else 0
        per_mig_mtt_s[k] = statistics.mean(col(DATA[k], "total_MTT_ms")) / 1000

    # Panel A: total fleet bandwidth (GB).
    axL = axes[0]
    for k in ORDER:
        bw_gb = fleet_sizes * mig_per_robot * per_mig_bytes[k] / 1024.0  # MB→GB
        bw_gb = np.maximum(bw_gb, 1e-5)
        axL.plot(fleet_sizes, bw_gb,
                 marker="o" if k == "A" else "s",
                 markersize=4,
                 linewidth=1.4 if k == "A" else 0.9,
                 color=COLORS[k],
                 label=LABELS[k].replace("\n", " "))
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel("Fleet size (robots)")
    axL.set_ylabel("Total bandwidth per migration wave (GB)")
    axL.set_title("Fleet bandwidth scales with N")
    axL.set_xlim(8, 13_000)

    # Panel B: aggregate robot-hours of downtime (assuming serial migrations
    # per robot, parallel across robots — i.e. each robot loses mtt × 5 seconds).
    axR = axes[1]
    for k in ORDER:
        hours = fleet_sizes * mig_per_robot * per_mig_mtt_s[k] / 3600.0
        hours = np.maximum(hours, 1e-5)
        axR.plot(fleet_sizes, hours,
                 marker="o" if k == "A" else "s",
                 markersize=4,
                 linewidth=1.4 if k == "A" else 0.9,
                 color=COLORS[k],
                 label=LABELS[k].replace("\n", " "))
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xlabel("Fleet size (robots)")
    axR.set_ylabel("Total robot-hours lost to migration")
    axR.set_title("Aggregate downtime scales with N")
    axR.set_xlim(8, 13_000)

    axR.legend(loc="upper left", frameon=True, framealpha=0.95,
               fontsize=7, handlelength=1.5)
    save(fig, "fig7_fleet_projection")


# -----------------------------------------------------------------------------
# Fig 8: Cumulative cost over the actual 20-robot run.
# Sort each condition's migration events by timestamp; plot cumulative
# bandwidth (MB) and cumulative downtime (s) vs migration event index.
# Diverging lines show the operational saving as the experiment progresses.
# -----------------------------------------------------------------------------
def fig_cumulative_cost():
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.9))

    for k in ORDER:
        rows = sorted(DATA[k], key=lambda r: float(r["timestamp"]))
        idx  = np.arange(1, len(rows) + 1)
        cum_mb = np.cumsum([float(r["network_bytes_transferred"]) / (1024 * 1024)
                            for r in rows])
        cum_s  = np.cumsum([float(r["total_MTT_ms"]) / 1000 for r in rows])

        axes[0].plot(idx, cum_mb,
                     color=COLORS[k],
                     linewidth=1.6 if k == "A" else 1.0,
                     label=LABELS[k].replace("\n", " "))
        axes[1].plot(idx, cum_s,
                     color=COLORS[k],
                     linewidth=1.6 if k == "A" else 1.0,
                     label=LABELS[k].replace("\n", " "))

    axes[0].set_xlabel("Migration event #")
    axes[0].set_ylabel("Cumulative bandwidth (MB)")
    axes[0].set_title("Bandwidth consumed by experiment")
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.1, 5000)

    axes[1].set_xlabel("Migration event #")
    axes[1].set_ylabel("Cumulative robot downtime (s)")
    axes[1].set_title("Downtime accumulated by experiment")

    axes[1].legend(loc="upper left", frameon=True, framealpha=0.95,
                   fontsize=7, handlelength=1.5)
    save(fig, "fig8_cumulative_cost")


# -----------------------------------------------------------------------------
# Fig 9: Post-migration recovery — tasks needed to return to pre-migration
# success rate. The behavioral-continuity metric (advisor request). Lower is
# better; A should be near 0 (policy survives), C/D higher, E highest (cold
# restart reverts to the pretrained policy and must relearn).
# -----------------------------------------------------------------------------
def fig_recovery_curve():
    xs = np.arange(len(ORDER))
    meds, present = [], []
    for k in ORDER:
        # -1 = "never recovered within the probe window"; treat as the window
        # cap so the bar is visible and honestly tall.
        v = [x for x in col(DATA[k], "recovery_tasks_to_pre")]
        v = [(80 if x < 0 else x) for x in v]   # cap == max_tasks in the probe
        if v:
            meds.append(statistics.median(v)); present.append(k)
        else:
            meds.append(0)
    if not present:
        print("  skip fig9_recovery_curve (no recovery_tasks_to_pre data yet)")
        return
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    bars = ax.bar(xs, meds,
                  color=[COLORS[k] for k in ORDER],
                  edgecolor="black", linewidth=0.5,
                  hatch=[HATCH[k] for k in ORDER])
    ax.set_xticks(xs); ax.set_xticklabels([LABELS[k] for k in ORDER])
    ax.set_ylabel("Tasks to recover pre-migration\nsuccess rate (median)")
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width()/2, meds[i] + 0.5,
                f"{meds[i]:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("Post-migration recovery (lower = better)")
    save(fig, "fig9_recovery_curve")


# -----------------------------------------------------------------------------
# Fig 10: Concurrency scaling — median migration downtime vs number of
# simultaneous migrations, for each mechanism (Condition F result dirs:
# concurrent_{cold,warm}_c{N}). DHT (Condition A run with MIGRATION_CONCURRENCY)
# folders named concurrent_dht_c{N} are picked up too if present. Flat = good.
# -----------------------------------------------------------------------------
def fig_concurrency_scaling():
    import re
    mech_series = {}   # mechanism -> {level: median_downtime_s}
    for p in sorted(RESULTS.glob("concurrent_*_c*")):
        m = re.match(r"concurrent_(\w+)_c(\d+)$", p.name)
        if not m:
            continue
        mech, level = m.group(1), int(m.group(2))
        rows = load_folder(p.name)
        v = [float(r["total_MTT_ms"]) / 1000 for r in rows
             if r.get("total_MTT_ms")]
        if v:
            mech_series.setdefault(mech, {})[level] = statistics.median(v)
    if not mech_series:
        print("  skip fig10_concurrency_scaling (no Condition F data yet)")
        return
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    style = {"dht": ("#1f3a93", "o"), "cold": ("#a04000", "s"),
             "warm": ("#d68910", "^")}
    for mech, series in sorted(mech_series.items()):
        levels = sorted(series)
        color, marker = style.get(mech, ("#444444", "d"))
        ax.plot(levels, [series[l] for l in levels],
                marker=marker, markersize=4, color=color,
                linewidth=1.4 if mech == "dht" else 1.0,
                label=mech.upper())
    ax.set_xlabel("Concurrent migrations")
    ax.set_ylabel("Median migration downtime (s)")
    ax.set_title("Downtime vs migration concurrency")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    save(fig, "fig10_concurrency_scaling")


# -----------------------------------------------------------------------------
# Fig 11: Failure recovery — fault-injected migration cost (Condition G,
# failure_injection dir) vs the normal cold baseline (C). Shows the extra
# downtime the checkpoint mechanism pays when a destination dies mid-migration.
# -----------------------------------------------------------------------------
def fig_failure_recovery():
    g = load_folder("failure_injection")
    if not g:
        print("  skip fig11_failure_recovery (no Condition G data yet)")
        return
    # Normal cold baseline downtime vs faulted-event total recovery.
    c_norm = [float(r["total_MTT_ms"]) / 1000 for r in DATA["C"]
              if r.get("total_MTT_ms")]
    g_recov = [float(r["total_recovery_ms"]) / 1000 for r in g
               if r.get("total_recovery_ms") and float(r.get("fault_injected", 0)) == 1]
    if not g_recov:
        print("  skip fig11_failure_recovery (no faulted events recorded)")
        return
    labels = ["C: normal\nmigration", "G: fault +\nretry"]
    vals   = [statistics.median(c_norm) if c_norm else 0,
              statistics.median(g_recov)]
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    bars = ax.bar([0, 1], vals, color=["#a04000", "#7a1f1f"],
                  edgecolor="black", linewidth=0.5, hatch=["xx", "++"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_ylabel("Median downtime / recovery (s)")
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width()/2, vals[i] * 1.02,
                f"{vals[i]:.1f}s", ha="center", va="bottom", fontsize=7)
    ax.set_title("Cost of a destination failure mid-migration")
    save(fig, "fig11_failure_recovery")


# -----------------------------------------------------------------------------
# Fig 12: Condition D bandwidth trade-off — migration-window bytes (delta
# rsync) vs the continuous pre-copy bandwidth paid during normal operation.
# Makes D's "spend background bandwidth to shrink the migration window" cost
# explicit, and shows A pays neither.
# -----------------------------------------------------------------------------
def fig_d_bandwidth_tradeoff():
    bg = col(DATA["D"], "background_bandwidth_mb")
    if not bg or max(bg) == 0:
        print("  skip fig12_d_bandwidth_tradeoff (no background_bandwidth_mb yet)")
        return
    mig_mb = [x / (1024 * 1024) for x in col(DATA["D"], "network_bytes_transferred")]
    a_mig  = [x / (1024 * 1024) for x in col(DATA["A"], "network_bytes_transferred")]
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    xs = [0, 1]
    mig_window = [statistics.mean(a_mig) if a_mig else 0,
                  statistics.mean(mig_mb) if mig_mb else 0]
    background = [0, statistics.mean(bg)]
    ax.bar(xs, mig_window, color="#888888", edgecolor="black",
           linewidth=0.5, label="Migration-window bytes")
    ax.bar(xs, background, bottom=mig_window, color="#d68910",
           edgecolor="black", linewidth=0.5, hatch="\\\\",
           label="Continuous pre-copy bytes")
    ax.set_xticks(xs)
    ax.set_xticklabels(["A. DHT+FRL", "D. App warm\ncheckpoint"])
    ax.set_ylabel("Mean bandwidth per migration (MB)")
    ax.set_yscale("log")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    ax.set_title("D's pre-copy bandwidth trade-off")
    save(fig, "fig12_d_bandwidth_tradeoff")


def _unused_old_pareto():
    # Pull A's reference values.
    A_mtt = statistics.median(col(DATA["A"], "total_MTT_ms"))
    A_net = statistics.mean(
        [x / (1024 * 1024) for x in col(DATA["A"], "network_bytes_transferred")]
    )
    A_sr  = statistics.mean(col(DATA["A"], "success_rate_pre"))

    baselines = ["C", "D", "E"]
    metric_names  = ["Migration speed", "Bandwidth efficiency", "Learning quality"]
    metric_colors = ["#1f3a93", "#a04000", "#117a65"]
    metric_hatch  = ["", "//", "xx"]

    ratios = {m: [] for m in metric_names}
    for b in baselines:
        b_mtt = statistics.median(col(DATA[b], "total_MTT_ms"))
        b_net_vals = [x / (1024 * 1024)
                      for x in col(DATA[b], "network_bytes_transferred")]
        b_net = statistics.mean(b_net_vals) if b_net_vals else 0
        b_sr  = statistics.mean(col(DATA[b], "success_rate_pre"))
        # Floor near-zero values so the ratio is bounded and the bar is visible.
        ratios["Migration speed"].append(max(b_mtt / max(A_mtt, 1), 0.1))
        # Bandwidth: floor A's bytes to a small positive so B/E (~0 MB) don't
        # explode. Use 0.05 MB as the practical floor for "negligible".
        ratios["Bandwidth efficiency"].append(
            max(b_net / max(A_net, 0.05), 0.1)
        )
        ratios["Learning quality"].append(max(A_sr / max(b_sr, 0.01), 0.1))

    fig, ax = plt.subplots(figsize=(W2, 3.2))
    n_groups = len(baselines)
    n_bars   = len(metric_names)
    bar_w    = 0.25
    xs       = np.arange(n_groups)

    for i, m in enumerate(metric_names):
        offset = (i - (n_bars - 1) / 2) * bar_w
        vals = ratios[m]
        bars = ax.bar(xs + offset, vals, bar_w,
                      label=m,
                      color=metric_colors[i],
                      edgecolor="black", linewidth=0.5,
                      hatch=metric_hatch[i])
        for b, v in zip(bars, vals):
            if v >= 1.0:
                txt = f"{v:.1f}×"
            else:
                txt = f"{v:.2f}×"
            ax.text(b.get_x() + b.get_width()/2, v * 1.12,
                    txt, ha="center", va="bottom", fontsize=7.5)

    # Reference line: 1.0× = parity with A.
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.text(n_groups - 0.5, 1.05, "parity with A (1.0×)",
            ha="right", va="bottom", fontsize=7,
            fontstyle="italic", color="black")

    ax.set_yscale("log")
    ax.set_ylim(0.3, 400)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"vs {b}.\n{LABELS[b].split('. ')[1].replace(chr(10),' ')}"
                        for b in baselines])
    ax.set_ylabel("Improvement factor of A over baseline (×, log scale)")
    ax.set_title("How many times better is the proposed DHT+FRL design (A)?")
    ax.legend(loc="upper right", ncol=1, frameon=True, framealpha=0.95)

    # Annotate the "ours is better" region.
    ax.axhspan(1.0, 400, color="#d6f5d6", alpha=0.25, zorder=0)
    ax.text(-0.4, 200, "A wins ↑", color="#1e6e1e",
            fontsize=8, fontstyle="italic", fontweight="bold")
    ax.text(-0.4, 0.45, "A loses ↓", color="#8b1a1a",
            fontsize=8, fontstyle="italic", fontweight="bold")

    save(fig, "fig6_pareto")
    fig, ax = plt.subplots(figsize=(W2, 3.4))

    # Shade the "ideal" region (low downtime, low bytes) so the story reads
    # without needing to interpret the axes.
    ax.axvspan(0.8, 6, ymin=0, ymax=1, color="#d6f5d6", alpha=0.5, zorder=0)
    ax.text(1.0, 50, "ideal region\n(low downtime,\nlow bandwidth)",
            color="#1e6e1e", fontsize=8, fontstyle="italic",
            ha="left", va="top")

    # Plot each condition.
    xs, ys, sizes, names = [], [], [], []
    for k in ORDER:
        mtt = statistics.median(col(DATA[k], "total_MTT_ms")) / 1000
        net_mb_vals = [x / (1024 * 1024)
                       for x in col(DATA[k], "network_bytes_transferred")]
        net = statistics.mean(net_mb_vals) if net_mb_vals else 0
        sr  = statistics.mean(col(DATA[k], "success_rate_pre"))
        # Bubble size proportional to success-rate squared so the visual
        # contrast between 0.45 and 0.87 reads at a glance.
        bubble = 60 + (sr ** 2) * 1800
        x = max(mtt, 0.5)
        y = max(net, 0.008)
        ax.scatter(x, y, s=bubble,
                   color=COLORS[k],
                   edgecolor="black", linewidth=0.8,
                   alpha=0.85, zorder=3)
        xs.append(x); ys.append(y); sizes.append(bubble); names.append(k)

    # Smart label placement so they don't overlap their bubbles.
    offsets = {
        "A": (14, -8),    # to the right
        "B": (12, 10),    # upper right
        "C": (-22, 10),   # upper left
        "D": (-22, 10),
        "E": (14, -2),
    }
    for k, x, y in zip(names, xs, ys):
        dx, dy = offsets.get(k, (10, 6))
        sr = statistics.mean(col(DATA[k], "success_rate_pre"))
        ax.annotate(f"{k}  sr={sr:.2f}", (x, y),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    fontsize=9, fontweight="bold",
                    color=COLORS[k])

    # Arrow pointing toward the desirable corner.
    ax.annotate("", xy=(1.1, 0.012), xytext=(6, 0.05),
                arrowprops=dict(arrowstyle="->", color="#1e6e1e", lw=1.2))
    ax.text(2.4, 0.018, "better", color="#1e6e1e",
            fontsize=8, fontstyle="italic")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Migration downtime (s, median)")
    ax.set_ylabel("Network bytes per migration (MB, mean)")
    ax.set_xlim(0.8, 200)
    ax.set_ylim(0.008, 100)
    ax.set_title("Downtime vs network cost vs learned quality "
                 "(bubble area ∝ task success rate)")

    # Legend explaining bubble-size encoding.
    sr_examples = [0.45, 0.65, 0.87]
    handles = [plt.scatter([], [], s=60 + (sr ** 2) * 1800,
                           color="lightgray", edgecolor="black",
                           linewidth=0.6, label=f"sr = {sr:.2f}")
               for sr in sr_examples]
    ax.legend(handles=handles, loc="upper right",
              title="Task success rate",
              frameon=True, framealpha=0.95,
              labelspacing=1.4, borderpad=0.8)
    save(fig, "fig6_pareto")


if __name__ == "__main__":
    print("Generating figures…")
    fig_mtt_bar()
    fig_network_bytes()
    fig_mtt_cdf()
    fig_time_breakdown()
    fig_success_rate()
    fig_mtt_boxplot()
    fig_fleet_projection()
    fig_cumulative_cost()
    # New ATC-revision figures (skip gracefully if the data isn't present yet).
    fig_recovery_curve()
    fig_concurrency_scaling()
    fig_failure_recovery()
    fig_d_bandwidth_tradeoff()
    print(f"\nAll figures written to: {OUTDIR}")
