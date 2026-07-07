#!/usr/bin/env python3
"""IEEE-style figures for the cluster experiment (Conditions A/C/D/E).

Reads:
  cluster/results/condition_{A,C,D,E}/migration_events.csv
  cluster/results/condition_A_dht_frl/fl_convergence.csv

Writes (next to this script) PDF + PNG for:
  fig1_mtt_cdf            — migration-time CDF (tail / predictability)
  fig2_phase_breakdown    — stacked bar of dump/transfer/restore/load
  fig3_downtime           — per-migration downtime (median + IQR)
  fig4_slo_compliance     — % of migrations completing within 5/10/30 s budgets
  fig5_lost_work_estimate — downtime x task-rate (regression proxy)
  fig6_worst_case_mtt     — max observed migration time per condition

Run:  python make_figs.py
"""
import csv
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# IEEE-paper rcParams. Times-like serif, small fonts, embedded TrueType so the
# PDFs pass IEEE PDF eXpress (Type-3 fonts are rejected).
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        9,
    "axes.labelsize":   9,
    "axes.titlesize":   9.5,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  7.5,
    "lines.linewidth":  1.3,
    "axes.linewidth":   0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "axes.grid":        True,
    "grid.linestyle":   ":",
    "grid.linewidth":   0.4,
    "grid.alpha":       0.6,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

HERE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))

# (tag, label, dir, color, marker, linestyle, hatch)
CONDS = [
    ("A", "DHT-FRL",      "condition_A_dht_frl",      "#1f77b4", "o", "-",  ""),
    ("C", "CRIU cold",    "condition_C_criu_cold",    "#d62728", "s", "--", "///"),
    ("D", "CRIU warm",    "condition_D_criu_warm",    "#2ca02c", "^", "-.", "\\\\\\"),
    ("E", "Cold restart", "condition_E_cold_restart", "#7f7f7f", "x", ":",  "xxx"),
]
COL_WIDTH = 3.5   # IEEE single-column width in inches


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load(cond_dir):
    p = os.path.join(RESULTS, cond_dir, "migration_events.csv")
    if not os.path.exists(p):
        print(f"  [warn] missing: {p}")
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def col(rows, c, cap=None):
    vals = [num(r.get(c)) for r in rows]
    vals = [v for v in vals if v is not None]
    if cap is not None:
        vals = [v for v in vals if v <= cap]
    return vals


def save(fig, name):
    pdf = os.path.join(HERE, f"{name}.pdf")
    png = os.path.join(HERE, f"{name}.png")
    fig.savefig(pdf)
    fig.savefig(png)
    print(f"  wrote {name}.{{pdf,png}}")
    plt.close(fig)


# Load all four CSVs once.
DATA = {tag: load(d) for tag, _, d, *_ in CONDS}


# --------------------------------------------------------------------------- #
# Fig 1 — migration-time CDF (tail / predictability)                           #
# --------------------------------------------------------------------------- #
def fig1():
    # Arial + larger fonts for this figure only; drop the Cold-restart condition.
    fig1_rc = {
        "font.family":     "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":       12,
        "axes.labelsize":  12,
        "axes.titlesize":  12.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
    }
    conds = [c for c in CONDS if c[0] != "E"]
    with plt.rc_context(fig1_rc):
        fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.4))
        for tag, label, _, color, _, ls, _ in conds:
            vals = sorted(col(DATA[tag], "total_MTT_ms", cap=60000))
            if not vals:
                continue
            y = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(np.array(vals) / 1000.0, y, label=label,
                    color=color, linestyle=ls, linewidth=1.4)
        ax.set_xscale("log")
        ax.set_xlim(1.5, 60)
        ax.set_ylim(0, 1.005)
        ax.set_xlabel("Migration time (s, log scale)")
        ax.set_ylabel("CDF")
        ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="0.4")
        save(fig, "fig1_mtt_cdf")


# --------------------------------------------------------------------------- #
# Fig 2 — MTT phase breakdown (stacked horizontal bar)                         #
# --------------------------------------------------------------------------- #
def fig2():
    phases = [
        ("Dump",     "trigger_to_dump_ms",     "#9ecae1"),
        ("Transfer", "dump_to_transfer_ms",    "#1f77b4"),
        ("Restore",  "transfer_to_restore_ms", "#d62728"),
        ("Load",     "policy_load_ms",         "#2ca02c"),
    ]
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.2))
    y_pos = np.arange(len(CONDS))
    left = np.zeros(len(CONDS))

    for pname, pcol, pcolor in phases:
        vals = np.array([
            (st.median(col(DATA[tag], pcol)) / 1000.0) if col(DATA[tag], pcol) else 0
            for tag, *_ in CONDS
        ])
        ax.barh(y_pos, vals, left=left, color=pcolor,
                edgecolor="black", linewidth=0.5, label=pname)
        left += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels([c[1] for c in CONDS])
    ax.invert_yaxis()
    ax.set_xlabel("Median time per phase (s)")
    ax.legend(loc="lower right", ncol=2, frameon=True,
              fancybox=False, edgecolor="0.4")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.grid(axis="y", visible=False)
    save(fig, "fig2_phase_breakdown")


# --------------------------------------------------------------------------- #
# Fig 3 — per-migration downtime (median + IQR)                                #
# --------------------------------------------------------------------------- #
def fig3():
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.2))
    labels  = [c[1] for c in CONDS]
    colors  = [c[3] for c in CONDS]
    hatches = [c[6] for c in CONDS]

    meds, lo, hi = [], [], []
    for tag, *_ in CONDS:
        v = np.array(col(DATA[tag], "downtime_ms", cap=60000)) / 1000.0
        if v.size == 0:
            meds.append(0); lo.append(0); hi.append(0); continue
        m = np.median(v)
        meds.append(m)
        lo.append(m - np.percentile(v, 25))
        hi.append(np.percentile(v, 75) - m)

    x = np.arange(len(labels))
    bars = ax.bar(x, meds, yerr=[lo, hi], capsize=3,
                  color=colors, edgecolor="black", linewidth=0.6)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Downtime per migration (s)")
    ax.set_title("Robot offline time per migration (median, IQR)")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.grid(axis="x", visible=False)
    for xi, m in zip(x, meds):
        ax.text(xi, m + 0.15, f"{m:.2f}", ha="center", va="bottom", fontsize=7.5)
    save(fig, "fig3_downtime")


# --------------------------------------------------------------------------- #
# Fig 4 — SLO compliance: % of migrations within {5, 10, 30} s budgets         #
# --------------------------------------------------------------------------- #
def fig4():
    # Only migration mechanisms; drop condition E (cold restart).
    conds = [c for c in CONDS if c[0] != "E"]
    budgets_s = [5, 10, 30]
    # Match the fig2_v2 (Vega/Tableau) palette used elsewhere in the paper.
    slo_colors = {"A": "#4c78a8", "C": "#e45756", "D": "#72b7b2"}
    # Arial + larger fonts to match fig2b_stacked_budget.
    fig4_rc = {
        "font.family":     "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":       10,
        "axes.labelsize":  10,
        "axes.titlesize":  10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        # Keep the legend at the original compact size so it tucks into the
        # bottom-right corner instead of overlapping the bars.
        "legend.fontsize": 6.5,
    }
    with plt.rc_context(fig4_rc):
        _fig4_body(conds, budgets_s, slo_colors)


def _fig4_body(conds, budgets_s, slo_colors):
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.3))

    n_cond = len(conds)
    n_bud  = len(budgets_s)
    bar_w  = 0.78 / n_cond
    x_base = np.arange(n_bud)

    for i, (tag, label, _, _default_color, _, _, hatch) in enumerate(conds):
        color = slo_colors.get(tag, _default_color)
        vals = col(DATA[tag], "total_MTT_ms")
        if not vals:
            continue
        v = np.array(vals)
        pcts = [100.0 * np.mean(v <= b * 1000.0) for b in budgets_s]
        x = x_base + (i - (n_cond - 1) / 2.0) * bar_w
        bars = ax.bar(x, pcts, bar_w, color=color, edgecolor="black",
                      linewidth=0.5, label=label)
        for bar in bars:
            bar.set_hatch(hatch)

    ax.set_xticks(x_base)
    ax.set_xticklabels([f"{b} s" for b in budgets_s])
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel("Migration-time budget (SLO)")
    ax.set_ylabel("Meeting budget (%)")
    ax.set_ylim(0, 108)
    ax.legend(loc="lower right", ncol=2, frameon=True,
              fancybox=False, edgecolor="0.4")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.grid(axis="x", visible=False)
    save(fig, "fig4_slo_compliance")


# --------------------------------------------------------------------------- #
# Fig 5 — lost-work estimate (downtime x task rate)                            #
# --------------------------------------------------------------------------- #
def fig5():
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.2))
    labels  = [c[1] for c in CONDS]
    colors  = [c[3] for c in CONDS]
    hatches = [c[6] for c in CONDS]

    meds = []
    for tag, *_ in CONDS:
        lw = []
        for r in DATA[tag]:
            dt   = num(r.get("downtime_ms"))
            rate = num(r.get("throughput_post_60s"))  # tasks per minute
            if dt is None or rate is None:
                continue
            if dt > 100000 or rate <= 0 or rate > 1000:
                continue
            lw.append(dt / 60000.0 * rate)  # tasks lost during the gap
        meds.append(np.median(lw) if lw else 0)

    x = np.arange(len(labels))
    bars = ax.bar(x, meds, color=colors, edgecolor="black", linewidth=0.6)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Estimated tasks lost per migration")
    ax.set_title("Productivity cost per migration (downtime $\\times$ task rate)")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.grid(axis="x", visible=False)
    for xi, m in zip(x, meds):
        ax.text(xi, m + max(meds) * 0.02, f"{m:.1f}",
                ha="center", va="bottom", fontsize=7.5)
    save(fig, "fig5_lost_work_estimate")


# --------------------------------------------------------------------------- #
# Fig 6 — worst-case (max) migration time per condition                        #
# --------------------------------------------------------------------------- #
def fig6():
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.2))
    labels  = [c[1] for c in CONDS]
    colors  = [c[3] for c in CONDS]
    hatches = [c[6] for c in CONDS]

    maxes = []
    for tag, *_ in CONDS:
        v = col(DATA[tag], "total_MTT_ms")
        maxes.append((max(v) / 1000.0) if v else 0)

    x = np.arange(len(labels))
    bars = ax.bar(x, maxes, color=colors, edgecolor="black", linewidth=0.6)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Worst-case migration time (s)")
    ax.set_title("Maximum observed migration time per condition")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.grid(axis="x", visible=False)
    for xi, m in zip(x, maxes):
        ax.text(xi, m + max(maxes) * 0.02, f"{m:.1f}",
                ha="center", va="bottom", fontsize=7.5)
    save(fig, "fig6_worst_case_mtt")


if __name__ == "__main__":
    print(f"Generating IEEE-style figures into {HERE}")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print("Done.")
