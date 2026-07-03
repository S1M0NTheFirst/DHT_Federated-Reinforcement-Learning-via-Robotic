#!/usr/bin/env python3
"""Three candidate designs for Fig 2 (per-phase migration breakdown).

Reads cluster/results/condition_{A,C,D}/migration_events.csv and writes, next
to this script:
  fig2a_grouped_log      - grouped horizontal bars, log time axis (every phase visible)
  fig2b_stacked_budget   - stacked linear "time budget" bars
  fig2c_small_multiples  - one mini panel per condition, log time axis

Phases: dump, transfer, restore, load, and (DHT-FRL only) a resume/overlay-join
phase equal to the stable MTT residual not captured by the four logged columns.

Arial, larger fonts, embedded TrueType (IEEE PDF eXpress safe).
"""
import csv
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":        12,
    "axes.labelsize":   12,
    "axes.titlesize":   12.5,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  10,
    "axes.linewidth":   0.8,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

HERE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))

# (tag, label, dir, condition-color)
CONDS = [
    ("A", "DHT-FRL",   "condition_A_dht_frl",   "#3b6fb6"),
    ("C", "CRIU cold", "condition_C_criu_cold", "#d1615d"),
    ("D", "CRIU warm", "condition_D_criu_warm", "#e2a63b"),
]

# (phase label, column, phase-color)  -- resume is derived, not a column
PHASES = [
    ("Dump",     "trigger_to_dump_ms",     "#8c9cb0"),
    ("Transfer", "dump_to_transfer_ms",    "#4c78a8"),
    ("Restore",  "transfer_to_restore_ms", "#e45756"),
    ("Load",     "policy_load_ms",         "#72b7b2"),
    ("Resume",   "_resume_ms",             "#f2a641"),  # overlay join (DHT-FRL)
]
PHASE_COLS = [p[1] for p in PHASES]


def load(cond_dir):
    p = os.path.join(RESULTS, cond_dir, "migration_events.csv")
    return list(csv.DictReader(open(p, newline="")))


def median_phases(rows):
    """Return {col: median_ms} incl. derived _resume_ms (MTT minus 4 phases)."""
    def med(c):
        v = [float(r[c]) for r in rows if r.get(c) not in (None, "")]
        return st.median(v) if v else 0.0
    out = {c: med(c) for c in PHASE_COLS[:-1]}
    residual = [
        float(r["total_MTT_ms"]) - sum(float(r[c]) for c in PHASE_COLS[:-1])
        for r in rows
    ]
    out["_resume_ms"] = max(0.0, st.median(residual))
    return out


DATA = {tag: median_phases(load(d)) for tag, _, d, _ in CONDS}
# True total = sum of the (median) phases, in seconds, for honest bar-end labels.
DATA_TOTAL = {tag: sum(DATA[tag][c] for c in PHASE_COLS) / 1000.0
              for tag, *_ in CONDS}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {name}.{{pdf,png}}")


# --------------------------------------------------------------------------- #
# Variant A - grouped horizontal bars on a log time axis                      #
# every phase is legible even across ~5 orders of magnitude                   #
# --------------------------------------------------------------------------- #
def variant_a():
    FLOOR = 0.05  # ms; log axis cannot render 0
    n_c = len(CONDS)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y_phase = np.arange(len(PHASES))[::-1]          # dump at top
    bar_h = 0.82 / n_c

    def fmt(ms):
        return f"{ms/1000:.1f}s" if ms >= 1000 else f"{ms:.0f}ms" if ms >= 1 \
            else f"{ms:.1f}ms"

    for i, (tag, label, _, color) in enumerate(CONDS):
        offs = ((n_c - 1) / 2.0 - i) * bar_h        # DHT-FRL on top of each group
        for j, (pname, pcol, _) in enumerate(PHASES):
            val = DATA[tag][pcol]
            y = y_phase[j] + offs
            if val < FLOOR:
                # structurally absent (no restore for DHT-FRL, no resume for CRIU)
                ax.plot(FLOOR * 1.4, y, marker="|", color=color, ms=8, mew=1.6)
                ax.text(FLOOR * 2.1, y, "none", va="center", ha="left",
                        fontsize=7.5, color=color, style="italic")
                continue
            ax.barh(y, val, height=bar_h, left=FLOOR, color=color,
                    edgecolor="black", linewidth=0.4,
                    label=label if j == 0 else None)
            # Value label: outside the bar end, but inside if it would collide
            # with the legend box (upper-right region).
            if val > 700 and y < 1.0:      # Resume DHT-FRL bar, near legend
                ax.text(val / 1.5, y, fmt(val), va="center", ha="right",
                        fontsize=7.5, color="white")
            else:
                ax.text(val * 1.35, y, fmt(val), va="center", ha="left",
                        fontsize=7.5, color="0.25")

    # Shade the two phases that decide the outcome.
    for j, (pname, *_ ) in enumerate(PHASES):
        if pname in ("Restore", "Resume"):
            ax.axhspan(y_phase[j] - 0.45, y_phase[j] + 0.45,
                       color="0.92", zorder=0)

    ax.set_xscale("log")
    ax.set_xlim(FLOOR, 60000)
    ax.set_yticks(y_phase)
    ax.set_yticklabels([p[0] for p in PHASES], fontweight="bold")
    ax.set_xlabel("Median time per phase (ms, log scale)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3,
              frameon=False, columnspacing=1.4, handlelength=1.3)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("CRIU migration is dominated by Restore (shaded); "
                 "DHT-FRL skips it entirely", fontsize=10, pad=26)
    save(fig, "fig2a_grouped_log")


# --------------------------------------------------------------------------- #
# Variant B - stacked linear "time budget" bars                               #
# keeps the where-does-the-time-go budget story; dump/load noted as tiny      #
# --------------------------------------------------------------------------- #
def variant_b():
    # Big phases (Transfer/Restore/Resume) are drawn to true linear scale.
    # Sub-second phases (Dump/Load) would be invisible, so their *width* is
    # log-scaled: still monotonic in the real value (bigger value -> wider
    # section) so the per-method difference is visible, but hatched + labelled
    # so it is clearly not the linear time scale.
    TINY = {"Dump", "Load"}
    BASE, SCALE = 0.10, 0.17   # drawn seconds = BASE + SCALE*log10(ms+1)

    def drawn_width(pname, val_s):
        if pname in TINY:
            return BASE + SCALE * np.log10(val_s * 1000.0 + 1.0)
        return val_s

    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    y = np.arange(len(CONDS))[::-1]
    bar_h = 0.55

    for k, (tag, label, _, _) in enumerate(CONDS):
        left = 0.0
        for pname, pcol, pcolor in PHASES:
            val = DATA[tag][pcol] / 1000.0  # true seconds
            if val <= 0:
                continue
            w = drawn_width(pname, val)
            ax.barh(y[k], w, height=bar_h, left=left, color=pcolor,
                    edgecolor="black", linewidth=0.6,
                    hatch="////" if pname in TINY else None,
                    label=pname if k == 0 else None)
            if pname in TINY:
                # true value labelled above the section (ms)
                ms = val * 1000.0
                vtxt = f"{ms:.1f}" if ms < 1 else f"{ms:.0f}"
                # nudge the leftmost (Dump) label right so it stays in-box
                lx = left + w / 2 + (0.22 if pname == "Dump" else 0.0)
                ax.text(lx, y[k] + bar_h / 2 + 0.05, f"{vtxt}ms",
                        ha="center", va="bottom",
                        fontsize=17 if pname == "Dump" else 14, color="0.25")
            elif w >= 0.5:
                txtcol = "white" if pname in ("Transfer", "Restore") else "black"
                ax.text(left + w / 2, y[k], f"{pname}\n{val:.1f}s", ha="center",
                        va="center", fontsize=14.5, color=txtcol, linespacing=0.95)
            left += w
        ax.text(left + 0.12, y[k], f"total {DATA_TOTAL[tag]:.1f} s", ha="left",
                va="center", fontsize=16, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([c[1] for c in CONDS], fontweight="bold", fontsize=17)
    ax.set_xlabel("Time per migration (s)", fontsize=17)
    ax.tick_params(axis="x", labelsize=15)
    ax.set_xlim(0, 8.2)
    ax.set_ylim(-0.7, len(CONDS) - 0.05)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=5,
              frameon=False, columnspacing=1.1, handlelength=1.1,
              handletextpad=0.5, fontsize=18)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig2b_stacked_budget")


# --------------------------------------------------------------------------- #
# Variant C - small multiples: one panel per condition, shared log axis       #
# --------------------------------------------------------------------------- #
def variant_c():
    FLOOR = 0.05
    fig, axes = plt.subplots(1, len(CONDS), figsize=(7.4, 2.5), sharex=True)
    y_phase = np.arange(len(PHASES))[::-1]

    for ax, (tag, label, _, ccolor) in zip(axes, CONDS):
        for j, (pname, pcol, pcolor) in enumerate(PHASES):
            val = DATA[tag][pcol]
            if val < FLOOR:
                ax.plot(FLOOR, y_phase[j], marker="|", color="0.5", ms=8, mew=1.4)
                ax.text(FLOOR * 1.6, y_phase[j], "none", va="center",
                        fontsize=8, color="0.5")
                continue
            ax.barh(y_phase[j], val, left=FLOOR, height=0.62, color=pcolor,
                    edgecolor="black", linewidth=0.4)
        ax.set_xscale("log")
        ax.set_xlim(FLOOR, 15000)
        ax.set_title(label, color=ccolor, fontweight="bold")
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        ax.set_axisbelow(True)

    axes[0].set_yticks(y_phase)
    axes[0].set_yticklabels([p[0] for p in PHASES])
    for ax in axes[1:]:
        ax.set_yticks(y_phase)
        ax.set_yticklabels([])
    fig.supxlabel("Median time per phase (ms, log scale)", fontsize=12, y=-0.02)
    fig.tight_layout()
    save(fig, "fig2c_small_multiples")


if __name__ == "__main__":
    print("Median phases (ms):")
    for tag, label, _, _ in CONDS:
        print(f"  {label:10s}", {p[0]: round(DATA[tag][p[1]], 1) for p in PHASES})
    variant_a()
    variant_b()
    variant_c()
    print("Done.")
