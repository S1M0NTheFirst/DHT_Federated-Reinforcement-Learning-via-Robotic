"""
Motivation figure: CRIU checkpoint overhead for two RL workloads.

X-axis = task:
    Task 1 = Robot task-bidding PPO agent   (criu_cold / criu_warm runners)
    Task 2 = D4RL Hopper-medium-v2          (run_motivation.py)

Per task we show, side by side, the COLD and WARM checkpoint:
  * left  axis : checkpoint size (MB) as bars
                 T1 547 MB / 2.16 GB,  T2 804 MB / 3.07 GB
  * right axis : migration latency (s) as a dot = dump + transfer
                 (transfer re-derived as size / 125 MB/s, the shared 1 Gbps
                  model used by evaluation/compare_tasks.py)

Reads:
    criu_cold/results/migration_events.csv      (Task 1 cold)
    criu_warm/results/migration_events.csv      (Task 1 warm)
    motivation/results/motivation.csv           (Task 2 cold + warm)
Writes:
    paper/figs/fig_motivation_criu.pdf  (and .png)

Run from swiftbot_rl/:
    python3 motivation/fig_motivation.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.abspath(os.path.join(HERE, ".."))          # swiftbot_rl/
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "paper", "figs"))
os.makedirs(OUT_DIR, exist_ok=True)

T1_COLD = os.path.join(ROOT, "criu_cold", "results", "migration_events.csv")
T1_WARM = os.path.join(ROOT, "criu_warm", "results", "migration_events.csv")
T2_CSV  = os.path.join(HERE, "results", "motivation.csv")

BW_MBPS = 125.0          # 1 Gbps normalised transfer link

# Paper-reported checkpoint sizes (MB).  T2 cold is the 804 MB single-run probe
# quoted in the paper; everything else matches the per-agent run medians.
PAPER_SIZE_MB = {
    (1, "cold"): 547.0,
    (1, "warm"): 2160.0,   # 2.16 GB
    (2, "cold"): 804.0,
    (2, "warm"): 3070.0,   # 3.07 GB
}

TASK_LABELS = {1: "Robot task-bidding\nPPO",
               2: "D4RL\nHopper-medium-v2"}

# ── clean sans-serif style (matches the motivation example) ─────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         18,
    "axes.labelsize":    19,
    "axes.titlesize":    19,
    "xtick.labelsize":   18,
    "ytick.labelsize":   18,
    "legend.fontsize":   16,
    "lines.linewidth":   1.3,
    "axes.linewidth":    1.0,
    "axes.grid":         False,
    "grid.linestyle":    ":",
    "grid.linewidth":    0.4,
    "grid.alpha":        0.6,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

# Size bars: cold = light salmon, warm = deep red (matches the example figure).
COLORS    = {"cold": "#e9a3a0", "warm": "#9e2b2b"}
BAR_EDGE  = "#4d1414"

# Latency dots: Task 1 vs Task 2 distinguished by colour + marker.
TASK_DOT = {
    1: dict(color="#08306b", marker="o", label="Robot task-bidding PPO"),   # navy circle
    2: dict(color="#e08214", marker="^", label="D4RL Hopper-medium-v2"),    # orange triangle
}


# ── data loading ──────────────────────────────────────────────────────────────
def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _dump_ms(path, mode):
    """Median real dump-phase latency (ms) for a task/mode."""
    df = pd.read_csv(path, dtype=str)
    if "criu_mode" in df.columns:
        df = df[df["criu_mode"].isin(
            {"cold": ["cold"], "warm": ["warm", "precopy"]}[mode])]

    # Task 2 cold rows in motivation.csv carry a 1-column schema shift (the real
    # returncode is in transfer_bandwidth_mbps, the real dump in criu_returncode).
    shifted = mode == "cold" and "transfer_bandwidth_mbps" in df.columns
    if shifted:
        rc, dump = _num(df["transfer_bandwidth_mbps"]), _num(df["criu_returncode"])
    else:
        rc = _num(df["criu_returncode"]) if "criu_returncode" in df.columns else None
        dump = _num(df.get("trigger_to_dump_ms", df.get("dump_ms")))

    if rc is not None:
        dump = dump[(rc == 0) | rc.isna()]
    return float(np.nanmedian(dump.values))


def load_all():
    """For each (task, mode): paper size (MB), measured dump (s), modeled
    transfer (s) = size / BW."""
    paths = {1: {"cold": T1_COLD, "warm": T1_WARM},
             2: {"cold": T2_CSV,  "warm": T2_CSV}}
    out = {}
    for t in (1, 2):
        for m in ("cold", "warm"):
            size = PAPER_SIZE_MB[(t, m)]
            dump_s = _dump_ms(paths[t][m], m) / 1000.0
            xfer_s = size / BW_MBPS               # MB / (MB/s) = s
            out[(t, m)] = dict(size=size, dump=dump_s, xfer=xfer_s)
    return out


# ── figure ────────────────────────────────────────────────────────────────────
def make_figure(data):
    fig, ax1 = plt.subplots(figsize=(7.0, 4.8))
    ax2 = ax1.twinx()

    centres = {1: 1.0, 2: 2.6}
    w = 0.42
    dx = {"cold": -0.23, "warm": 0.23}

    for t in (1, 2):
        c = centres[t]
        for m in ("cold", "warm"):
            d = data[(t, m)]
            x = c + dx[m]

            # ── SIZE bar (left linear axis) ───────────────────────────────────
            ax1.bar(x, d["size"], w, color=COLORS[m], edgecolor=BAR_EDGE,
                    linewidth=1.0, zorder=3)
            ax1.text(x, d["size"] + 120,
                     f"{d['size']/1000:.2f} GB" if d["size"] >= 1000
                     else f"{d['size']:.0f} MB",
                     ha="center", va="bottom", fontsize=17, color="0.15")

            # ── LATENCY dot (right axis): one dot = dump + transfer total ─────
            total = d["dump"] + d["xfer"]
            dot = TASK_DOT[t]
            ax2.scatter(x, total, s=160, marker=dot["marker"],
                        facecolor=dot["color"], edgecolor="white",
                        linewidth=1.0, zorder=6)
            ax2.annotate(f"{total:.0f} s", (x, total),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", va="bottom", fontsize=17,
                         fontweight="bold", color=dot["color"])

    # ── left axis: size (MB, linear) ──────────────────────────────────────────
    ax1.set_ylim(0, 7000)
    ax1.set_yticks([0, 1000, 2000, 3000])
    ax1.set_xticks(list(centres.values()))
    ax1.set_xticklabels([TASK_LABELS[1], TASK_LABELS[2]])
    ax1.set_xlim(0.3, 3.3)
    ax1.set_ylabel("CRIU checkpoint size (MB)")
    ax1.spines["top"].set_visible(False)

    # ── right axis: latency (s) — 0 sits above the bars via the negative limit;
    # extra headroom keeps the dots clear of the top-right legend.
    ax2.set_ylim(-38, 72)
    ax2.set_yticks([0, 10, 20, 30, 40])
    ax2.set_ylabel("Migration latency (s)")
    ax2.spines["top"].set_visible(False)

    # ── legends ───────────────────────────────────────────────────────────────
    size_leg = ax1.legend(
        handles=[mpatches.Patch(facecolor=COLORS["cold"], edgecolor=BAR_EDGE,
                                linewidth=1.0, label="Cold"),
                 mpatches.Patch(facecolor=COLORS["warm"], edgecolor=BAR_EDGE,
                                linewidth=1.0, label="Warm")],
        loc="upper left", frameon=False, handlelength=1.0, fontsize=16)
    ax1.add_artist(size_leg)

    ax2.legend(
        handles=[plt.Line2D([0], [0], color=TASK_DOT[t]["color"],
                            marker=TASK_DOT[t]["marker"], linestyle="none",
                            markeredgecolor="white", markersize=12,
                            label=TASK_DOT[t]["label"])
                 for t in (1, 2)],
        loc="upper right", frameon=False, handlelength=1.0, fontsize=16)

    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"fig_motivation_criu.{ext}")
        fig.savefig(path)
        print(f"  Saved {path}")
    plt.close(fig)


# ── summary to stdout ─────────────────────────────────────────────────────────
def _print_summary(data):
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  Transfer model: {BW_MBPS:.0f} MB/s (1 Gbps), shared across tasks")
    print(f"  {'group':10s}{'size MB':>10s}{'dump s':>9s}{'xfer s':>9s}"
          f"{'total s':>9s}")
    for (t, m), d in data.items():
        print(f"  {('T%d %s' % (t, m)):10s}{d['size']:>10.0f}"
              f"{d['dump']:>9.1f}{d['xfer']:>9.1f}{d['dump']+d['xfer']:>9.1f}")
    print(f"{sep}\n")


if __name__ == "__main__":
    for p in (T1_COLD, T1_WARM, T2_CSV):
        if not os.path.exists(p):
            print(f"ERROR: {p} not found.")
            raise SystemExit(1)

    print("Loading task data ...")
    data = load_all()
    _print_summary(data)
    print("Generating figure ...")
    make_figure(data)
    print("Done.")
