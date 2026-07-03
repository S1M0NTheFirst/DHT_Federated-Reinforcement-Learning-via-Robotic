"""
Timeline comparison figure — SwiftBot-RL migration procedure.
Style inspired by Poby ATC'25 Fig 3: horizontal Gantt bars showing
concurrent phases per condition. Log-scale x-axis handles CRIU Warm outlier.

Key insight visualized:
  DHT+FRL: robot stays ALIVE during checkpoint (7s); paused only 331ms for restore.
  All baselines: robot fully STOPPED for their entire downtime.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import pandas as pd

ROOT    = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────────
C = {
    "dump":     "#4A90D9",
    "precopy":  "#93C6E8",
    "transfer": "#F5A623",
    "restore":  "#7ED321",
    "policy":   "#E24B4A",
    "alive":    "#AEDBA9",
    "dead":     "#F0A0A0",
    "bid":      "#C89FD4",
}

# ── Load results ──────────────────────────────────────────────────────────────
FILES = {
    "DHT+FRL\n(Ours)":       "dht_frl/results/migration_events.csv",
    "CRIU Cold":              "criu_cold/results/migration_events.csv",
    "CRIU Warm\n(Pre-copy)": "criu_warm/results/migration_events.csv",
    "Docker\nCheckpoint":    "docker_checkpoint/results/migration_events.csv",
    "Cold\nRestart":         "cold_restart/results/migration_events.csv",
}
DATA = {}
for label, rel in FILES.items():
    full = os.path.join(ROOT, rel)
    if os.path.exists(full):
        df = pd.read_csv(full)
        DATA[label] = {col: df[col].mean() for col in df.select_dtypes("number").columns}

LABELS = list(DATA.keys())
N = len(LABELS)

# ── Segment builder ───────────────────────────────────────────────────────────
# Returns list of dicts: {start, end, color, label, row}
# row 0 = Host,  row 1 = Robot

EPSILON = 0.5  # minimum visible width on log axis

def seg(start, width, color, label, row):
    w = max(width, EPSILON)
    return dict(start=start, end=start+w, color=color, label=label, row=row)

def build(cond_label, m):
    dump  = m["trigger_to_dump_ms"]
    xfer  = m["dump_to_transfer_ms"]
    rest  = m["transfer_to_restore_ms"]
    pol   = m["policy_load_ms"]
    down  = m["downtime_ms"]
    mtt   = m["total_MTT_ms"]

    segs = []
    if "DHT+FRL" in cond_label:
        # Host performs checkpoint while robot is still alive
        segs += [
            seg(1, dump,      C["dump"],     "Checkpoint",        0),
            seg(1+dump, xfer, C["transfer"], "Transfer",          0),
            seg(1+dump+xfer, rest, C["restore"], "Restore",       0),
            seg(1+dump+xfer+rest, pol, C["policy"], "Policy\nLoad", 0),
        ]
        # Robot alive during checkpoint, paused only for restore
        segs += [
            seg(1,     dump,  C["alive"], "Robot Running\n(alive)", 1),
            seg(1+dump, down, C["dead"],  "Paused",                  1),
            seg(1+dump+down, rest*0.6, C["bid"], "First bid\n(policy intact)", 1),
        ]
        return segs, 1+dump, down

    elif "Warm" in cond_label or "Pre-copy" in cond_label:
        precopy    = xfer           # the long pre-copy transfer
        final_dump = dump           # brief final freeze+dump
        segs += [
            seg(1, precopy,              C["precopy"],  "Pre-copy Transfer",  0),
            seg(1+precopy, final_dump,   C["dump"],     "Final\nDump",        0),
            seg(1+precopy+final_dump, rest, C["restore"], "Restore",          0),
        ]
        segs += [
            seg(1, precopy,              C["alive"], "Robot Running\n(live during pre-copy)", 1),
            seg(1+precopy, down,         C["dead"],  "Paused",               1),
            seg(1+precopy+down, rest*0.6, C["bid"],  "First bid\n(no policy)", 1),
        ]
        return segs, 1+precopy, down

    else:
        # All cold-style: robot fully paused the whole time
        segs += [
            seg(1, dump,  C["dump"],     "Dump",     0),
        ]
        if xfer > 0:
            segs += [seg(1+dump, xfer, C["transfer"], "Transfer", 0)]
        segs += [
            seg(1+dump+xfer, rest, C["restore"], "Restore", 0),
        ]
        segs += [
            seg(1, mtt,           C["dead"], "Paused",              1),
            seg(1+mtt, rest*0.6,  C["bid"],  "First bid\n(no policy)", 1),
        ]
        return segs, 1, down


# ── Layout ────────────────────────────────────────────────────────────────────
ROW_H  = 0.30
VSEP   = 0.12   # gap between host & robot bars
COND_H = 1.05   # vertical slot per condition

fig, ax = plt.subplots(figsize=(13, N * COND_H + 1.8))
ax.set_xscale("log")

for idx, label in enumerate(LABELS):
    m = DATA[label]
    segs, pause_start, down = build(label, m)
    mtt = m["total_MTT_ms"]

    y_base  = (N - 1 - idx) * COND_H
    y_host  = y_base + VSEP + ROW_H + 0.04
    y_robot = y_base + VSEP

    # alternating bg
    ax.axhspan(y_base, y_base + COND_H - 0.02,
               color="#F7F7F7" if idx % 2 == 0 else "#EEEEEE", zorder=0)

    # condition label
    ax.text(0.55, y_base + COND_H / 2,
            label, ha="right", va="center",
            fontsize=9.5, fontweight="bold", multialignment="center")

    for s in segs:
        y = y_host if s["row"] == 0 else y_robot
        rect = plt.Rectangle(
            (s["start"], y), s["end"] - s["start"], ROW_H,
            facecolor=s["color"], edgecolor="white", linewidth=0.7, zorder=2,
        )
        ax.add_patch(rect)
        width_log = np.log10(s["end"]) - np.log10(s["start"])
        if width_log > 0.12:          # only label if wide enough on log scale
            mid = np.sqrt(s["start"] * s["end"])   # geometric midpoint
            ax.text(mid, y + ROW_H / 2, s["label"],
                    ha="center", va="center", fontsize=6.5,
                    multialignment="center", zorder=3)

    # row type labels on right
    xmax_plot = 1.1e5
    ax.text(xmax_plot * 1.02, y_host  + ROW_H/2, "Host",  va="center", fontsize=8, color="#555")
    ax.text(xmax_plot * 1.02, y_robot + ROW_H/2, "Robot", va="center", fontsize=8, color="#555")

    # downtime annotation below robot bar
    ax.annotate(
        f"Downtime: {down:.0f} ms",
        xy=(pause_start, y_robot),
        xytext=(pause_start, y_robot - 0.32),
        fontsize=7.5, ha="left", va="top",
        color="#BB2222", fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="#BB2222", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#BB2222", lw=0.8),
        zorder=5,
    )


# ── Legend ────────────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(color=C["dump"],     label="Checkpoint / Dump"),
    mpatches.Patch(color=C["precopy"],  label="Pre-copy Transfer (CRIU Warm)"),
    mpatches.Patch(color=C["transfer"], label="Transfer"),
    mpatches.Patch(color=C["restore"],  label="Restore"),
    mpatches.Patch(color=C["policy"],   label="Policy Load (DHT+FRL only)"),
    mpatches.Patch(color=C["alive"],    label="Robot: Alive / Running"),
    mpatches.Patch(color=C["dead"],     label="Robot: Paused / Offline"),
    mpatches.Patch(color=C["bid"],      label="Robot: First Bid"),
]
ax.legend(handles=patches, fontsize=8, framealpha=0.95,
          ncol=4, loc="lower center",
          bbox_to_anchor=(0.5, -0.15))

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(0.8, 1.1e5)
ax.set_ylim(-0.55, N * COND_H + 0.1)
ax.set_xlabel("Time since migration trigger (ms, log scale)", fontsize=11)
ax.set_yticks([])
ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda v, _: f"{int(v):,}" if v >= 1 else f"{v:.1f}"
))
ax.set_title(
    "Figure X  —  SwiftBot-RL Migration Procedure Timeline\n"
    "DHT+FRL robot stays alive during checkpoint (downtime = restore only); "
    "all baselines are fully offline",
    fontsize=11, fontweight="bold", pad=12,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.grid(axis="x", alpha=0.25, linestyle="--", which="both")

fig.tight_layout(rect=[0.08, 0.1, 0.97, 1.0])
out = f"{OUT_DIR}/fig_timeline_migration.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}")
