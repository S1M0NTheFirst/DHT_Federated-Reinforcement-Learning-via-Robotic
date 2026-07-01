"""
Paper-ready figures for SwiftBot-RL migration comparison.
Run from swiftbot_rl/:  python3 evaluation/paper_figures.py
Saves all figures to evaluation/figures/ as PDF + PNG.
"""

import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

RESULTS = {
    "DHT+FRL\n(ours)":        "dht_frl/results/migration_events.csv",
    "Docker\nCheckpoint":     "docker_checkpoint/results/migration_events.csv",
    "CRIU cold":              "criu_cold/results/migration_events.csv",
    "CRIU warm\n(pre-copy)":  "criu_warm/results/migration_events.csv",
    "Cold\nRestart":          "cold_restart/results/migration_events.csv",
}
COLORS = {
    "DHT+FRL\n(ours)":        "#1D9E75",
    "Docker\nCheckpoint":     "#F5A623",
    "CRIU cold":              "#4A90D9",
    "CRIU warm\n(pre-copy)":  "#7F77DD",
    "Cold\nRestart":          "#E24B4A",
}
LINE_STYLES = {
    "DHT+FRL\n(ours)":        ("solid",   2.2),
    "Docker\nCheckpoint":     ("dashed",  1.8),
    "CRIU cold":              ("dashdot", 1.8),
    "CRIU warm\n(pre-copy)":  ("dotted",  2.0),
    "Cold\nRestart":          ((0,(3,1,1,1)), 1.8),
}

# ── style helpers ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Serif",
    "font.size":        10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
    "axes.axisbelow":   True,
})

def save(fig, name):
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}.pdf / .png")


def load_all():
    data = {}
    for label, rel in RESULTS.items():
        full = os.path.join(ROOT, rel)
        if os.path.exists(full):
            data[label] = pd.read_csv(full)
            print(f"  Loaded {label.replace(chr(10),' '):30s} — {len(data[label])} events")
        else:
            print(f"  WARNING: {full} not found — skipping")
    return data


# ── Figure 1: CDF of downtime ──────────────────────────────────────────────────
def fig1_downtime_cdf(data):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for label, df in data.items():
        vals = np.sort(df["downtime_ms"].dropna().values)
        cdf  = np.arange(1, len(vals) + 1) / len(vals)
        ls, lw = LINE_STYLES[label]
        ax.plot(vals, cdf, label=label.replace("\n", " "),
                color=COLORS[label], linestyle=ls, linewidth=lw)

    ax.axvline(500,  color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(2000, color="gray", linestyle=":",  linewidth=1, alpha=0.7)
    ax.text(500,  0.05, " 500 ms",  fontsize=8, color="gray", va="bottom")
    ax.text(2000, 0.05, " 2 s",     fontsize=8, color="gray", va="bottom")

    ax.set_xlabel("Downtime per migration event (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of Robot Downtime per Migration Event")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "fig1_downtime_cdf")


# ── Figure 2: Stacked bar — MTT phase breakdown ────────────────────────────────
def fig2_mtt_breakdown(data):
    labels  = list(data.keys())
    x       = np.arange(len(labels))
    w       = 0.5
    phase_colors = ["#4A90D9", "#F5A623", "#7ED321"]
    phase_keys   = ["trigger_to_dump_ms", "dump_to_transfer_ms", "transfer_to_restore_ms"]
    phase_names  = ["Dump", "Transfer", "Restore"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bottoms = np.zeros(len(labels))

    for i, (key, name, color) in enumerate(zip(phase_keys, phase_names, phase_colors)):
        means = np.array([data[l][key].mean() for l in labels])
        ax.bar(x, means, w, bottom=bottoms,
               color=color, label=name, edgecolor="white", linewidth=0.5)
        bottoms += means

    # policy_load on top, per-condition color
    policy = np.array([data[l]["policy_load_ms"].mean() for l in labels])
    for i, (xi, val, label) in enumerate(zip(x, policy, labels)):
        if val > 0:
            ax.bar(xi, val, w, bottom=bottoms[i],
                   color=COLORS[label], alpha=0.9,
                   label="Policy load" if i == 0 else "",
                   edgecolor="white", linewidth=0.5)

    totals = bottoms + policy
    for xi, total in zip(x, totals):
        ax.text(xi, total + 60, f"{total:.0f} ms",
                ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("\n", "\n") for l in labels], fontsize=9)
    ax.set_ylabel("Migration Total Time (ms)")
    ax.set_title("Migration Time Breakdown by Phase")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, "fig2_mtt_breakdown")


# ── Figure 3: Log-scale grouped bar — checkpoint footprint ────────────────────
def fig3_checkpoint_footprint(data):
    labels   = list(data.keys())
    x        = np.arange(len(labels))
    w        = 0.35

    ck_mean  = []
    ck_std   = []
    net_mean = []
    net_std  = []
    for l in labels:
        df  = data[l]
        ck  = df["checkpoint_size_mb"].replace(0, np.nan)
        net = df["network_bytes_transferred"] / 1e6
        ck_mean.append(ck.mean() if not ck.isna().all() else 0.01)
        ck_std.append(ck.std()   if not ck.isna().all() else 0)
        net_mean.append(net.mean())
        net_std.append(net.std())

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [COLORS[l] for l in labels]

    bars1 = ax.bar(x - w/2, ck_mean,  w, yerr=ck_std,  capsize=4,
                   color=colors, alpha=0.85, label="Checkpoint image (MB)",
                   edgecolor="white")
    bars2 = ax.bar(x + w/2, net_mean, w, yerr=net_std, capsize=4,
                   color=colors, alpha=0.45, hatch="///",
                   label="Network transfer (MB)", edgecolor="white")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Size (MB, log scale)")
    ax.set_title("Migration Footprint: Checkpoint Size vs. Network Transfer")
    ax.legend(fontsize=8)
    ax.grid(axis="y", which="both")

    for bar, val in zip(bars1, ck_mean):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h * 1.5,
                    f"{h:.1f}", ha="center", fontsize=7.5, fontweight="bold")
    for bar, val in zip(bars2, net_mean):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, max(h * 1.5, 0.015),
                f"{h:.2f}", ha="center", fontsize=7.5)

    # annotate DHT+FRL
    dht_idx = list(labels).index("DHT+FRL\n(ours)")
    ax.annotate("weights only\n~0.5 MB",
                xy=(x[dht_idx] - w/2, ck_mean[dht_idx]),
                xytext=(x[dht_idx] - w/2 - 0.6, ck_mean[dht_idx] * 8),
                fontsize=7.5, color="#1D9E75",
                arrowprops=dict(arrowstyle="->", color="#1D9E75", lw=0.8))

    fig.tight_layout()
    save(fig, "fig3_checkpoint_footprint")


# ── Figure 4: Scatter — downtime vs. regression trade-off ─────────────────────
def fig4_tradeoff_scatter(data):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # ideal zone
    ideal = mpatches.FancyBboxPatch((0, -35), 500, 40,
                                     boxstyle="round,pad=5",
                                     facecolor="#1D9E75", alpha=0.08,
                                     edgecolor="#1D9E75", linewidth=0.8,
                                     linestyle="--", zorder=0)
    ax.add_patch(ideal)
    ax.text(250, -30, "Ideal zone", fontsize=8, color="#1D9E75",
            ha="center", style="italic")

    for label, df in data.items():
        ax.scatter(df["downtime_ms"], df["regression_pct"],
                   color=COLORS[label], alpha=0.65, s=50,
                   label=label.replace("\n", " "),
                   edgecolors="white", linewidths=0.4, zorder=2)
        mx = df["downtime_ms"].median()
        my = df["regression_pct"].median()
        ax.plot(mx, my, marker="D", ms=9, color=COLORS[label],
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.annotate(label.replace("\n", " "), (mx, my),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=7.5, color=COLORS[label], fontweight="bold")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Downtime per migration event (ms)")
    ax.set_ylabel("Policy regression (%)")
    ax.set_title("Per-Event Trade-off: Downtime vs. Policy Regression")
    ax.set_xlim(left=0)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, "fig4_tradeoff_scatter")


# ── Figure 5: 3-panel violin — per-phase latency ──────────────────────────────
def fig5_phase_violin(data):
    phases = [
        ("trigger_to_dump_ms",      "Dump phase"),
        ("dump_to_transfer_ms",     "Transfer phase"),
        ("transfer_to_restore_ms",  "Restore phase"),
    ]
    labels = list(data.keys())
    colors = [COLORS[l] for l in labels]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax, (col, title) in zip(axes, phases):
        plot_data = []
        for l in labels:
            vals = data[l][col].dropna().values
            vals = vals[vals > 0]
            plot_data.append(vals if len(vals) > 1 else np.array([0.01]))

        parts = ax.violinplot(plot_data, positions=range(len(labels)),
                              showmedians=True, showextrema=True, widths=0.7)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.65)
            pc.set_edgecolor("white")
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.8)
        for part in ("cbars", "cmins", "cmaxes"):
            parts[part].set_color("gray")
            parts[part].set_linewidth(0.8)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace("\n", "\n") for l in labels], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y")
        if ax is axes[0]:
            ax.set_ylabel("Latency (ms)")

    fig.suptitle("Per-Phase Migration Latency Distribution", fontsize=11, y=1.01)
    fig.tight_layout()
    save(fig, "fig5_phase_violin")


# ── Figure 6: FL convergence timeline with migration markers ──────────────────
def fig6_convergence_timeline(data):
    conv_path = os.path.join(ROOT, "dht_frl/results/fl_convergence.csv")
    mig_path  = os.path.join(ROOT, "dht_frl/results/migration_events.csv")

    if not os.path.exists(conv_path):
        print("  WARNING: fl_convergence.csv not found — skipping fig6")
        return

    conv = pd.read_csv(conv_path)
    mig  = pd.read_csv(mig_path) if os.path.exists(mig_path) else None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                    gridspec_kw={"hspace": 0.08})

    rounds = conv["round"].values
    ax1.plot(rounds, conv["success_rate"], color="#1D9E75",
             linewidth=2, label="Task success rate")
    ax1.set_ylabel("Task success rate")
    ax1.set_ylim(-0.05, 1.15)
    ax1.fill_between(rounds, conv["success_rate"], alpha=0.08, color="#1D9E75")

    loss = conv["train_loss"].replace(0, np.nan) if "train_loss" in conv.columns else None
    if loss is not None:
        ax2.plot(rounds, loss, color="#4A90D9", linewidth=2, label="Train loss")
        ax2.set_ylabel("Training loss")
        ax2.fill_between(rounds, loss, alpha=0.08, color="#4A90D9")

    # migration event markers
    if mig is not None and len(mig) > 0:
        t0    = mig["timestamp"].min()
        t_max = mig["timestamp"].max()
        r_max = rounds[-1]
        annotated = False
        for _, row in mig.iterrows():
            frac  = (row["timestamp"] - t0) / max(t_max - t0, 1)
            r_est = 1 + frac * (r_max - 1)
            for ax in (ax1, ax2):
                ax.axvline(r_est, color="#E24B4A", linestyle="--",
                           alpha=0.4, linewidth=1.0)
            if not annotated:
                ax1.annotate("↑ migration events",
                             xy=(r_est, 0.6),
                             xytext=(r_est + 1, 0.45),
                             fontsize=8, color="#E24B4A",
                             arrowprops=dict(arrowstyle="->",
                                             color="#E24B4A", lw=0.8))
                annotated = True

    mig_patch = mpatches.Patch(color="#E24B4A", alpha=0.5, label="Migration event")
    ax1.legend(handles=[ax1.get_lines()[0], mig_patch], fontsize=8, loc="lower right")
    if loss is not None:
        ax2.legend(handles=[ax2.get_lines()[0], mig_patch], fontsize=8, loc="upper right")

    ax2.set_xlabel("FL Round")
    fig.suptitle("DHT+FRL: Learning Continuity Under Live Migrations", fontsize=11)
    save(fig, "fig6_convergence_timeline")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    data = load_all()
    if not data:
        print("No data found. Run experiments first.")
        exit(1)

    print("\nGenerating figures...")
    fig1_downtime_cdf(data)
    fig2_mtt_breakdown(data)
    fig3_checkpoint_footprint(data)
    fig4_tradeoff_scatter(data)
    fig5_phase_violin(data)
    fig6_convergence_timeline(data)

    print(f"\nAll figures saved to {OUT_DIR}/")
