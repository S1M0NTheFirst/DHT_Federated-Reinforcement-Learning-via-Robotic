"""
Final comparison script — reads all 3 results directories and produces
4 paper-ready figures + 1 LaTeX summary table.
Run AFTER all 3 conditions have completed and metrics are collected.

Usage:
  python3 evaluation/compare_all.py
"""
import os, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = {
    "DHT+FRL (ours)":     "dht_frl/results/migration_events.csv",
    "CRIU cold":          "criu_cold/results/migration_events.csv",
    "CRIU warm (pre-copy)": "criu_warm/results/migration_events.csv",
}
COLORS = {
    "DHT+FRL (ours)":       "#1D9E75",
    "CRIU cold":            "#E24B4A",
    "CRIU warm (pre-copy)": "#7F77DD",
}
OUT_DIR = "evaluation/figures"
os.makedirs(OUT_DIR, exist_ok=True)
ROOT    = os.path.join(os.path.dirname(__file__), "..")


def load_all():
    data = {}
    for label, path in RESULTS.items():
        full = os.path.join(ROOT, path)
        if os.path.exists(full):
            data[label] = pd.read_csv(full)
            print(f"Loaded {label}: {len(data[label])} migration events")
        else:
            print(f"WARNING: {full} not found — skipping {label}")
    return data


def fig1_mtt_stacked_bar(data):
    """Figure 1: Migration Total Time breakdown — stacked bar."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x  = np.arange(len(data))
    w  = 0.5
    for i, (label, df) in enumerate(data.items()):
        dump     = df["trigger_to_dump_ms"].mean()
        transfer = df["dump_to_transfer_ms"].mean()
        restore  = df["transfer_to_restore_ms"].mean()
        policy   = df["policy_load_ms"].mean()
        color    = COLORS[label]
        ax.bar(i, dump,     w, label="Dump"          if i == 0 else "", color="#4A90D9")
        ax.bar(i, transfer, w, bottom=dump,           label="Transfer"    if i == 0 else "", color="#F5A623")
        ax.bar(i, restore,  w, bottom=dump+transfer,  label="Restore"     if i == 0 else "", color="#7ED321")
        ax.bar(i, policy,   w, bottom=dump+transfer+restore,
               label="Policy load (new)" if i == 0 else "", color=color, alpha=0.9)
        total = dump + transfer + restore + policy
        ax.text(i, total + 5, f"{total:.0f}ms", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(list(data.keys()), rotation=15, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Fig 1 — Migration Total Time breakdown")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig1_MTT_breakdown.png", dpi=150)
    plt.close(fig)
    print("Saved fig1_MTT_breakdown.png")


def fig2_regression_boxplot(data):
    """Figure 2: Policy regression distribution across 50 migration events."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_data  = [df["regression_pct"].dropna().values for df in data.values()]
    labels     = list(data.keys())
    colors     = [COLORS[l] for l in labels]
    bp = ax.boxplot(plot_data, patch_artist=True, notch=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Policy regression (%)")
    ax.set_title("Fig 2 — Policy regression per migration event")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig2_regression_boxplot.png", dpi=150)
    plt.close(fig)
    print("Saved fig2_regression_boxplot.png")


def fig3_downtime_cdf(data):
    """Figure 3: CDF of downtime across migration events."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, df in data.items():
        dt     = np.sort(df["downtime_ms"].dropna().values)
        cdf    = np.arange(1, len(dt) + 1) / len(dt)
        ax.plot(dt, cdf, label=label, color=COLORS[label], linewidth=2)
    ax.set_xlabel("Downtime (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Fig 3 — CDF of robot downtime during migration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig3_downtime_cdf.png", dpi=150)
    plt.close(fig)
    print("Saved fig3_downtime_cdf.png")


def fig4_gpu_cpu_during_migration(data):
    """Figure 4: GPU and CPU utilization during migration window."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    labels = list(data.keys())
    x = np.arange(len(labels))
    for ax, metric, title in [
        (ax1, "gpu_util_during_migration", "GPU utilization during migration"),
        (ax2, "cpu_util_during_migration", "CPU utilization during migration"),
    ]:
        means  = [data[l][metric].mean() for l in labels]
        stds   = [data[l][metric].std()  for l in labels]
        colors = [COLORS[l] for l in labels]
        ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Utilization (0-1)")
        ax.set_title(title)
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig4_resource_usage.png", dpi=150)
    plt.close(fig)
    print("Saved fig4_resource_usage.png")


def summary_table(data):
    """LaTeX and CSV summary table."""
    rows = []
    for label, df in data.items():
        rows.append({
            "System":                   label,
            "Mean MTT (ms)":            f"{df['total_MTT_ms'].mean():.1f} ± {df['total_MTT_ms'].std():.1f}",
            "Mean downtime (ms)":       f"{df['downtime_ms'].mean():.1f} ± {df['downtime_ms'].std():.1f}",
            "Mean regression (%)":      f"{df['regression_pct'].mean():.1f} ± {df['regression_pct'].std():.1f}",
            "Policy load (ms)":         f"{df['policy_load_ms'].mean():.1f}",
            "GPU during migration":     f"{df['gpu_util_during_migration'].mean():.2f}",
            "CPU during migration":     f"{df['cpu_util_during_migration'].mean():.2f}",
            "Net bytes xferred (MB)":   f"{df['network_bytes_transferred'].mean() / 1e6:.1f}",
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(f"{OUT_DIR}/summary_table.csv", index=False)
    with open(f"{OUT_DIR}/summary_table.tex", "w") as f:
        f.write(summary.to_latex(index=False, escape=False))
    print("\n=== SUMMARY TABLE ===")
    print(summary.to_string(index=False))
    print(f"\nSaved summary_table.csv and summary_table.tex to {OUT_DIR}/")


if __name__ == "__main__":
    data = load_all()
    if not data:
        print("No data found. Run experiments first.")
        exit(1)
    fig1_mtt_stacked_bar(data)
    fig2_regression_boxplot(data)
    fig3_downtime_cdf(data)
    fig4_gpu_cpu_during_migration(data)
    summary_table(data)
    print("\nAll comparison figures saved to evaluation/figures/")
