"""
Task2 final figures — consistent Arial font everywhere, PDF only.
  fig1: CDF of total migration time
  fig2: reward vs FL round — DHT-FRL, cold restart, no migration
  fig3: latency per condition — downtime & MTT (median bar + IQR error bar)
"""
import csv, os, statistics as st
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator

# ---- ONE font size / family everywhere, larger fonts + lines, full box ----
FS = 24
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS - 2,
    "figure.titlesize": FS,
    "axes.spines.right": True, "axes.spines.top": True,   # full box
    "axes.linewidth": 1.1, "legend.frameon": True,
    "lines.linewidth": 2.6,
})

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results"); FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# consistent color per condition (DHT-FRL blue is shared across fig1/fig2)
COLOR = {"dht_frl": "#0072B2", "app_cold": "#009E73", "app_warm": "#9467BD",
         "tcp_scp": "#E69F00", "dmtcp": "#CC79A7", "cold_restart": "#D55E00",
         "no_migration": "#000000"}
# distinct line style per condition (fig1 identifies by color AND dash)
LS = {"dht_frl": "-", "tcp_scp": "--", "app_cold": "-.", "app_warm": (0, (1, 1)),
      "dmtcp": (0, (5, 1))}
LAB = {"dht_frl": "DHT-FRL", "app_cold": "App-CR cold", "app_warm": "App-CR warm",
       "tcp_scp": "App-CR direct", "dmtcp": "DMTCP", "cold_restart": "Cold restart",
       "no_migration": "No migration"}


def events(c):
    p = os.path.join(RES, f"task2_{c}", "migration_events.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def curve(c):
    p = os.path.join(RES, f"task2_{c}", "task_logs.csv"); d = {}
    for r in csv.DictReader(open(p)):
        try:
            rd = int(float(r["fl_round"])); ev = float(r["eval_return"])
        except Exception:
            continue
        if ev >= 0:
            d.setdefault(rd, []).append(ev)
    xs = sorted(d)
    return xs, [st.mean(d[x]) for x in xs]


def mtime_s(c):
    rw = events(c); k = "downtime_ms" if c == "dmtcp" else "total_MTT_ms"
    if c == "dmtcp":
        rw = [r for r in rw if float(r["checkpoint_size_mb"]) > 500]
    return sorted(float(r[k]) / 1000 for r in rw
                  if r.get(k) not in (None, "", "-1") and float(r[k]) >= 0)


def _frame(leg):
    leg.get_frame().set_edgecolor("0.4")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_boxstyle("square")


def save(fig, name, tight=True):
    # tight=True trims whitespace (per-figure, so widths can differ). tight=False
    # keeps the exact figsize canvas -> figures with equal figsize get equal
    # rendered width (used for fig2/fig3, which must match in the paper).
    kw = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(os.path.join(FIG, f"{name}.pdf"), **kw)
    plt.close(fig); print("wrote", name + ".pdf")


# ---------------- fig1: CDF of total migration time ----------------
def fig1_cdf():
    # cold restart is a separate baseline and is intentionally not shown here.
    order = ["dht_frl", "tcp_scp", "app_cold", "app_warm", "dmtcp"]
    # Wide-but-not-tall frame reads best for a CDF and stops the curves from
    # looking squished.
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for c in order:
        v = mtime_s(c)
        if not v:
            continue
        # Clean CDF *line* (not a staircase): each sample steps the CDF by 1/n,
        # anchored at y=0 on the left so every curve starts from the baseline.
        y = np.arange(1, len(v) + 1) / len(v)
        xs = np.concatenate(([v[0]], v))
        ys = np.concatenate(([0.0], y))
        ax.plot(xs, ys, color=COLOR[c], ls=LS[c],
                lw=3.0 if c == "dht_frl" else 2.4, label=LAB[c],
                solid_capstyle="round", solid_joinstyle="round",
                zorder=5 if c == "dht_frl" else 3)
    ax.set_xscale("log")
    ax.set_xlabel("Total migration time (s)")
    ax.set_ylabel("CDF")
    # Small headroom above CDF=1 only; the compact legend drops down into the
    # empty upper-left region (curves are all low at small x), so nothing is
    # squished and the legend never touches a curve.
    ax.set_ylim(0, 1.24)
    ax.set_yticks(np.arange(0, 1.01, 0.25))

    # full box with clean, evenly-spaced decade ticks on x;
    # no log minor ticks -- their uneven spacing read as "inconsistent"
    for s in ax.spines.values():
        s.set_visible(True)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="major", direction="in", length=6, width=1.1,
                   top=True, right=True)
    # empty background: no grid lines behind the curves

    # compact, smaller legend tucked into the empty upper-left corner
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.015),
                    fontsize=15, handlelength=1.6, handletextpad=0.5,
                    labelspacing=0.28, borderpad=0.3, frameon=False)
    # fixed canvas (tight=False) so fig1 width matches fig2/fig3 exactly;
    # pin margins so the plot box is exactly 210 pt tall (== fig2/fig3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2196, top=0.914)
    save(fig, "fig1", tight=False)


# ---------------- fig2: reward — DHT vs cold restart vs no migration ----------------
def fig2_reward():
    sel = ["no_migration", "dht_frl", "cold_restart"]
    MIGS = [30, 60, 90, 120, 140]
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    for m in MIGS:
        ax.axvline(m, color="red", lw=1.2, ls=":", zorder=0)
    for c in sel:
        xs, ys = curve(c)
        ls = "--" if c == "no_migration" else "-"
        ax.plot(xs, ys, color=COLOR[c], lw=2.4, ls=ls, label=LAB[c], zorder=3)
    ax.set_xlabel("Federated learning round")
    ax.set_ylabel(r"Eval Return ($\times10^{3}$)")
    ax.set_xlim(0, 150)
    # show returns in thousands: 1000/2000/3000 -> 1/2/3
    ax.set_yticks([0, 1000, 2000, 3000])
    ax.set_yticklabels(["0", "1", "2", "3"])
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.plot([], [], color="red", lw=1.2, ls=":", label="Migration round")
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 1.0),
              handlelength=2.0, labelspacing=0.25, borderpad=0.4,
              borderaxespad=0.0, fontsize=FS - 5, frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2196, top=0.914)
    save(fig, "fig2", tight=False)


# ---------------- fig3: latency — median bar + IQR error bar ----------------
def fig3_latency():
    order = ["dht_frl", "tcp_scp", "app_cold", "app_warm", "dmtcp"]

    def triplet(c, key):
        rw = events(c)
        if c == "dmtcp":
            rw = [r for r in rw if float(r["checkpoint_size_mb"]) > 500]
        v = sorted(float(r[key]) / 1000 for r in rw
                   if r.get(key) not in (None, "", "-1") and float(r[key]) >= 0)
        if not v:
            return 0.0, 0.0, 0.0
        med = st.median(v)
        p25 = np.percentile(v, 25); p75 = np.percentile(v, 75)
        return med, med - p25, p75 - med

    # dmtcp MTT reflects the heavy path (use its downtime)
    dt = [triplet(c, "downtime_ms") for c in order]
    mt = [triplet(c, "downtime_ms" if c == "dmtcp" else "total_MTT_ms") for c in order]
    dt_med = [t[0] for t in dt]; dt_err = [[t[1] for t in dt], [t[2] for t in dt]]
    mt_med = [t[0] for t in mt]; mt_err = [[t[1] for t in mt], [t[2] for t in mt]]

    x = np.arange(len(order)); w = 0.26
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    ax.bar(x - w / 2, dt_med, w, color="#0072B2", edgecolor="black", linewidth=0.6,
           yerr=dt_err, capsize=3, error_kw=dict(lw=1.1), label="Downtime")
    ax.bar(x + w / 2, mt_med, w, color="#E69F00", edgecolor="black", linewidth=0.6,
           yerr=mt_err, capsize=3, error_kw=dict(lw=1.1), label="MTT")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([LAB[c].replace(" ", "\n", 1) for c in order],
                       fontsize=FS - 2)
    ax.set_ylabel("Migration latency (s)")
    ax.tick_params(direction="in", which="both", right=True)
    ax.legend(loc="upper left", handlelength=1.6, borderpad=0.5, frameon=False)
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2196, top=0.914)
    save(fig, "fig3", tight=False)


if __name__ == "__main__":
    fig1_cdf(); fig2_reward(); fig3_latency()
    print("\n->", FIG)
