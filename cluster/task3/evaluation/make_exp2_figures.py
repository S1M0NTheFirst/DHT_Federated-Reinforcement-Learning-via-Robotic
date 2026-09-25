"""
Exp2 figures (same look as task2/evaluation/make_task2_figures.py), PDF only.
  exp2_latency : lookup + publish latency vs ring size, DHT vs Redis
  exp2_hops    : messages and lookup rounds per DHT lookup vs ring size
  exp2_churn   : lookup success vs failed coordination nodes (DHT k=3/k=8, Redis)
  exp2_churn_latency : median successful-lookup latency vs failed nodes
Usage: python3 make_exp2_figures.py [--results DIR] [--figdir DIR]
"""
import argparse
import csv
import glob
import os
import statistics as st

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator

FS = 24
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS - 2,
    "figure.titlesize": FS,
    "axes.spines.right": True, "axes.spines.top": True,
    "axes.linewidth": 1.1, "legend.frameon": True,
    "lines.linewidth": 2.6,
})

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RES = os.path.join(HERE, "..", "results", "exp2_dht_rendezvous")
DEFAULT_FIG = os.path.join(HERE, "figures")

DHT_BLUE = "#0072B2"      # DHT-FRL blue, as in the task2 figures
REDIS_GREY = "#555555"    # central-coordinator baseline
LEG = dict(fontsize=15, handlelength=1.8, handletextpad=0.5, labelspacing=0.28,
           borderpad=0.3, frameon=False)


def load(res_dir):
    path = os.path.join(res_dir, "exp2_ops.csv")
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["ms"] = float(r["ms"])
        r["ok"] = int(r["ok"])
        for k in ("rpcs", "rounds", "timeouts", "replicas", "fail_count", "ring_size"):
            r[k] = int(r[k]) if r[k] not in ("", None) else 0
    return rows


def pick(rows, **kw):
    return [r for r in rows if all(str(r[k]) == str(v) for k, v in kw.items())]


def med_iqr(vals):
    if not vals:
        return np.nan, 0.0, 0.0
    m = st.median(vals)
    return m, m - np.percentile(vals, 25), np.percentile(vals, 75) - m


def frame(ax):
    ax.tick_params(direction="in", which="both", top=True, right=True)


def save(fig, figdir, name):
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2196, top=0.914)
    fig.savefig(os.path.join(figdir, f"{name}.pdf"))
    plt.close(fig)
    print("wrote", name + ".pdf")


def log2_sizes_axis(ax, sizes):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(sizes[0] / 1.25, sizes[-1] * 1.25)


def fig_latency(rows, figdir):
    scale = pick(rows, phase="scale")
    sizes = sorted({r["ring_size"] for r in scale if r["backend"] == "dht"})
    if not sizes:
        return
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    series = [("dht", "get", DHT_BLUE, "-", "o", "DHT lookup"),
              ("dht", "put", DHT_BLUE, "--", "^", "DHT publish"),
              ("redis", "get", REDIS_GREY, "-", "s", "Redis GET"),
              ("redis", "put", REDIS_GREY, "--", "D", "Redis SET")]
    for backend, op, color, ls, mk, label in series:
        pts = [med_iqr([r["ms"] for r in pick(scale, backend=backend, op=op,
                                              ring_size=n) if r["ok"]])
               for n in sizes]
        m = np.array([p[0] for p in pts])
        if np.all(np.isnan(m)):
            continue
        err = [[p[1] for p in pts], [p[2] for p in pts]]
        ax.errorbar(sizes, m, yerr=err, color=color, ls=ls, marker=mk, ms=8,
                    capsize=3, lw=2.4, elinewidth=1.1, label=label)
    log2_sizes_axis(ax, sizes)
    ax.set_yscale("log")
    ax.set_xlabel("DHT ring size (nodes)")
    ax.set_ylabel("Latency (ms)")
    frame(ax)
    ax.legend(loc="upper left", ncol=2, **LEG)
    ax.margins(y=0.25)
    save(fig, figdir, "exp2_latency")


def fig_hops(rows, figdir):
    scale = pick(rows, phase="scale", backend="dht", op="get")
    sizes = sorted({r["ring_size"] for r in scale})
    if not sizes:
        return
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    for field, ls, mk, label in (("rpcs", "-", "o", "Messages per lookup"),
                                 ("rounds", "--", "^", "Lookup rounds")):
        pts = [med_iqr([r[field] for r in scale if r["ok"] and r["ring_size"] == n])
               for n in sizes]
        m = [p[0] for p in pts]
        err = [[p[1] for p in pts], [p[2] for p in pts]]
        ax.errorbar(sizes, m, yerr=err, color=DHT_BLUE, ls=ls, marker=mk, ms=8,
                    capsize=3, lw=2.4, elinewidth=1.1, label=label)
    ax.axhline(1, color=REDIS_GREY, ls=":", lw=2.0, label="Redis (1 request)")
    log2_sizes_axis(ax, sizes)
    ax.set_xlabel("DHT ring size (nodes)")
    ax.set_ylabel("Count per lookup")
    ax.set_ylim(0, None)
    frame(ax)
    ax.legend(loc="upper left", **LEG)
    ax.margins(y=0.3)
    save(fig, figdir, "exp2_hops")


def churn_points(rows, k):
    churn = pick(rows, phase="churn", backend="dht", op="get", ksize=k)
    counts = sorted({r["fail_count"] for r in churn})
    out = []
    for c in counts:
        per_trial = {}
        for r in churn:
            if r["fail_count"] == c:
                per_trial.setdefault(r["trial"], []).append(r)
        rates = [100.0 * sum(x["ok"] for x in v) / len(v) for v in per_trial.values()]
        okms = [x["ms"] for v in per_trial.values() for x in v if x["ok"]]
        out.append((c, st.mean(rates), min(rates), max(rates), med_iqr(okms)))
    return out


def fig_churn(rows, figdir):
    ks = sorted({r["ksize"] for r in pick(rows, phase="churn", backend="dht")}, key=int)
    if not ks:
        return
    n = pick(rows, phase="churn", backend="dht")[0]["ring_size"]
    counts = sorted({r["fail_count"] for r in pick(rows, phase="churn", backend="dht")})
    xpos = {c: i for i, c in enumerate(counts)}
    styles = [("-", "o"), ("-.", "s"), (":", "^")]

    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    for (ls, mk), k in zip(styles, ks):
        pts = churn_points(rows, k)
        x = [xpos[p[0]] for p in pts]
        y = [p[1] for p in pts]
        err = [[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]]
        ax.errorbar(x, y, yerr=err, color=DHT_BLUE, ls=ls, marker=mk, ms=8,
                    capsize=3, lw=2.4, elinewidth=1.1,
                    label=f"DHT, {k} replicas")
    red = pick(rows, phase="redis_down", backend="redis", op="get")
    # Redis has one server: it is only measured at 0 and 1 failed nodes.
    red_x = [c for c in (0, 1) if c in xpos]
    if red and red_x:
        ry = []
        for c in red_x:
            sel = [r for r in red if r["fail_count"] == c]
            ry.append(100.0 * sum(r["ok"] for r in sel) / len(sel) if sel else np.nan)
        ax.plot([xpos[c] for c in red_x], ry, color=REDIS_GREY, ls="--",
                marker="D", ms=8, lw=2.4, label="Redis (1 server)")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([str(c) for c in counts])
    ax.set_xlabel(f"Failed coordination nodes (of {n} DHT nodes)")
    ax.set_ylabel("Lookup success (%)")
    ax.set_ylim(-5, 125)
    ax.set_yticks([0, 25, 50, 75, 100])
    frame(ax)
    ax.legend(loc="upper right", ncol=3, **LEG)
    save(fig, figdir, "exp2_churn")

    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    for (ls, mk), k in zip(styles, ks):
        pts = churn_points(rows, k)
        x = [xpos[p[0]] for p in pts]
        m = [p[4][0] for p in pts]
        err = [[p[4][1] for p in pts], [p[4][2] for p in pts]]
        ax.errorbar(x, m, yerr=err, color=DHT_BLUE, ls=ls, marker=mk, ms=8,
                    capsize=3, lw=2.4, elinewidth=1.1, label=f"DHT, {k} replicas")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([str(c) for c in counts])
    ax.set_yscale("log")
    ax.set_xlabel(f"Failed DHT nodes (of {n})")
    ax.set_ylabel("Lookup latency (ms)")
    frame(ax)
    ax.legend(loc="upper left", **LEG)
    ax.margins(y=0.3)
    save(fig, figdir, "exp2_churn_latency")


def print_summary(rows):
    print("\n--- scale (successful ops, median ms) ---")
    scale = pick(rows, phase="scale")
    for n in sorted({r["ring_size"] for r in scale}):
        parts = []
        for b, op in (("dht", "get"), ("dht", "put"), ("redis", "get"), ("redis", "put")):
            sel = pick(scale, ring_size=n, backend=b, op=op)
            if sel:
                ok = [r["ms"] for r in sel if r["ok"]]
                parts.append(f"{b} {op} {st.median(ok):.2f}ms "
                             f"({100 * len(ok) / len(sel):.0f}% ok)" if ok else f"{b} {op} n/a")
        print(f"N={n:4d}: " + " | ".join(parts))
    print("--- churn (mean success over trials) ---")
    for k in sorted({r["ksize"] for r in pick(rows, phase="churn", backend="dht")}, key=int):
        pts = churn_points(rows, k)
        print(f"k={k}: " + ", ".join(f"{c} dead -> {m:.1f}%" for c, m, *_ in pts))
    red = pick(rows, phase="redis_down", backend="redis", op="get")
    for c, label in ((0, "up"), (1, "down")):
        sel = [r for r in red if r["fail_count"] == c]
        if sel:
            print(f"redis server {label}: {100 * sum(r['ok'] for r in sel) / len(sel):.1f}% ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="",
                    help="folder with exp2_ops.csv (default: newest job under "
                         "task3/results/exp2_dht_rendezvous)")
    ap.add_argument("--figdir", default=DEFAULT_FIG)
    a = ap.parse_args()
    res = a.results
    if not res:
        runs = sorted(glob.glob(os.path.join(DEFAULT_RES, "*", "exp2_ops.csv")),
                      key=os.path.getmtime)
        if not runs:
            raise SystemExit(f"no exp2_ops.csv under {DEFAULT_RES}")
        res = os.path.dirname(runs[-1])
    os.makedirs(a.figdir, exist_ok=True)
    print("results:", res)
    rows = load(res)
    fig_latency(rows, a.figdir)
    fig_hops(rows, a.figdir)
    fig_churn(rows, a.figdir)
    print_summary(rows)
    print("\n->", a.figdir)


if __name__ == "__main__":
    main()
