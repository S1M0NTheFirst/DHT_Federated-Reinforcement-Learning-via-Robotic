"""
Exp1 figures (same look as make_exp2_figures.py / task2 figures), PDF only.
  exp1_state_preserved : % of migrations whose robot resumed with its full state
                         (replay buffer + optimizer), per kill point and condition
  exp1_reward          : fleet mean eval return vs FL round per condition, with
                         the migration waves marked (source killed every time)
  exp1_downtime        : median robot downtime per kill point and condition
Reads the newest job of each condition under
task3/results/exp1_source_failure/<condition>/<jobid>/.
Usage: python3 make_exp1_figures.py [--results DIR] [--figdir DIR]
"""
import argparse
import csv
import glob
import json
import os
import statistics as st

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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
DEFAULT_RES = os.path.join(HERE, "..", "results", "exp1_source_failure")
DEFAULT_FIG = os.path.join(HERE, "figures")

# task2 colours for the shared conditions; the DHT ablation gets a light blue.
ORDER = ["dht_frl", "dht_r0", "tcp_scp", "cold_restart"]
COLOR = {"dht_frl": "#0072B2", "dht_r0": "#56B4E9", "tcp_scp": "#E69F00",
         "cold_restart": "#D55E00"}
HATCH = {"dht_frl": "", "dht_r0": "//", "tcp_scp": "", "cold_restart": "\\\\"}
LS = {"dht_frl": "-", "dht_r0": (0, (5, 1)), "tcp_scp": "--", "cold_restart": "-."}
LAB = {"dht_frl": "DHT-FRL", "dht_r0": "DHT, no replica", "tcp_scp": "App-CR direct",
       "cold_restart": "Cold restart"}
KP = ["after_save", "mid_transfer", "after_transfer"]
KP_LAB = {"after_save": "After save", "mid_transfer": "Mid-transfer",
          "after_transfer": "After transfer"}
LEG = dict(fontsize=15, handlelength=1.8, handletextpad=0.5, labelspacing=0.28,
           borderpad=0.3, frameon=False)


def newest_runs(res_root):
    runs = {}
    for cond in ORDER:
        cands = glob.glob(os.path.join(res_root, cond, "*", "migration_events.csv"))
        if cands:
            runs[cond] = os.path.dirname(max(cands, key=os.path.getmtime))
    return runs


def events(run):
    return list(csv.DictReader(open(os.path.join(run, "migration_events.csv"))))


def num(r, k, default=np.nan):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return default


def curve(run):
    # task_logs.csv is written by Flower when FL finishes; a run cut short
    # (stall / walltime) only has the runner's periodic snapshot.
    p = os.path.join(run, "task_logs.csv")
    if not os.path.exists(p):
        p = os.path.join(run, "task_logs.partial.csv")
    if not os.path.exists(p):
        return [], []
    d = {}
    for r in csv.DictReader(open(p)):
        try:
            rd, ev = int(float(r["fl_round"])), float(r["eval_return"])
        except (KeyError, ValueError):
            continue
        d.setdefault(rd, []).append(ev)
    xs = sorted(d)
    return xs, [st.mean(d[x]) for x in xs]


def meta(run):
    p = os.path.join(run, "exp1_meta.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def frame(ax):
    ax.tick_params(direction="in", which="both", top=True, right=True)


def save(fig, figdir, name):
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2196, top=0.914)
    fig.savefig(os.path.join(figdir, f"{name}.pdf"))
    plt.close(fig)
    print("wrote", name + ".pdf")


def preserved_pct(rows):
    if not rows:
        return np.nan
    return 100.0 * sum(num(r, "restored", 0) == 1 for r in rows) / len(rows)


def grouped_bars(ax, runs, value_fn, fmt, top):
    conds = [c for c in ORDER if c in runs]
    w = 0.8 / max(1, len(conds))
    for i, c in enumerate(conds):
        rows = events(runs[c])
        vals = [value_fn([r for r in rows if r.get("kill_point") == kp]) for kp in KP]
        xs = np.arange(len(KP)) + (i - (len(conds) - 1) / 2) * w
        bars = ax.bar(xs, [0 if np.isnan(v) else v for v in vals], w,
                      color=COLOR[c], hatch=HATCH[c], edgecolor="black",
                      linewidth=0.8, label=LAB[c], zorder=2)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * 0.02,
                        fmt(v), ha="center", va="bottom", fontsize=13)
    ax.set_xticks(range(len(KP)))
    ax.set_xticklabels([KP_LAB[k] for k in KP])
    ax.set_xlabel("When the source was killed")
    frame(ax)


def fig_state(runs, figdir):
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    grouped_bars(ax, runs, preserved_pct, lambda v: f"{v:.0f}", 100)
    ax.set_ylabel("State preserved (%)")
    ax.set_ylim(0, 150)          # headroom for a 2-row legend above the 100% bars
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc="upper center", ncol=2, **LEG)
    save(fig, figdir, "exp1_state_preserved")


def fig_downtime(runs, figdir):
    def med_s(rows):
        v = [num(r, "downtime_ms") / 1000 for r in rows if num(r, "robot_back", 0) == 1]
        return st.median(v) if v else np.nan
    tops = [med_s([r for r in events(runs[c]) if r.get("kill_point") == kp])
            for c in runs for kp in KP]
    top = np.nanmax(tops) if tops and not np.all(np.isnan(tops)) else 1.0
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    grouped_bars(ax, runs, med_s, lambda v: f"{v:.1f}", top)
    ax.set_ylabel("Median downtime (s)")
    ax.set_ylim(0, top * 1.6)
    ax.legend(loc="upper center", ncol=2, **LEG)
    save(fig, figdir, "exp1_downtime")


def fig_reward(runs, figdir):
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    waves = set()
    for c in ORDER:
        if c not in runs:
            continue
        xs, ys = curve(runs[c])
        if xs:
            ax.plot(xs, ys, color=COLOR[c], ls=LS[c], lw=2.4, label=LAB[c], zorder=3)
        waves |= {int(x) for x in str(meta(runs[c]).get("migration_rounds") or "")
                  .split(",") if x.strip()}
    for w in sorted(waves):
        ax.axvline(w, color="red", ls=":", lw=1.4, zorder=1)
    ax.set_xlabel("FL round")
    ax.set_ylabel("Eval return")
    frame(ax)
    ax.legend(loc="upper left", **LEG)
    save(fig, figdir, "exp1_reward")


def wave_dips(run):
    """Per migration wave: fleet eval return in the 3 rounds before the wave vs
    the lowest mean in the 10 rounds from the wave on (the stagger spans 5)."""
    xs, ys = curve(run)
    d = dict(zip(xs, ys))
    out = []
    for w in (int(x) for x in str(meta(run).get("migration_rounds") or "").split(",")
              if x.strip()):
        pre = [d[r] for r in range(w - 3, w) if r in d]
        post = [d[r] for r in range(w, w + 10) if r in d]
        if pre and post and st.mean(pre) > 0:
            out.append((w, 100.0 * (min(post) - st.mean(pre)) / st.mean(pre)))
    return out


def print_summary(runs):
    for c in ORDER:
        if c not in runs:
            continue
        rows = events(runs[c])
        m = meta(runs[c])
        print(f"\n=== {LAB[c]} ({c}): {len(rows)} migrations, run {os.path.basename(runs[c])}"
              f", replicas={m.get('replicas')}")
        for kp in KP:
            sel = [r for r in rows if r.get("kill_point") == kp]
            if not sel:
                continue
            back = [r for r in sel if num(r, "robot_back", 0) == 1]
            dt = [num(r, "downtime_ms") / 1000 for r in back]
            rep = [num(r, "replay_buffer_entries_restored", 0) for r in sel
                   if num(r, "restored", 0) == 1]
            served = {}
            for r in sel:
                served[r.get("served_by", "")] = served.get(r.get("served_by", ""), 0) + 1
            print(f"  {kp:14s} n={len(sel):3d} state preserved {preserved_pct(sel):5.1f}% "
                  f"| robot back {100 * len(back) / len(sel):5.1f}% "
                  f"| median downtime {st.median(dt) if dt else float('nan'):6.1f}s "
                  f"| replay restored ~{st.median(rep) if rep else 0:.0f} "
                  f"| served_by {served}")
        if c.startswith("dht"):
            for k in ("replicate_ms", "dht_put_ms", "dht_get_ms"):
                v = [num(r, k) for r in rows if not np.isnan(num(r, k))]
                if v:
                    print(f"  median {k}: {st.median(v):.1f}")
        dips = wave_dips(runs[c])
        if dips:
            print("  fleet eval-return change after each wave: " +
                  ", ".join(f"round {w}: {p:+.1f}%" for w, p in dips))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=DEFAULT_RES,
                    help="exp1_source_failure results root (newest job per condition)")
    ap.add_argument("--figdir", default=DEFAULT_FIG)
    a = ap.parse_args()
    runs = newest_runs(a.results)
    if not runs:
        raise SystemExit(f"no migration_events.csv under {a.results}/<condition>/<jobid>/")
    os.makedirs(a.figdir, exist_ok=True)
    for c, run in runs.items():
        print(f"{c}: {run}")
    fig_state(runs, a.figdir)
    fig_downtime(runs, a.figdir)
    fig_reward(runs, a.figdir)
    print_summary(runs)
    print("\n->", a.figdir)


if __name__ == "__main__":
    main()
