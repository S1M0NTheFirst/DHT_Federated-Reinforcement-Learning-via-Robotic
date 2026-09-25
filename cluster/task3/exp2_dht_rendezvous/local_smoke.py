#!/usr/bin/env python3
"""Run a tiny Exp2 on one machine (3 ring hosts on 127.0.0.1) to check the
harness end to end before submitting run.sh to the cluster. Pass --redis-port
to include the Redis baseline against a local server (the last phase shuts
that server down)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HOST_PY = os.path.join(HERE, "dht_ring_host.py")
BENCH_PY = os.path.join(HERE, "bench.py")
BASE, CTRL = 19700, 19690


def start_ring(n: int, k: int, logdir: str, label: str) -> list[subprocess.Popen]:
    per, rem = divmod(n, 3)
    procs, off = [], 0
    for h in range(3):
        n_h = per + (1 if h < rem else 0)
        log = open(os.path.join(logdir, f"ring_{label}_h{h}.log"), "w")
        procs.append(subprocess.Popen(
            [sys.executable, "-u", HOST_PY, "--host-index", str(h),
             "--n-local", str(n_h), "--id-offset", str(off),
             "--base-port", str(BASE + 100 * h), "--ctrl-port", str(CTRL + h),
             "--bootstrap-host", "127.0.0.1", "--bootstrap-port", str(BASE),
             "--ksize", str(k), "--rpc-timeout", "1.0", "--bind", "127.0.0.1",
             "--max-lifetime", "600"],
            stdout=log, stderr=subprocess.STDOUT))
        off += n_h
    return procs


def stop_ring(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def bench(out: str, *args: str) -> int:
    hosts = ",".join(f"127.0.0.1:{CTRL + h}" for h in range(3))
    return subprocess.call([sys.executable, "-u", BENCH_PY, "--hosts", hosts,
                            "--out", out, "--quit-hosts", "--rpc-timeout", "1.0",
                            *args])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--sizes", default="8,16,32")
    ap.add_argument("--keys", type=int, default=60)
    ap.add_argument("--churn-n", type=int, default=24)
    ap.add_argument("--churn-fail", default="0,1,6,12,18")
    ap.add_argument("--churn-k", default="3,8")
    ap.add_argument("--redis-port", type=int, default=0)
    a = ap.parse_args()
    out = a.out or tempfile.mkdtemp(prefix="exp2_smoke_")
    logdir = os.path.join(out, "logs")
    os.makedirs(logdir, exist_ok=True)
    redis = (["--redis-host", "127.0.0.1", "--redis-port", str(a.redis_port)]
             if a.redis_port else [])
    failures = 0
    t0 = time.time()

    for n in map(int, a.sizes.split(",")):
        procs = start_ring(n, 3, logdir, f"scale_n{n}")
        failures += bench(out, "--phase", "scale", "--ring-size", str(n),
                          "--ksize", "3", "--keys", str(a.keys), "--seed", str(n),
                          *redis) != 0
        stop_ring(procs)

    for k in map(int, a.churn_k.split(",")):
        for c in map(int, a.churn_fail.split(",")):
            procs = start_ring(a.churn_n, k, logdir, f"churn_k{k}_f{c}")
            failures += bench(out, "--phase", "churn", "--ring-size", str(a.churn_n),
                              "--ksize", str(k), "--fail-count", str(c),
                              "--keys", str(a.keys), "--settle", "1",
                              "--seed", str(100 * k + c)) != 0
            stop_ring(procs)

    if redis:
        procs = start_ring(6, 3, logdir, "redis_down")
        failures += bench(out, "--phase", "redis_down", "--ring-size", "6",
                          "--ksize", "3", "--keys", str(a.keys), *redis) != 0
        stop_ring(procs)

    print(f"\nsmoke finished in {time.time() - t0:.0f}s, failed phases: {failures}")
    print(f"results: {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
