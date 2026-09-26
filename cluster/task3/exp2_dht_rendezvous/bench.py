#!/usr/bin/env python3
"""Exp2 controller: drives the ring hosts' control ports through one phase
(scale | churn | redis_down) and appends every operation to exp2_ops.csv."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import socket
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# fail_count : nodes the run kills (identifies the run; same on every row of it)
# dead_at_op : nodes actually dead when this op ran (churn PUTs run before the
#              kill -> 0; Redis GETs after SHUTDOWN -> 1)
# replicas   : copies written (PUT rows only; empty on GET rows)
FIELDS = ["phase", "trial", "ksize", "ring_size", "fail_count", "dead_at_op",
          "backend", "op", "key_idx", "client_host", "node_g", "ms", "ok",
          "rpcs", "rounds", "timeouts", "replicas", "err"]
WARMUP_KEYS = 20


class HostConn:
    def __init__(self, idx: int, host: str, port: int, connect_timeout: float):
        self.idx, self.host, self.port = idx, host, port
        deadline = time.time() + connect_timeout
        while True:
            try:
                self.sock = socket.create_connection((host, port), timeout=10)
                break
            except OSError:
                if time.time() > deadline:
                    raise RuntimeError(f"ring host {idx} at {host}:{port} never "
                                       f"opened its control port")
                time.sleep(1.0)
        self.sock.settimeout(None)
        self.f = self.sock.makefile("rwb")
        self.hostname = host

    def call(self, req: dict, check: bool = True) -> dict:
        self.f.write((json.dumps(req) + "\n").encode())
        self.f.flush()
        line = self.f.readline()
        if not line:
            raise RuntimeError(f"ring host {self.idx} closed the connection")
        resp = json.loads(line)
        if check and not resp.get("ok"):
            raise RuntimeError(f"ring host {self.idx} {req.get('cmd')}: "
                               f"{resp.get('error')}")
        return resp


class Bench:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.rng = random.Random(a.seed)
        specs = []
        for s in a.hosts.split(","):
            h, p = s.rsplit(":", 1)
            specs.append((h, int(p)))
        self.hosts = [HostConn(i, h, p, a.connect_timeout)
                      for i, (h, p) in enumerate(specs)]
        # Host 0 shares a machine with Redis; queries come from the others,
        # the way robots on worker nodes would issue them.
        self.workers = [1, 2] if len(self.hosts) >= 3 else list(range(len(self.hosts)))
        self.alive: dict[int, list[int]] = {}
        self.tag = f"exp2:{a.phase}:k{a.ksize}:n{a.ring_size}:t{a.trial}:f{a.fail_count}"
        self.rows: list[dict] = []
        self.dead_now = 0          # ring nodes killed so far in this run

    # -- helpers ------------------------------------------------------------ #
    def call_all(self, reqs: dict[int, dict]) -> dict[int, dict]:
        with ThreadPoolExecutor(max_workers=max(1, len(reqs))) as ex:
            futs = {i: ex.submit(self.hosts[i].call, r) for i, r in reqs.items()}
            return {i: f.result() for i, f in futs.items()}

    def ping_all(self) -> dict[int, dict]:
        return self.call_all({i: {"cmd": "ping"} for i in range(len(self.hosts))})

    def wait_ready(self) -> None:
        deadline = time.time() + self.a.ready_timeout
        while True:
            pings = self.ping_all()
            errs = {i: p["error"] for i, p in pings.items() if p.get("error")}
            if errs:
                raise RuntimeError(f"ring failed to start: {errs}")
            if all(p["ready"] for p in pings.values()):
                break
            if time.time() > deadline:
                raise RuntimeError("ring not ready before timeout")
            time.sleep(1.0)
        for i, p in pings.items():
            self.alive[i] = p["alive"]
            self.hosts[i].hostname = p["host"]
        total = sum(len(v) for v in self.alive.values())
        if total != self.a.ring_size:
            raise RuntimeError(f"expected {self.a.ring_size} ring nodes, "
                               f"hosts report {total}")
        print(f"[bench] ring ready: {total} nodes, per host "
              f"{[len(self.alive[i]) for i in sorted(self.alive)]}", flush=True)

    def refresh(self) -> None:
        t = time.time()
        need = max(1, min(self.a.ksize, self.a.ring_size - 1))
        resps = self.call_all({i: {"cmd": "refresh", "min_contacts": need}
                               for i in range(len(self.hosts))})
        mins = [resps[i]["min_contacts"] for i in sorted(resps)]
        meds = [resps[i]["median_contacts"] for i in sorted(resps)]
        rejoined = sum(r["rejoined"] for r in resps.values())
        print(f"[bench] routing tables refreshed in {time.time() - t:.1f}s: "
              f"contacts min/median per host {list(zip(mins, meds))}, "
              f"rejoined {rejoined}", flush=True)
        if min(mins) < need:
            raise RuntimeError(f"overlay not formed: a node has only {min(mins)} "
                               f"contacts (< {need}) after rejoin")

    def value(self, i: int) -> str:
        # Pointer-sized record, like DHT-FRL's {node, path, ts} bundle pointer.
        return json.dumps({
            "node": f"n{i % 97:03d}.cluster",
            "path": f"/checkpoints/robot_{i % 20:03d}",
            "sha1": hashlib.sha1(f"{self.tag}:{i}".encode()).hexdigest(),
        })

    def key(self, prefix: str, i: int) -> str:
        return f"{self.tag}:{prefix}:{i}"

    def other_worker(self, h: int) -> int:
        others = [w for w in self.workers if w != h]
        return self.rng.choice(others) if others else h

    def pick_node(self, h: int) -> int:
        return self.rng.choice(self.alive[h])

    def record(self, backend: str, op: str, host_idx: int,
               results: list[dict], fail_count: int | None = None) -> None:
        # fail_count is only passed for Redis (0 = server up, 1 = shut down),
        # where it is also what was dead at the time of the op.
        fc = self.a.fail_count if fail_count is None else fail_count
        dead = self.dead_now if fail_count is None else fail_count
        for r in results:
            row = {k: r.get(k, "") for k in FIELDS}
            # Redis rows keep the ring size of the run they were measured in
            # (Redis itself is one server) so figures can pair them by run.
            row.update(phase=self.a.phase, trial=self.a.trial,
                       ksize=self.a.ksize if backend == "dht" else "",
                       ring_size=self.a.ring_size,
                       fail_count=fc, dead_at_op=dead, backend=backend, op=op,
                       client_host=self.hosts[host_idx].hostname,
                       ms=round(r["ms"], 4))
            if op == "get":
                row["replicas"] = ""
            self.rows.append(row)

    # -- DHT operations ----------------------------------------------------- #
    def dht_put(self, prefix: str, n: int, record: bool) -> dict[int, int]:
        items: dict[int, list] = {h: [] for h in self.workers}
        putter: dict[int, int] = {}
        for i in range(n):
            h = self.rng.choice(self.workers)
            putter[i] = h
            items[h].append({"key_idx": i, "g": self.pick_node(h),
                             "key": self.key(prefix, i), "value": self.value(i)})
        resps = self.call_all({h: {"cmd": "put_many", "items": it,
                                   "concurrency": self.a.concurrency}
                               for h, it in items.items() if it})
        if record:
            for h, resp in resps.items():
                self.record("dht", "put", h, resp["results"])
        return putter

    def dht_get(self, prefix: str, putter: dict[int, int], record: bool,
                cross_host: bool = True) -> None:
        items: dict[int, list] = {h: [] for h in self.workers}
        for i, ph in putter.items():
            h = self.other_worker(ph) if cross_host else self.rng.choice(self.workers)
            items[h].append({"key_idx": i, "g": self.pick_node(h),
                             "key": self.key(prefix, i), "expect": self.value(i)})
        resps = self.call_all({h: {"cmd": "get_many", "items": it,
                                   "concurrency": self.a.concurrency}
                               for h, it in items.items() if it})
        if record:
            for h, resp in resps.items():
                self.record("dht", "get", h, resp["results"])

    # -- Redis operations ----------------------------------------------------- #
    def redis_op(self, op: str, prefix: str, n: int, putter: dict[int, int] | None,
                 record: bool, fail_count: int | None = None) -> dict[int, int]:
        items: dict[int, list] = {h: [] for h in self.workers}
        out: dict[int, int] = {}
        for i in range(n):
            if op == "set":
                h = self.rng.choice(self.workers)
                out[i] = h
                items[h].append({"key_idx": i, "key": self.key(prefix, i),
                                 "value": self.value(i)})
            else:
                h = self.other_worker(putter[i]) if putter else self.rng.choice(self.workers)
                items[h].append({"key_idx": i, "key": self.key(prefix, i),
                                 "expect": self.value(i)})
        resps = self.call_all({h: {"cmd": "redis", "op": op, "items": it,
                                   "redis_host": self.a.redis_host,
                                   "redis_port": self.a.redis_port,
                                   "timeout": self.a.rpc_timeout,
                                   "concurrency": self.a.concurrency}
                               for h, it in items.items() if it})
        if record:
            for h, resp in resps.items():
                self.record("redis", op if op == "get" else "put", h,
                            resp["results"], fail_count)
        return out

    # -- phases --------------------------------------------------------------- #
    def warmup(self) -> None:
        p = self.dht_put("warm", WARMUP_KEYS, record=False)
        self.dht_get("warm", p, record=False)
        if self.a.redis_host:
            rp = self.redis_op("set", "warm", WARMUP_KEYS, None, record=False)
            self.redis_op("get", "warm", WARMUP_KEYS, rp, record=False)

    def phase_scale(self) -> None:
        self.wait_ready()
        self.refresh()
        self.warmup()
        putter = self.dht_put("k", self.a.keys, record=True)
        self.dht_get("k", putter, record=True)
        if self.a.redis_host:
            rp = self.redis_op("set", "k", self.a.keys, None, record=True)
            self.redis_op("get", "k", self.a.keys, rp, record=True)

    def kill_nodes(self, count: int) -> None:
        if count <= 0:
            return
        # Keep >= 2 live nodes on every worker host so lookups can still be issued.
        cand = [(h, g) for h, gs in self.alive.items() for g in gs]
        self.rng.shuffle(cand)
        left = {h: len(gs) for h, gs in self.alive.items()}
        victims: dict[int, list[int]] = {h: [] for h in self.alive}
        for h, g in cand:
            if sum(len(v) for v in victims.values()) >= count:
                break
            if h in self.workers and left[h] <= 2:
                continue
            victims[h].append(g)
            left[h] -= 1
        n = sum(len(v) for v in victims.values())
        if n < count:
            raise RuntimeError(f"can only kill {n} of the requested {count} nodes")
        resps = self.call_all({h: {"cmd": "stop", "g": v}
                               for h, v in victims.items() if v})
        for h, r in resps.items():
            self.alive[h] = r["alive"]
        self.dead_now += sum(len(v) for v in victims.values())
        print(f"[bench] killed {n} nodes, per host "
              f"{[len(victims[h]) for h in sorted(victims)]}", flush=True)

    def phase_churn(self) -> None:
        self.wait_ready()
        self.refresh()
        self.warmup()
        putter = self.dht_put("k", self.a.keys, record=True)
        self.kill_nodes(self.a.fail_count)
        time.sleep(self.a.settle)
        self.dht_get("k", putter, record=True, cross_host=False)

    def phase_redis_down(self) -> None:
        if not self.a.redis_host:
            raise RuntimeError("redis_down needs --redis-host")
        self.wait_ready()
        rp = self.redis_op("set", "k", self.a.keys, None, record=True, fail_count=0)
        self.redis_op("get", "k", self.a.keys, rp, record=True, fail_count=0)
        res = self.hosts[self.workers[0]].call(
            {"cmd": "redis_shutdown", "redis_host": self.a.redis_host,
             "redis_port": self.a.redis_port})
        if res["still_up"]:
            raise RuntimeError(f"Redis still answers after SHUTDOWN ({res['result']})")
        print(f"[bench] redis server is down ({res['result']})", flush=True)
        time.sleep(1.0)
        self.redis_op("get", "k", self.a.keys, rp, record=True, fail_count=1)

    # -- output --------------------------------------------------------------- #
    def write(self) -> None:
        os.makedirs(self.a.out, exist_ok=True)
        path = os.path.join(self.a.out, "exp2_ops.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(self.rows)
        meta = {k: v for k, v in vars(self.a).items()}
        meta.update(written_rows=len(self.rows), finished=time.time(),
                    hosts_resolved=[h.hostname for h in self.hosts],
                    alive_per_host=[len(self.alive.get(i, []))
                                    for i in range(len(self.hosts))])
        with open(os.path.join(self.a.out, "exp2_meta.jsonl"), "a") as f:
            f.write(json.dumps(meta) + "\n")
        self.summarize()

    def summarize(self) -> None:
        groups: dict[tuple, list[dict]] = {}
        for r in self.rows:
            groups.setdefault((r["backend"], r["op"], r["dead_at_op"]), []).append(r)
        for (b, op, fc), rs in sorted(groups.items(), key=str):
            okms = [r["ms"] for r in rs if r["ok"]]
            succ = 100.0 * sum(1 for r in rs if r["ok"]) / len(rs)
            med = f"{st.median(okms):8.3f} ms" if okms else "     n/a   "
            rpcs = [r["rpcs"] for r in rs if r["ok"] and r["rpcs"] != ""]
            rnd = [r["rounds"] for r in rs if r["ok"] and r["rounds"] != ""]
            extra = (f" rpcs~{st.median(rpcs):.0f} rounds~{st.median(rnd):.0f}"
                     if rpcs and rnd else "")
            print(f"[bench] {self.a.phase:10s} {b:5s} {op:3s} dead={fc!s:>3} "
                  f"n={len(rs):4d} success={succ:6.1f}% median={med}{extra}",
                  flush=True)

    def quit_hosts(self) -> None:
        for h in self.hosts:
            try:
                h.call({"cmd": "quit"}, check=False)
            except Exception:                            # noqa: BLE001
                pass


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hosts", required=True,
                   help="host:ctrl_port,... in ring-host index order (0 = Redis host)")
    p.add_argument("--phase", required=True, choices=["scale", "churn", "redis_down"])
    p.add_argument("--ring-size", type=int, required=True)
    p.add_argument("--ksize", type=int, required=True)
    p.add_argument("--keys", type=int, default=500)
    p.add_argument("--trial", type=int, default=1)
    p.add_argument("--fail-count", type=int, default=0)
    p.add_argument("--settle", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--rpc-timeout", type=float, default=2.0)
    p.add_argument("--redis-host", default="")
    p.add_argument("--redis-port", type=int, default=6679)
    p.add_argument("--connect-timeout", type=float, default=120.0)
    p.add_argument("--ready-timeout", type=float, default=300.0)
    p.add_argument("--out", required=True)
    p.add_argument("--quit-hosts", action="store_true")
    a = p.parse_args()

    b = None
    try:
        b = Bench(a)
        print(f"[bench] phase={a.phase} ring={a.ring_size} k={a.ksize} "
              f"trial={a.trial} fail={a.fail_count} keys={a.keys}", flush=True)
        {"scale": b.phase_scale, "churn": b.phase_churn,
         "redis_down": b.phase_redis_down}[a.phase]()
        b.write()
        return 0
    except Exception as e:                               # noqa: BLE001
        print(f"[bench] FAILED: {e!r}", file=sys.stderr, flush=True)
        if b is not None and b.rows:
            b.write()
        return 1
    finally:
        if b is not None and a.quit_hosts:
            b.quit_hosts()


if __name__ == "__main__":
    sys.exit(main())
