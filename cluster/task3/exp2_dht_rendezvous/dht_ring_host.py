#!/usr/bin/env python3
"""Exp2 ring host: one process per physical host runs a slice of a multi-host
Kademlia ring and serves a JSON-lines control port that bench.py drives."""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import inspect
import json
import logging
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from kademlia.crawling import NodeSpiderCrawl, ValueSpiderCrawl
from kademlia.network import Server
from kademlia.node import Node
from kademlia.protocol import KademliaProtocol
from kademlia.utils import digest

# rpcudp logs every RPC timeout at ERROR; timeouts are counted per op instead.
for _n in ("kademlia", "kademlia.network", "kademlia.protocol",
           "kademlia.crawling", "rpcudp", "rpcudp.protocol"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

LOG = logging.getLogger("exp2_host")

# Per-operation counters. Each measured op runs in its own asyncio task, so the
# RPCs it spawns (gathered sub-tasks copy the context) all land in its counter.
_OP: contextvars.ContextVar = contextvars.ContextVar("exp2_op", default=None)


def _bump(field: str) -> None:
    c = _OP.get()
    if c is not None:
        c[field] += 1


class CountingProtocol(KademliaProtocol):
    def __init__(self, source_node, storage, ksize, wait_timeout):
        super().__init__(source_node, storage, ksize)
        self._wait_timeout = wait_timeout

    async def _counted(self, call, *args):
        _bump("rpcs")
        res = await call(*args)
        if not res[0]:
            _bump("timeouts")
        return res

    async def call_find_node(self, *a):
        return await self._counted(super().call_find_node, *a)

    async def call_find_value(self, *a):
        return await self._counted(super().call_find_value, *a)

    async def call_store(self, *a):
        return await self._counted(super().call_store, *a)

    async def call_ping(self, *a):
        return await self._counted(super().call_ping, *a)


class CountingServer(Server):
    def __init__(self, ksize: int, alpha: int, wait_timeout: float):
        super().__init__(ksize=ksize, alpha=alpha)
        self.wait_timeout = wait_timeout

    def _create_protocol(self):
        return CountingProtocol(self.node, self.storage, self.ksize,
                                self.wait_timeout)


class _CountRounds:
    async def _find(self, rpcmethod):
        _bump("rounds")
        return await super()._find(rpcmethod)


class CountingNodeCrawl(_CountRounds, NodeSpiderCrawl):
    pass


class CountingValueCrawl(_CountRounds, ValueSpiderCrawl):
    pass


async def routed_put(server: Server, key: str, value: str):
    """Mirror of Server.set_digest with round counting; returns (ok, replicas)."""
    dkey = digest(key)
    target = Node(dkey)
    nearest = server.protocol.router.find_neighbors(target)
    if not nearest:
        return False, 0
    spider = CountingNodeCrawl(server.protocol, target, nearest,
                               server.ksize, server.alpha)
    nodes = await spider.find()
    if not nodes:
        return False, 0
    local = 0
    biggest = max(n.distance_to(target) for n in nodes)
    if server.node.distance_to(target) < biggest:
        server.storage[dkey] = value
        local = 1
    res = await asyncio.gather(*(server.protocol.call_store(n, dkey, value)
                                 for n in nodes))
    acks = sum(1 for r in res if r[0] and r[1])
    return acks + local > 0, acks + local


async def routed_get(server: Server, key: str):
    # Deliberately skips server.storage: Server.get() returns a local replica
    # without routing (task2's 13 us "lookup"). Here every GET crosses the ring.
    target = Node(digest(key))
    nearest = server.protocol.router.find_neighbors(target)
    if not nearest:
        return None
    spider = CountingValueCrawl(server.protocol, target, nearest,
                                server.ksize, server.alpha)
    return await spider.find()


def contacts(server: Server) -> int:
    return sum(len(b.nodes) for b in server.protocol.router.buckets)


async def refresh(server: Server) -> None:
    for target_id in (server.node.id, os.urandom(20)):
        target = Node(target_id)
        nearest = server.protocol.router.find_neighbors(target)
        if nearest:
            await NodeSpiderCrawl(server.protocol, target, nearest,
                                  server.ksize, server.alpha).find()


def _redis_client(host: str, port: int, timeout: float):
    import redis
    kw = dict(host=host, port=port, socket_timeout=timeout,
              socket_connect_timeout=timeout)
    params = inspect.signature(redis.Redis.__init__).parameters
    # redis-py >= 6 retries failed commands by default; a coordinator outage
    # must show up as a failed lookup, not be hidden behind retries.
    if "retry" in params:
        kw["retry"] = None
    # redis-py 8 defaults to RESP3 (HELLO 3), which Redis < 6 rejects.
    if "protocol" in params:
        kw["protocol"] = 2
    return redis.Redis(**kw)


class RingHost:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.nodes: dict[int, CountingServer] = {}
        self.dead: set[int] = set()
        self.ready = False
        self.error = ""
        self.hostname = socket.gethostname()
        self.done: asyncio.Event | None = None
        self.boot: tuple[str, int] | None = None

    # -- ring lifecycle ----------------------------------------------------- #
    async def start_nodes(self) -> None:
        a = self.a
        try:
            self.boot = (socket.gethostbyname(a.bootstrap_host), a.bootstrap_port)
            for i in range(a.n_local):
                g = a.id_offset + i
                s = CountingServer(a.ksize, a.alpha, a.rpc_timeout)
                await s.listen(a.base_port + i, interface=a.bind)
                self.nodes[g] = s
                if g != 0:
                    await self._join(s, self.boot, g)
            self.ready = True
            LOG.info("host %d ready: %d nodes (g=%d..%d) on %s ports %d..%d",
                     a.host_index, a.n_local, a.id_offset,
                     a.id_offset + a.n_local - 1, self.hostname,
                     a.base_port, a.base_port + a.n_local - 1)
        except Exception as e:                           # noqa: BLE001
            self.error = repr(e)
            LOG.error("ring start failed: %s", self.error)

    async def _join(self, s: CountingServer, boot, g: int) -> None:
        deadline = time.time() + self.a.join_timeout
        while time.time() < deadline:
            await s.bootstrap([boot])
            if s.protocol.router.find_neighbors(s.node):
                return
            await asyncio.sleep(1.0)
        raise RuntimeError(f"node {g} could not join via {boot[0]}:{boot[1]} "
                           f"within {self.a.join_timeout}s (UDP blocked?)")

    async def refresh_all(self, min_contacts: int) -> dict:
        sem = asyncio.Semaphore(8)

        async def one(s):
            async with sem:
                await refresh(s)

        ids = self.alive_ids()
        await asyncio.gather(*(one(self.nodes[g]) for g in ids))
        rejoined = 0
        for _ in range(3):
            weak = [g for g in ids if g != 0
                    and contacts(self.nodes[g]) < min_contacts]
            if not weak:
                break
            for g in weak:
                await self.nodes[g].bootstrap([self.boot])
                await refresh(self.nodes[g])
            rejoined += len(weak)
        sizes = sorted(contacts(self.nodes[g]) for g in ids)
        return {"min_contacts": sizes[0], "median_contacts": sizes[len(sizes) // 2],
                "rejoined": rejoined}

    def _alive(self, g: int) -> bool:
        return g in self.nodes and g not in self.dead

    def alive_ids(self) -> list[int]:
        return sorted(g for g in self.nodes if g not in self.dead)

    def stop_all(self) -> None:
        for g, s in self.nodes.items():
            if g not in self.dead:
                s.stop()
                self.dead.add(g)

    # -- measured operations ------------------------------------------------ #
    async def _measure(self, fn):
        counter = {"rpcs": 0, "rounds": 0, "timeouts": 0}
        token = _OP.set(counter)
        t = time.perf_counter()
        out, err = None, ""
        try:
            out = await asyncio.wait_for(fn(), self.a.op_timeout)
        except asyncio.TimeoutError:
            err = "op_timeout"
        except Exception as e:                           # noqa: BLE001
            err = type(e).__name__
        finally:
            _OP.reset(token)
        return out, (time.perf_counter() - t) * 1000.0, counter, err

    async def put_many(self, items: list[dict], conc: int) -> list[dict]:
        sem = asyncio.Semaphore(conc)

        async def one(it):
            async with sem:
                g = it["g"]
                if not self._alive(g):
                    return {"key_idx": it["key_idx"], "node_g": g, "ms": 0.0,
                            "ok": 0, "replicas": 0, "rpcs": 0, "rounds": 0,
                            "timeouts": 0, "err": "dead_node"}
                out, ms, c, err = await self._measure(
                    lambda: routed_put(self.nodes[g], it["key"], it["value"]))
                ok, replicas = out if out else (False, 0)
                return {"key_idx": it["key_idx"], "node_g": g, "ms": ms,
                        "ok": int(ok), "replicas": replicas, **c, "err": err}

        return list(await asyncio.gather(*(one(it) for it in items)))

    async def get_many(self, items: list[dict], conc: int) -> list[dict]:
        sem = asyncio.Semaphore(conc)

        async def one(it):
            async with sem:
                g = it["g"]
                if not self._alive(g):
                    return {"key_idx": it["key_idx"], "node_g": g, "ms": 0.0,
                            "ok": 0, "replicas": 0, "rpcs": 0, "rounds": 0,
                            "timeouts": 0, "err": "dead_node"}
                out, ms, c, err = await self._measure(
                    lambda: routed_get(self.nodes[g], it["key"]))
                ok = int(out is not None and out == it["expect"])
                if not err and out is None:
                    err = "not_found"
                elif not err and not ok:
                    err = "wrong_value"
                return {"key_idx": it["key_idx"], "node_g": g, "ms": ms,
                        "ok": ok, "replicas": 0, **c, "err": err}

        return list(await asyncio.gather(*(one(it) for it in items)))

    def redis_batch(self, op: str, items: list[dict], host: str, port: int,
                    timeout: float, conc: int) -> list[dict]:
        r = _redis_client(host, port, timeout)

        def one(it):
            t = time.perf_counter()
            ok, err = 0, ""
            try:
                if op == "set":
                    ok = int(bool(r.set(it["key"], it["value"])))
                else:
                    v = r.get(it["key"])
                    v = v.decode() if v is not None else None
                    ok = int(v is not None and v == it["expect"])
                    if v is None:
                        err = "not_found"
            except Exception as e:                       # noqa: BLE001
                err = type(e).__name__
            return {"key_idx": it["key_idx"], "node_g": -1,
                    "ms": (time.perf_counter() - t) * 1000.0, "ok": ok,
                    "replicas": 1, "rpcs": 1, "rounds": 1,
                    "timeouts": int(bool(err) and err != "not_found"),
                    "err": err}

        with ThreadPoolExecutor(max_workers=max(1, conc)) as ex:
            return list(ex.map(one, items))

    def redis_shutdown(self, host: str, port: int, timeout: float) -> dict:
        try:
            _redis_client(host, port, timeout).shutdown(nosave=True)
            result = "ok"
        except Exception as e:                           # noqa: BLE001
            # A real server drops the connection as it exits.
            result = type(e).__name__
        time.sleep(1.0)
        try:
            still_up = bool(_redis_client(host, port, timeout).ping())
        except Exception:                                # noqa: BLE001
            still_up = False
        return {"result": result, "still_up": still_up}

    # -- control protocol --------------------------------------------------- #
    async def dispatch(self, req: dict) -> dict:
        cmd = req.get("cmd")
        if cmd == "ping":
            return {"ok": 1, "host": self.hostname,
                    "host_index": self.a.host_index, "ready": int(self.ready),
                    "error": self.error, "alive": self.alive_ids(),
                    "dead": sorted(self.dead)}
        if cmd == "quit":
            return {"ok": 1}
        if self.error:
            return {"ok": 0, "error": f"ring failed: {self.error}"}
        if not self.ready:
            return {"ok": 0, "error": "ring not ready"}
        loop = asyncio.get_running_loop()
        conc = int(req.get("concurrency", 16))
        if cmd == "refresh":
            t = time.perf_counter()
            stats = await self.refresh_all(int(req.get("min_contacts", 1)))
            return {"ok": 1, "ms": (time.perf_counter() - t) * 1000.0, **stats}
        if cmd == "put_many":
            return {"ok": 1, "host": self.hostname,
                    "results": await self.put_many(req["items"], conc)}
        if cmd == "get_many":
            return {"ok": 1, "host": self.hostname,
                    "results": await self.get_many(req["items"], conc)}
        if cmd == "stop":
            for g in req["g"]:
                if self._alive(g):
                    self.nodes[g].stop()
                    self.dead.add(g)
            return {"ok": 1, "alive": self.alive_ids(), "dead": sorted(self.dead)}
        if cmd == "redis":
            res = await loop.run_in_executor(
                None, self.redis_batch, req["op"], req["items"],
                req["redis_host"], int(req["redis_port"]),
                float(req.get("timeout", self.a.rpc_timeout)), conc)
            return {"ok": 1, "host": self.hostname, "results": res}
        if cmd == "redis_shutdown":
            res = await loop.run_in_executor(
                None, self.redis_shutdown, req["redis_host"],
                int(req["redis_port"]), float(req.get("timeout", 5.0)))
            return {"ok": 1, **res}
        return {"ok": 0, "error": f"unknown cmd {cmd!r}"}

    async def handle(self, reader: asyncio.StreamReader,
                     writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    req = json.loads(line)
                    resp = await self.dispatch(req)
                except Exception as e:                   # noqa: BLE001
                    req, resp = {}, {"ok": 0, "error": repr(e)}
                writer.write((json.dumps(resp) + "\n").encode())
                await writer.drain()
                if req.get("cmd") == "quit":
                    self.done.set()
                    break
        finally:
            writer.close()


async def amain(a: argparse.Namespace) -> None:
    host = RingHost(a)
    host.done = asyncio.Event()
    srv = await asyncio.start_server(host.handle, a.bind, a.ctrl_port,
                                     limit=1 << 26)
    LOG.info("host %d control port %s:%d", a.host_index, a.bind, a.ctrl_port)
    starter = asyncio.ensure_future(host.start_nodes())
    try:
        await asyncio.wait_for(host.done.wait(), a.max_lifetime)
    except asyncio.TimeoutError:
        LOG.warning("max lifetime %ss reached, exiting", a.max_lifetime)
    starter.cancel()
    host.stop_all()
    srv.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host-index", type=int, required=True)
    p.add_argument("--n-local", type=int, required=True)
    p.add_argument("--id-offset", type=int, required=True)
    p.add_argument("--base-port", type=int, required=True)
    p.add_argument("--ctrl-port", type=int, required=True)
    p.add_argument("--bootstrap-host", required=True)
    p.add_argument("--bootstrap-port", type=int, required=True)
    p.add_argument("--ksize", type=int, default=3)
    p.add_argument("--alpha", type=int, default=3)
    p.add_argument("--rpc-timeout", type=float, default=2.0)
    p.add_argument("--op-timeout", type=float, default=60.0)
    p.add_argument("--join-timeout", type=float, default=180.0)
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--max-lifetime", type=float, default=1800.0)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s [exp2_host] %(message)s")
    if sys.platform == "win32":
        # Local smoke test only: on the Proactor loop, one ICMP port-unreachable
        # from a killed peer ends the UDP read loop and the node goes deaf.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(amain(a))


if __name__ == "__main__":
    main()
