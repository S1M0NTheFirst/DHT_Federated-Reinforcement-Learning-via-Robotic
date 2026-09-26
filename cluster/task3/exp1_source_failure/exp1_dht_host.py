#!/usr/bin/env python3
"""Exp1 DHT host: one process per physical node runs that node's slice of a
cross-node Kademlia ring and serves a JSON-lines control port the Exp1 runner
uses to publish / resolve bundle-replica records.

Same library and fixes as task3/exp2_dht_rendezvous/dht_ring_host.py (copied,
not imported, so Exp2 can change independently): corrected neighbour selection,
routed GETs that never answer from local storage, per-op RPC counters,
self-exit when the launcher disappears or after --max-lifetime."""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import heapq
import json
import logging
import os
import random
import socket
import sys
import time

from kademlia.crawling import NodeSpiderCrawl, ValueSpiderCrawl
from kademlia.network import Server
from kademlia.node import Node
from kademlia.protocol import KademliaProtocol
from kademlia.routing import RoutingTable
from kademlia.utils import digest

for _n in ("kademlia", "kademlia.network", "kademlia.protocol",
           "kademlia.crawling", "rpcudp", "rpcudp.protocol"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

LOG = logging.getLogger("exp1_dht")

_OP: contextvars.ContextVar = contextvars.ContextVar("exp1_op", default=None)


def _bump(field: str) -> None:
    c = _OP.get()
    if c is not None:
        c[field] += 1


class FullScanRoutingTable(RoutingTable):
    """kademlia 2.2.3's TableTraverser can return too few (or no) neighbours
    when buckets on one side of the target are empty. Rank every known contact:
    the k closest known contacts, as Kademlia specifies."""

    def find_neighbors(self, node, k=None, exclude=None):
        k = k or self.ksize
        self.buckets[self.get_bucket_for(node)].touch_last_updated()
        peers = [n for b in self.buckets for n in b.get_nodes()
                 if n.id != node.id
                 and (exclude is None or not n.same_home_as(exclude))]
        return heapq.nsmallest(k, peers, key=node.distance_to)


class CountingProtocol(KademliaProtocol):
    def __init__(self, source_node, storage, ksize, wait_timeout):
        super().__init__(source_node, storage, ksize)
        self._wait_timeout = wait_timeout
        self.router = FullScanRoutingTable(self, ksize, source_node)

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
    """Mirror of Server.set_digest; returns (ok, copies written)."""
    dkey = digest(key)
    target = Node(dkey)
    nearest = server.protocol.router.find_neighbors(target)
    if not nearest:
        return False, 0
    nodes = await CountingNodeCrawl(server.protocol, target, nearest,
                                    server.ksize, server.alpha).find()
    if not nodes:
        return False, 0
    local = 0
    if server.node.distance_to(target) < max(n.distance_to(target) for n in nodes):
        server.storage[dkey] = value
        local = 1
    res = await asyncio.gather(*(server.protocol.call_store(n, dkey, value)
                                 for n in nodes))
    acks = sum(1 for r in res if r[0] and r[1])
    return acks + local > 0, acks + local


async def routed_get(server: Server, key: str):
    # Never answers from local storage: every GET crosses the ring.
    target = Node(digest(key))
    nearest = server.protocol.router.find_neighbors(target)
    if not nearest:
        return None
    return await CountingValueCrawl(server.protocol, target, nearest,
                                    server.ksize, server.alpha).find()


def contacts(server: Server) -> int:
    return sum(len(b.nodes) for b in server.protocol.router.buckets)


async def refresh(server: Server) -> None:
    for target_id in (server.node.id, os.urandom(20)):
        target = Node(target_id)
        nearest = server.protocol.router.find_neighbors(target)
        if nearest:
            await NodeSpiderCrawl(server.protocol, target, nearest,
                                  server.ksize, server.alpha).find()


class DHTHost:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.nodes: dict[int, CountingServer] = {}
        self.ready = False
        self.error = ""
        self.hostname = socket.gethostname()
        self.done: asyncio.Event | None = None
        self.boot: tuple[str, int] | None = None

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
                    await self._join(s, g)
            self.ready = True
            LOG.info("host %d ready: %d nodes (g=%d..%d) on %s ports %d..%d",
                     a.host_index, a.n_local, a.id_offset,
                     a.id_offset + a.n_local - 1, self.hostname,
                     a.base_port, a.base_port + a.n_local - 1)
        except Exception as e:                           # noqa: BLE001
            self.error = repr(e)
            LOG.error("ring start failed: %s", self.error)

    async def _join(self, s: CountingServer, g: int) -> None:
        deadline = time.time() + self.a.join_timeout
        while time.time() < deadline:
            await s.bootstrap([self.boot])
            if s.protocol.router.find_neighbors(s.node):
                return
            await asyncio.sleep(1.0)
        raise RuntimeError(f"node {g} could not join via {self.boot[0]}:"
                           f"{self.boot[1]} within {self.a.join_timeout}s "
                           f"(UDP blocked?)")

    async def refresh_all(self, min_contacts: int) -> dict:
        sem = asyncio.Semaphore(8)

        async def one(s):
            async with sem:
                await refresh(s)

        await asyncio.gather(*(one(s) for s in self.nodes.values()))
        rejoined = 0
        for _ in range(3):
            weak = [g for g, s in self.nodes.items()
                    if g != 0 and contacts(s) < min_contacts]
            if not weak:
                break
            for g in weak:
                await self.nodes[g].bootstrap([self.boot])
                await refresh(self.nodes[g])
            rejoined += len(weak)
        sizes = sorted(contacts(s) for s in self.nodes.values())
        return {"min_contacts": sizes[0], "median_contacts": sizes[len(sizes) // 2],
                "rejoined": rejoined}

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

    def _pick(self, req: dict) -> int:
        g = req.get("g")
        if g is not None and int(g) in self.nodes:
            return int(g)
        return random.choice(sorted(self.nodes))

    async def dispatch(self, req: dict) -> dict:
        cmd = req.get("cmd")
        if cmd == "ping":
            return {"ok": 1, "host": self.hostname, "host_index": self.a.host_index,
                    "ready": int(self.ready), "error": self.error,
                    "nodes": sorted(self.nodes)}
        if cmd == "quit":
            return {"ok": 1}
        if self.error:
            return {"ok": 0, "error": f"ring failed: {self.error}"}
        if not self.ready:
            return {"ok": 0, "error": "ring not ready"}
        if cmd == "refresh":
            t = time.perf_counter()
            stats = await self.refresh_all(int(req.get("min_contacts", 1)))
            return {"ok": 1, "ms": (time.perf_counter() - t) * 1000.0, **stats}
        if cmd == "put":
            g = self._pick(req)
            out, ms, c, err = await self._measure(
                lambda: routed_put(self.nodes[g], req["key"], req["value"]))
            ok, copies = out if out else (False, 0)
            return {"ok": int(ok), "g": g, "ms": ms, "copies": copies, **c,
                    "error": err}
        if cmd == "get":
            g = self._pick(req)
            out, ms, c, err = await self._measure(
                lambda: routed_get(self.nodes[g], req["key"]))
            return {"ok": int(out is not None), "g": g, "ms": ms, "value": out,
                    **c, "error": err or ("" if out is not None else "not_found")}
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
    host = DHTHost(a)
    host.done = asyncio.Event()
    srv = await asyncio.start_server(host.handle, a.bind, a.ctrl_port,
                                     limit=1 << 22)
    LOG.info("host %d control port %s:%d", a.host_index, a.bind, a.ctrl_port)
    starter = asyncio.ensure_future(host.start_nodes())

    # Exit when the launcher (the job's SSH session or shell) goes away: killing
    # the local ssh client does not reliably signal a remote tty-less command.
    parent = os.getppid()

    async def orphan_watch() -> None:
        while os.getppid() == parent:
            await asyncio.sleep(2.0)
        LOG.warning("launcher (pid %d) is gone, exiting", parent)
        host.done.set()

    watcher = asyncio.ensure_future(orphan_watch())
    try:
        await asyncio.wait_for(host.done.wait(), a.max_lifetime)
    except asyncio.TimeoutError:
        LOG.warning("max lifetime %ss reached, exiting", a.max_lifetime)
    watcher.cancel()
    starter.cancel()
    for s in host.nodes.values():
        s.stop()
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
    p.add_argument("--rpc-timeout", type=float, default=1.0)
    p.add_argument("--op-timeout", type=float, default=60.0)
    p.add_argument("--join-timeout", type=float, default=180.0)
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--max-lifetime", type=float, default=21600.0)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s [exp1_dht] %(message)s")
    asyncio.run(amain(a))


if __name__ == "__main__":
    main()
