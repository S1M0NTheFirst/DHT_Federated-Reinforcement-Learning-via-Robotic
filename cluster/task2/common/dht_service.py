"""
Real Kademlia DHT overlay for task2 condition dht_frl.

Model: DHT-COORDINATED DIRECT TRANSFER (the BitTorrent/IPFS pattern).
The DHT is a genuine, decentralized store — on migration the source PUTs a small
pointer (which node holds the bundle + path + content hash) into the overlay, and
the destination GETs it back via real Kademlia iterative routing. The ~11 MB
bundle itself then transfers point-to-point (rsync). So:

    DHT overhead  = put_ms + get_ms      (small pointer, ~ms, routed over k-buckets)
    bulk transfer = direct node->node    (== the rsync/scp baselines)

=> end-to-end migration latency ~= direct transfer, and "DHT transport overhead
is trivial" is literally true and measurable (dht_put_ms / dht_get_ms columns).

A REAL overlay is formed: DHT_RING_NODES independent Kademlia `Server` instances
are bootstrapped into one ring. PUT is issued on one node and GET on a DIFFERENT
node, so values actually traverse the routing table (XOR distance, k=20 buckets,
replication) rather than being read from a local dict.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from typing import Optional, Tuple

from kademlia.network import Server

LOG = logging.getLogger("task2_dht")

# kademlia logs every routing hop at INFO — far too noisy for the runner log.
for _n in ("kademlia", "kademlia.network", "kademlia.protocol",
           "kademlia.crawling", "rpcudp"):
    logging.getLogger(_n).setLevel(logging.WARNING)

_singleton: Optional["DHTService"] = None
_singleton_lock = threading.Lock()


class DHTService:
    """A multi-node Kademlia ring running on a private asyncio loop in a daemon
    thread. Exposes synchronous put/get (safe to call from the runner's monitor
    thread) that return the measured Kademlia operation latency in ms."""

    def __init__(self, n_nodes: int, base_port: int, host: str = "127.0.0.1"):
        self.n_nodes = max(2, n_nodes)          # a ring needs >= 2 nodes
        self.base_port = base_port
        self.host = host
        self.servers: list[Server] = []
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._err: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, name="dht-loop",
                                        daemon=True)

    # -- lifecycle --------------------------------------------------------- #
    def start(self, timeout: float = 60.0) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=timeout):
            raise RuntimeError("DHT overlay failed to bootstrap within "
                               f"{timeout}s")
        if self._err:
            raise RuntimeError(f"DHT overlay bootstrap error: {self._err!r}")
        LOG.info("DHT overlay up: %d-node Kademlia ring on %s:%d-%d",
                 self.n_nodes, self.host, self.base_port,
                 self.base_port + self.n_nodes - 1)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._bootstrap())
        except BaseException as e:                # noqa: BLE001
            self._err = e
            self._ready.set()
            return
        self._ready.set()
        self.loop.run_forever()

    async def _bootstrap(self) -> None:
        boot = Server()
        await boot.listen(self.base_port, interface=self.host)
        self.servers.append(boot)
        boot_addr = (self.host, self.base_port)
        for i in range(1, self.n_nodes):
            s = Server()
            await s.listen(self.base_port + i, interface=self.host)
            await s.bootstrap([boot_addr])
            self.servers.append(s)
        # Warm the routing tables: a PUT on node[0] then a GET on node[-1] forces
        # every node to learn the ring before the FIRST real migration, so the
        # first measured lookup isn't a cold-cache outlier (which could otherwise
        # miss and fall back). Best-effort.
        try:
            await self.servers[0].set("__warmup__", "1")
            for _ in range(3):
                if await self.servers[-1].get("__warmup__") is not None:
                    break
                await asyncio.sleep(0.2)
        except Exception:                             # noqa: BLE001
            pass

    # -- operations -------------------------------------------------------- #
    def _put_node(self) -> Server:
        # PUT from the FIRST node.
        return self.servers[0]

    def _get_node(self) -> Server:
        # GET from a DIFFERENT node so the value is resolved through routing,
        # not a local read — this is what makes the measurement a real lookup.
        return self.servers[-1]

    def put(self, key: str, value: dict, timeout: float = 60.0) -> float:
        payload = json.dumps(value)

        async def _do() -> float:
            t = time.perf_counter()
            await self._put_node().set(key, payload)
            return (time.perf_counter() - t) * 1000.0

        fut = asyncio.run_coroutine_threadsafe(_do(), self.loop)
        return fut.result(timeout=timeout)

    def get(self, key: str, timeout: float = 60.0,
            retries: int = 3) -> Tuple[Optional[dict], float]:
        async def _do() -> Tuple[Optional[str], float]:
            t = time.perf_counter()
            raw = None
            # A freshly-populated key can miss on the very first iterative lookup
            # (routing still converging); retry a few times. The measured latency
            # is the full coordination cost until the pointer resolves — honest.
            for _ in range(max(1, retries)):
                raw = await self._get_node().get(key)
                if raw is not None:
                    break
                await asyncio.sleep(0.2)
            return raw, (time.perf_counter() - t) * 1000.0

        fut = asyncio.run_coroutine_threadsafe(_do(), self.loop)
        raw, ms = fut.result(timeout=timeout)
        return (json.loads(raw) if raw else None), ms

    def stop(self) -> None:
        for s in self.servers:
            try:
                s.stop()
            except Exception:                     # noqa: BLE001
                pass
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:                         # noqa: BLE001
            pass


def get_dht() -> DHTService:
    """Lazily start (once) and return the shared overlay. Ring size / base port
    are configurable via DHT_RING_NODES (default 8) and DHT_BASE_PORT
    (default 8600, clear of flower 8570 / redis 6579)."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            n = int(os.environ.get("DHT_RING_NODES", "8"))
            port = int(os.environ.get("DHT_BASE_PORT", "8600"))
            svc = DHTService(n_nodes=n, base_port=port)
            svc.start()
            _singleton = svc
        return _singleton
