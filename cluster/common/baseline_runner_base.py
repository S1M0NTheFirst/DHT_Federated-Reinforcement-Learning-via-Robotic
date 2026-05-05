"""
Shared scaffolding for the four baseline conditions on the cluster (B, C, D, E).
Each condition supplies only:
  - condition name (str)
  - worker script path (relative to /app inside the apptainer instance)
  - trigger_fn(cfg, redis, robot_id, success_rate_pre, task_counter_pre) → metrics dict

Everything else (Redis init, launching robot instances on both client nodes,
migration monitor loop, live status, completion polling) lives here.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from typing import Callable

import redis

from .cluster_runner import (
    ClusterConfig, MigrationMetricsWriter,
    apptainer_instance_name,
    launch_robot, kill_robot,
    live_status_loop, ssh_run,
)

LOG = logging.getLogger("baseline")


# Tracks the current host node of each robot. Each migration flips it.
_robot_node: dict[str, str] = {}
_robot_lock = threading.Lock()


def current_node(robot_id: str) -> str:
    with _robot_lock:
        return _robot_node[robot_id]


def update_node(robot_id: str, new_node: str) -> None:
    with _robot_lock:
        _robot_node[robot_id] = new_node


def run_baseline(
    *,
    condition: str,
    image: str,
    worker_script: str,
    trigger_fn: Callable,
) -> int:
    cfg = ClusterConfig()
    LOG.info("=" * 78)
    LOG.info("Condition %s — server=%s clients=%s",
             condition, cfg.server_node, cfg.client_nodes)
    LOG.info("NUM_CLIENTS=%d ROBOTS_PER_NODE=%d MIGRATION_OFFSET=%d TOTAL_TASKS=%d",
             cfg.num_clients, cfg.robots_per_node,
             cfg.migration_offset, cfg.total_tasks)
    LOG.info("SIMULATE_CRIU=%s", cfg.simulate_criu)
    LOG.info("=" * 78)

    r = redis.Redis(host=cfg.redis_host, port=cfg.redis_port, decode_responses=True)
    r.flushall()

    writer = MigrationMetricsWriter(condition, cfg.results_dir)

    LOG.info("Launching %d robot instances", cfg.num_clients)
    for cid in range(cfg.num_clients):
        node = cfg.home_node_for_client(cid)
        update_node(f"robot_{cid:03d}", node)
        launch_robot(cfg, node, cid, image, worker_script)
        time.sleep(2)
    LOG.info("All robots launched.")

    threading.Thread(
        target=_migration_loop,
        args=(cfg, r, writer, trigger_fn),
        daemon=True,
    ).start()
    threading.Thread(
        target=live_status_loop,
        args=(cfg, r, writer, 15),
        daemon=True,
    ).start()

    try:
        _wait_for_completion(cfg, r)
    except KeyboardInterrupt:
        LOG.info("Interrupted")
    LOG.info("Migration events written: %d → %s", writer.event_count, writer.path)
    return 0


def _migration_loop(cfg: ClusterConfig, r: redis.Redis,
                    writer: MigrationMetricsWriter,
                    trigger_fn: Callable) -> None:
    LOG.info("[Monitor] watching for migration requests")
    while True:
        try:
            for key in r.keys("migration_request:robot_*"):
                raw = r.get(key)
                if not raw:
                    continue
                info = json.loads(raw)
                robot_id = info["robot_id"]
                r.delete(key)
                metrics = trigger_fn(
                    cfg, r, robot_id,
                    float(info.get("success_rate", 0)),
                    int(info.get("task_counter", 0)),
                )
                writer.write_event(metrics)
        except Exception as e:
            LOG.error("[Monitor] %r", e)
        time.sleep(1)


def _wait_for_completion(cfg: ClusterConfig, r: redis.Redis) -> None:
    LOG.info("[Wait] waiting for all %d robots to finish", cfg.num_clients)
    while True:
        done = sum(1 for cid in range(cfg.num_clients)
                   if r.get(f"robot_done:robot_{cid:03d}"))
        if done >= cfg.num_clients:
            LOG.info("[Wait] all robots done.")
            return
        time.sleep(15)
