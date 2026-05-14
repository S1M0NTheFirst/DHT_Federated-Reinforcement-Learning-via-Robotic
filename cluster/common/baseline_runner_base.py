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
from typing import Any, Callable, Optional

import redis

from .cluster_runner import (
    ClusterConfig, MigrationMetricsWriter,
    apptainer_instance_name,
    install_signal_handlers, terminate_all_tracked,
    launch_robot, kill_robot,
    live_status_loop, ssh_run,
    establish_ssh_master, close_ssh_master, close_all_ssh_masters_fast,
)
import traceback

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
    initial_extra_env: Optional[dict] = None,
    pre_loop: Optional[Callable[[ClusterConfig, Any], None]] = None,
) -> int:
    # Install SIGTERM/SIGINT handlers + atexit hook BEFORE we start any
    # SSH/apptainer Popens, so a fast-arriving signal still cleans them up.
    install_signal_handlers()

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

    # Pre-establish a per-cid SSH master on EACH client node for EVERY robot.
    # Each cid gets one master per client node (so 2 masters per cid, total
    # 2 * NUM_CLIENTS masters). Migration relaunches then reuse the existing
    # master on the destination — they NEVER open a new TCP, which keeps us
    # immune to sshd MaxStartups bursts during clustered migration waves.
    # Pre-establishing serially with 1s gap stays well under MaxStartups.
    total_masters = cfg.num_clients * len(cfg.client_nodes)
    LOG.info("Pre-establishing %d SSH masters (%d cids × %d nodes)",
             total_masters, cfg.num_clients, len(cfg.client_nodes))
    for cid in range(cfg.num_clients):
        for client_node in cfg.client_nodes:
            establish_ssh_master(client_node, cid)
            time.sleep(1)

    LOG.info("Launching %d robot instances", cfg.num_clients)
    for cid in range(cfg.num_clients):
        node = cfg.home_node_for_client(cid)
        update_node(f"robot_{cid:03d}", node)
        launch_robot(cfg, node, cid, image, worker_script,
                     extra_env=initial_extra_env)
        # 0.5s spacing — robot launches are slaves on pre-established
        # per-cid masters, so this is a near-instant socket handshake.
        time.sleep(0.5)
    LOG.info("All robots launched.")

    if pre_loop is not None:
        pre_loop(cfg, r)

    threading.Thread(
        target=_migration_loop,
        args=(cfg, r, writer, trigger_fn, image, worker_script),
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
    finally:
        # Terminate every tracked SSH/apptainer Popen before we exit.
        # The atexit hook also does this, but calling it explicitly here
        # gives us a clear log line and guarantees cleanup even if the
        # interpreter is shut down abnormally.
        terminate_all_tracked(hard_after=5.0)
        # Close every per-cid SSH master (fire-and-forget). Then nuke any
        # stragglers so the bash wrapper can exit quickly and MOAB marks the
        # job complete.
        for cid in range(cfg.num_clients):
            for client_node in cfg.client_nodes:
                close_ssh_master(client_node, cid)
        time.sleep(2)  # let the -O exit messages reach the masters
        close_all_ssh_masters_fast()
        LOG.info("Shutdown complete — exiting runner cleanly.")
    LOG.info("Migration events written: %d → %s", writer.event_count, writer.path)
    return 0


def _recover_stranded_robot(cfg: ClusterConfig, r: redis.Redis, robot_id: str,
                            image: str, worker_script: str) -> None:
    """Force-progress a robot whose trigger_fn raised mid-migration. Kill on
    current src node (in case the worker is still sleeping there waiting for
    SIGTERM), then relaunch on the other node so it keeps running and will
    eventually set robot_done. Without this, an SSH timeout during the
    trigger_fn leaves the worker stranded and _wait_for_completion hangs."""
    try:
        cid = int(robot_id.split("_")[1])
        src = current_node(robot_id)
        dst = cfg.other_node(src)
        LOG.warning("[Recover] %s: force kill on %s then relaunch on %s",
                    robot_id, src, dst)
        try:
            kill_robot(cfg, src, cid)
        except Exception as e:
            LOG.error("[Recover] kill_robot %s failed: %r", robot_id, e)
        time.sleep(2)
        try:
            launch_robot(cfg, dst, cid, image, worker_script)
            update_node(robot_id, dst)
            r.set(f"migration_done:{robot_id}", "1", ex=600)
            LOG.warning("[Recover] %s relaunched on %s (no metrics row)",
                        robot_id, dst)
        except Exception as e:
            LOG.error("[Recover] launch_robot %s on %s failed: %r — robot lost",
                      robot_id, dst, e)
            r.set(f"robot_done:{robot_id}", "1")
    except Exception as e:
        LOG.error("[Recover] unhandled failure for %s: %r", robot_id, e)


def _migration_loop(cfg: ClusterConfig, r: redis.Redis,
                    writer: MigrationMetricsWriter,
                    trigger_fn: Callable,
                    image: str, worker_script: str) -> None:
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
                try:
                    metrics = trigger_fn(
                        cfg, r, robot_id,
                        float(info.get("success_rate", 0)),
                        int(info.get("task_counter", 0)),
                    )
                    writer.write_event(metrics)
                except Exception as e:
                    # trigger_fn failed mid-migration (typically SSH timeout
                    # during the dump/rsync step). The worker may still be
                    # sleeping on src waiting for kill, so it will never set
                    # robot_done and the runner would hang. Recover by force
                    # kill+relaunch on the other node so the experiment
                    # continues, then skip writing this event.
                    LOG.error("[Monitor] trigger_fn failed for %s: %r\n%s",
                              robot_id, e, traceback.format_exc())
                    _recover_stranded_robot(cfg, r, robot_id, image, worker_script)
                # Throttle: spread out migration relaunches so multiple fresh
                # TCPs to the same dst node don't pile up and trip sshd's
                # MaxStartups rate limit. Saw 5-6 robots/run die during the
                # task-200 / task-260 / task-400 migration waves without this.
                # 6s leaves ~10 migrations/minute peak which is comfortable.
                time.sleep(6)
        except Exception as e:
            LOG.error("[Monitor] %r", e)
        time.sleep(1)


def _wait_for_completion(cfg: ClusterConfig, r: redis.Redis,
                         stall_timeout_s: int = 1800) -> None:
    """Wait for every robot to set robot_done in redis. The stall guard only
    activates AFTER the first robot finishes — before that, done=0 is
    expected (cold start through 2000 tasks takes ~40 min). Once at least
    one robot completes, if no further progress is made for stall_timeout_s,
    we give up rather than hang the MOAB job forever.
    """
    LOG.info("[Wait] waiting for all %d robots to finish", cfg.num_clients)
    last_done = 0
    last_change = time.time()
    while True:
        done = sum(1 for cid in range(cfg.num_clients)
                   if r.get(f"robot_done:robot_{cid:03d}"))
        if done >= cfg.num_clients:
            LOG.info("[Wait] all robots done.")
            return
        if done != last_done:
            last_done = done
            last_change = time.time()
        elif done > 0 and time.time() - last_change > stall_timeout_s:
            LOG.error("[Wait] %d/%d robots done — no progress for %ds, giving up",
                      done, cfg.num_clients, stall_timeout_s)
            return
        time.sleep(15)
