"""Exp1 migration state machine, free of cluster I/O so it can be unit-tested.

The runner supplies an `ops` object with the real actions (ssh, rsync/scp,
DHT control calls, SIGKILL, relaunch); tests supply fakes. One call of
run_migration() handles one migration event end to end:

  1. wait until the source worker has saved its bundle
  2. dht conditions: replicate the bundle to the replica nodes and publish the
     replica set in the DHT (this is part of DHT-FRL's checkpoint step)
  3. kill the source at the event's kill point (SIGKILL + delete its bundle)
  4. destination resolves where the bundle lives (DHT lookup, or "the source"
     for the baselines) and pulls each file from the first copy that works
  5. relaunch the robot on the destination: restore if the bundle is complete,
     otherwise start cold (the robot must rejoin FedAvg either way)

app_warm (pre-copy): EXP1_PRECOPY_ROUNDS before the migration the worker
writes a second bundle (<robot>.pre) and the runner copies it to the
destination in the background (not downtime). At the migration the final
bundle is synced INTO A STAGING DIR seeded with the pre-copy (rsync then only
moves the delta); only a complete staging dir replaces the bundle, so a sync
cut short never leaves a mix of old and new files. If the source dies before
the final sync completes, the destination restores the older pre-copy:
restored_stale=1, with its age in rounds and replay entries.
"""
from __future__ import annotations

import csv
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable

# Written by the worker in this order (manifest last).
FILES = ("sac_state.pt", "replay_buffer.pkl", "manifest.json")
KILL_POINTS = ("after_save", "mid_transfer", "after_transfer")


@dataclass(frozen=True)
class Spec:
    dht: bool            # publish/resolve the bundle location through the DHT
    transport: str       # rsync | scp | none
    restore: bool        # try to carry state (False = cold restart)
    replicas: bool       # place extra bundle copies (dht_frl only)
    precopy: bool = False  # background pre-copy + final delta sync (app_warm)


CONDITIONS = {
    "dht_frl": Spec(dht=True, transport="rsync", restore=True, replicas=True),
    "dht_r0": Spec(dht=True, transport="rsync", restore=True, replicas=False),
    "app_cold": Spec(dht=False, transport="rsync", restore=True, replicas=False),
    "app_warm": Spec(dht=False, transport="rsync", restore=True, replicas=False,
                     precopy=True),
    "tcp_scp": Spec(dht=False, transport="scp", restore=True, replicas=False),
    "cold_restart": Spec(dht=False, transport="none", restore=False, replicas=False),
}
# Same checkpoint_mode names as task2 where the condition exists there.
CHECKPOINT_MODE = {"dht_frl": "dht_replicated", "dht_r0": "dht_bundle",
                   "app_cold": "app", "app_warm": "app_warm",
                   "tcp_scp": "tcp", "cold_restart": "none"}


def parse_kill_points(s: str) -> tuple[str, ...]:
    pts = tuple(p.strip() for p in s.split(",") if p.strip())
    bad = [p for p in pts if p not in KILL_POINTS]
    if bad or not pts:
        raise ValueError(f"bad kill points {s!r}; choose from {KILL_POINTS}")
    return pts


def kill_point_for(cid: int, migration_index: int, points: tuple[str, ...]) -> str:
    """Rotates per robot and per wave; identical in every condition."""
    return points[(cid + migration_index) % len(points)]


def replica_nodes(src: str, dst: str, nodes: list[str], r: int,
                  exclude_dst: bool = True) -> list[str]:
    """Nodes that get a bundle copy: never the source; third nodes first, the
    destination only when allowed and still short of r."""
    third = [n for n in nodes if n not in (src, dst)]
    cands = third + ([] if exclude_dst or dst == src else [dst])
    return cands[:max(0, r)]


def record_key(robot_id: str, migration_index: int, ns: str = "") -> str:
    # One key per migration (and per condition, when several conditions share
    # one ring in a job): a lookup can never return a stale replica set.
    return f"bundle:{ns}:{robot_id}:m{migration_index}"


@dataclass
class Event:
    robot_id: str
    cid: int
    migration_index: int
    src: str
    dst: str
    nodes: list[str]
    kill_point: str
    src_bundle: str
    dst_bundle: str
    replica_dir: Callable[[str], str]      # node -> bundle dir for this robot
    n_replicas: int = 1
    exclude_dst: bool = True
    ns: str = ""                           # DHT key namespace (condition tag)
    fl_round: int = 0                      # round of the migration request
    replay_entries_pre: int = -1           # replay entries in the final bundle


def run_migration(ev: Event, spec: Spec, ops) -> dict:
    """Returns the metrics row (task2 migration_events columns + Exp1 columns)."""
    now = ops.now
    out: dict = {
        "robot_id": ev.robot_id, "src_node": ev.src, "dst_node": ev.dst,
        "kill_point": ev.kill_point, "migration_index": ev.migration_index,
        "fault_injected": 1, "replicas_placed": 0, "replicate_ms": 0.0,
        "dht_put_ms": 0.0, "dht_get_ms": 0.0, "dht_found": -1,
        "fetch_complete": 0, "files_from_src": 0, "files_from_replica": 0,
        "fetch_attempts": 0, "served_by": "none", "retry_count": 0,
        "network_bytes_transferred": 0, "checkpoint_size_mb": 0.0,
        "kill_at_ms": -1.0, "kill_ms": -1.0, "kill_ok": -1, "restored": 0,
        "restore_error": "", "robot_back": 0,
        "restored_stale": 0, "stale_rounds": "", "stale_replay_entries": "",
        "precopy_ok": "", "precopy_ms": "", "precopy_round": "",
        "background_bandwidth_mb": "", "replay_entries_pre": ev.replay_entries_pre,
    }
    t0 = now()
    ops.wait_saved()
    t_saved = now()
    out["trigger_to_dump_ms"] = (t_saved - t0) * 1000

    killed = {"done": False}

    def kill() -> None:
        if not killed["done"]:
            t = now()
            ok = ops.kill_source()
            killed["done"] = True
            out["kill_ms"] = (now() - t) * 1000          # the kill step itself
            out["kill_at_ms"] = (now() - t0) * 1000       # source confirmed dead
            out["kill_ok"] = int(bool(ok))

    src_loc = {"node": ev.src, "path": ev.src_bundle, "role": "src"}
    sizes: dict = {}
    key = record_key(ev.robot_id, ev.migration_index, ev.ns)
    if spec.restore:
        sizes = ops.stat_files(ev.src, ev.src_bundle)
        out["checkpoint_size_mb"] = round(sum(sizes.values()) / (1024 * 1024), 3)

    if spec.dht:
        locs = [src_loc]
        if spec.replicas and ev.n_replicas > 0:
            t = now()
            for n in replica_nodes(ev.src, ev.dst, ev.nodes, ev.n_replicas,
                                   ev.exclude_dst):
                path = ev.replica_dir(n)
                if ops.copy_dir(ev.src, ev.src_bundle, n, path):
                    locs.append({"node": n, "path": path, "role": "replica"})
            out["replicate_ms"] = (now() - t) * 1000
            out["replicas_placed"] = len(locs) - 1
        put = ops.dht_put(key, json.dumps({"files": sizes, "locs": locs}), ev.src)
        out["dht_put_ms"] = float(put.get("ms", 0.0))

    if ev.kill_point == "after_save":
        kill()

    # app_warm: the background pre-copy (if it finished) seeds a staging dir.
    pre = ops.precopy_status() if spec.precopy else None
    if spec.precopy:
        out["precopy_ok"] = int(bool(pre and pre.get("ok")))
        if pre:
            out["precopy_ms"] = round(float(pre.get("ms", 0.0)), 3)
            out["precopy_round"] = pre.get("round", "")
            out["background_bandwidth_mb"] = round(
                float(pre.get("bytes", 0)) / (1024 * 1024), 3)
    use_pre = bool(pre and pre.get("ok"))
    fetch_dir = ev.dst_bundle + ".new" if use_pre else ev.dst_bundle

    complete = False
    stale = False
    t_fetch = now()
    if spec.restore:
        # No pre-fetch clear of the destination: a restore happens only if all
        # files arrived in THIS fetch, and the dir is cleared below otherwise.
        if use_pre and not ops.seed_dir(ev.dst, pre["dir"], fetch_dir):
            use_pre, fetch_dir = False, ev.dst_bundle
        fetch_locs, expect = [src_loc], dict(sizes)
        if spec.dht:
            got = ops.dht_get(key, ev.dst)
            out["dht_get_ms"] = float(got.get("ms", 0.0))
            out["dht_found"] = int(bool(got.get("ok")))
            if got.get("ok"):
                rec = json.loads(got["value"])
                fetch_locs = rec["locs"]
                expect = rec.get("files") or expect
        served, failed = [], set()
        for i, name in enumerate(FILES):
            ok = False
            for j, loc in enumerate(fetch_locs):
                # A copy that already failed (its node lost the bundle, or the
                # copy timed out after retries) is not tried again for the
                # remaining files: that re-paid the failure for every file.
                if j in failed:
                    continue
                out["fetch_attempts"] += 1
                if ops.copy_file(loc["node"], f"{loc['path']}/{name}", ev.dst,
                                 f"{fetch_dir}/{name}", expect.get(name),
                                 spec.transport):
                    served.append(loc.get("role", "src"))
                    out["network_bytes_transferred"] += int(expect.get(name, 0))
                    ok = True
                    break
                failed.add(j)
            if not ok:
                break
            if i == 0 and ev.kill_point == "mid_transfer":
                kill()
        complete = len(served) == len(FILES)
        if use_pre:
            # Only a complete final sync replaces the bundle; otherwise the
            # older, complete pre-copy is what the destination has.
            if complete:
                complete = ops.promote_dir(ev.dst, fetch_dir, ev.dst_bundle)
            elif ops.promote_dir(ev.dst, pre["dir"], ev.dst_bundle):
                stale = True
        out["fetch_complete"] = int(complete)
        out["files_from_src"] = served.count("src")
        out["files_from_replica"] = served.count("replica")
        out["served_by"] = "+".join(sorted(set(served))) if served else "none"
        if stale:
            out["served_by"] = "precopy"
    out["dump_to_transfer_ms"] = (now() - t_fetch) * 1000

    kill()   # after_transfer, or whatever has not died yet: a migration moves the robot
    restore = spec.restore and (complete or stale)
    if not restore:
        ops.clear_dir(ev.dst, ev.dst_bundle)

    probe = None
    for attempt in range(2):
        t_launch = now()
        ops.launch(restore)
        probe = ops.wait_probe()
        if probe is not None:
            break
        out["retry_count"] += 1
        ops.kill_dst()
    t_back = now()
    out["transfer_to_restore_ms"] = (t_back - t_launch) * 1000
    if probe is not None:
        out["robot_back"] = 1
        out["restored"] = int(probe.get("restored", 0))
        out["restore_error"] = str(probe.get("restore_error", ""))[:200]
        out["policy_load_ms"] = probe.get("policy_load_ms", 0)
        out["policy_action_mse"] = probe.get("policy_action_mse", -1)
        out["policy_weight_l2"] = probe.get("policy_weight_l2", -1)
        out["replay_buffer_entries_restored"] = probe.get("replay_entries_post", 0)
    else:
        out["restore_error"] = "relaunched worker never reported"
    if stale and out["restored"] == 1:
        out["restored_stale"] = 1
        if pre.get("round") not in (None, "") and ev.fl_round:
            out["stale_rounds"] = ev.fl_round - int(pre["round"])
        if ev.replay_entries_pre >= 0 and pre.get("replay_entries", -1) >= 0:
            out["stale_replay_entries"] = ev.replay_entries_pre - int(pre["replay_entries"])
    if spec.precopy:
        ops.clear_dir(ev.dst, ev.dst_bundle + ".new")
        ops.clear_dir(ev.dst, ev.dst_bundle + ".pre")
    out["downtime_ms"] = (t_back - t_saved) * 1000
    out["total_MTT_ms"] = (t_back - t0) * 1000
    out["total_recovery_ms"] = out["downtime_ms"]
    return out


class EventWriter:
    """task2 migration_events.csv columns + the Exp1 columns."""
    FIELDNAMES = [
        "condition", "robot_id", "migration_event_id", "timestamp", "fl_round",
        "src_node", "dst_node",
        "trigger_to_dump_ms", "dump_to_transfer_ms", "transfer_to_restore_ms",
        "policy_load_ms", "downtime_ms", "total_MTT_ms",
        "success_rate_pre", "success_rate_post", "regression_pct",
        "fl_rounds_to_recover", "replay_buffer_entries_restored",
        "network_bytes_transferred", "checkpoint_size_mb", "checkpoint_mode",
        "concurrency_level", "fault_injected", "retry_count", "total_recovery_ms",
        "policy_action_mse", "policy_weight_l2", "dht_put_ms", "dht_get_ms",
        # --- Exp1 ---
        "kill_point", "migration_index", "eval_return_pre", "restored",
        "restore_error", "robot_back", "fetch_complete", "served_by",
        "files_from_src", "files_from_replica", "fetch_attempts",
        "replicas_placed", "replicate_ms", "dht_found", "kill_at_ms",
        "kill_ms", "kill_ok",
        # --- app_warm: pre-copy and stale restores ---
        "restored_stale", "stale_rounds", "stale_replay_entries", "replay_entries_pre",
        "precopy_ok", "precopy_round", "precopy_ms", "background_bandwidth_mb",
    ]

    def __init__(self, condition: str, results_dir: str):
        self.condition = condition
        self.path = os.path.join(results_dir, "migration_events.csv")
        self._lock = threading.Lock()
        self._n = 0
        os.makedirs(results_dir, exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()

    def write_event(self, metrics: dict) -> None:
        with self._lock:
            self._n += 1
            row = {k: metrics.get(k, -1) for k in self.FIELDNAMES}
            for k in ("dump_to_transfer_ms", "trigger_to_dump_ms", "downtime_ms",
                      "total_MTT_ms", "transfer_to_restore_ms", "replicate_ms",
                      "dht_put_ms", "dht_get_ms", "kill_at_ms", "kill_ms",
                      "total_recovery_ms"):
                if isinstance(row[k], float):
                    row[k] = round(row[k], 3)
            row.update(condition=self.condition, migration_event_id=self._n,
                       timestamp=round(time.time(), 3))
            with open(self.path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDNAMES).writerow(row)

    @property
    def event_count(self) -> int:
        return self._n


# --------------------------------------------------------------------------- #
# Learning-progress columns, filled when the run ends (from task_logs rows).   #
# --------------------------------------------------------------------------- #
RECOVER_FRACTION = 0.9     # "recovered" = eval return back to >= 90% of pre
PRE_WINDOW = 3             # pre level = mean eval return over the 3 rounds before


def recovery_metrics(task_rows: list, robot_id: str, fl_round: int) -> dict:
    """Per event, from that robot's per-round eval rows (task_logs):
      success_rate_post    eval success in the first round >= the kill round
                           (the killed source never evaluates that round, so
                           this is the relaunched robot)
      regression_pct       100 * (first post eval return - pre) / pre
      fl_rounds_to_recover rounds after the kill round until eval return is back
                           to >= RECOVER_FRACTION * pre (0 = never dropped below)
    A value that cannot be computed (no rows, pre <= 0, never recovered within
    the run) is left empty rather than -1."""
    pts = []
    for r in task_rows:
        if r.get("robot_id") != robot_id:
            continue
        try:
            pts.append((int(float(r["fl_round"])), float(r["eval_return"]),
                        float(r.get("eval_success") or 0)))
        except (KeyError, TypeError, ValueError):
            continue
    pts.sort()
    pre = [e for rd, e, _ in pts if fl_round - PRE_WINDOW <= rd < fl_round]
    post = [p for p in pts if p[0] >= fl_round]
    out = {"success_rate_post": "", "regression_pct": "", "fl_rounds_to_recover": ""}
    if post:
        out["success_rate_post"] = post[0][2]
    if pre and post:
        p = sum(pre) / len(pre)
        if p > 0:
            out["regression_pct"] = round(100.0 * (post[0][1] - p) / p, 2)
            for rd, e, _ in post:
                if e >= RECOVER_FRACTION * p:
                    out["fl_rounds_to_recover"] = rd - fl_round
                    break
    return out


def fill_recovery(events_path: str, task_rows: list) -> int:
    """Rewrite migration_events.csv (atomically) with the recovery columns."""
    if not os.path.exists(events_path):
        return 0
    with open(events_path, newline="") as f:
        rd = csv.DictReader(f)
        fields, rows = rd.fieldnames, list(rd)
    for row in rows:
        try:
            fr = int(float(row.get("fl_round") or 0))
        except ValueError:
            continue
        row.update(recovery_metrics(task_rows, row.get("robot_id", ""), fr))
    tmp = events_path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, events_path)
    return len(rows)
