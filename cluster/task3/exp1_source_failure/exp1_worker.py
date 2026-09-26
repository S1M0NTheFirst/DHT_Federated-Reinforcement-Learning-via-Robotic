#!/usr/bin/env python3
"""Exp1 robot worker = task2's online-SAC worker (imported unchanged from
/cluster_app/task2/worker) plus two things a real relocation needs:

  - restore-on-start: a robot relaunched on the destination after its source was
    killed loads the bundle the runner fetched (RESTORE_DIR), or starts cold if
    none arrived, and reports what it got (probe_metrics:<robot>) to the runner;
  - orphan safety: exits if its launcher (apptainer / the job's SSH session)
    disappears, or after EXP1_MAX_LIFETIME seconds.

Env (besides task2's): RESTORE_DIR, EXP1_RELAUNCH=1, EXP1_MAX_LIFETIME.
The --run-tag argument only makes this job's processes uniquely matchable
by the runner's pkill (a SIGKILL must never hit another job's robot)."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import threading
import time

sys.path.insert(0, os.environ.get("TASK2_WORKER_DIR", "/cluster_app/task2/worker"))


def orphan_guard(max_lifetime: float) -> None:
    parent, t_end = os.getppid(), time.time() + max_lifetime

    def watch() -> None:
        while os.getppid() == parent and time.time() < t_end:
            time.sleep(2.0)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def restore_on_start(robot, r, restore_dir: str, relaunch: bool,
                     action_mse=None) -> dict | None:
    """Load the fetched bundle (if any) and publish probe metrics for the
    runner. Never raises: a bad bundle means a cold start, not a dead robot."""
    if not relaunch:
        return None
    rid = robot.robot_id
    t = time.perf_counter()
    restored, err = 0, ""
    if restore_dir:
        try:
            robot.load_bundle(restore_dir)
            restored = 1
        except Exception as e:                           # noqa: BLE001
            err = repr(e)[:200]
    load_ms = (time.perf_counter() - t) * 1000
    mse = -1.0
    raw = r.get(f"probe_pre:{rid}")
    if raw and action_mse is not None:
        try:
            a_pre = pickle.loads(bytes.fromhex(raw))
            mse = float(action_mse(a_pre, robot.probe_actions()))
        except Exception:                                # noqa: BLE001
            pass
    m = {"policy_action_mse": mse, "policy_weight_l2": -1,
         "policy_load_ms": round(load_ms, 2),
         "replay_entries_post": len(robot.replay),
         "restored": restored, "restore_error": err}
    r.set(f"probe_metrics:{rid}", json.dumps(m), ex=3600)
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--client-id", type=int, required=True)
    p.add_argument("--num-clients", type=int, default=20)
    p.add_argument("--container-type", type=str, default="cpu_specialist")
    p.add_argument("--run-tag", type=str, default="exp1")
    args = p.parse_args()
    orphan_guard(float(os.environ.get("EXP1_MAX_LIFETIME", "21600")))

    import redis
    import flwr as fl
    import online_sac_worker as osw               # task2 worker, unchanged
    from probe import PolicyProbe

    r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", "6379")),
                    decode_responses=True)
    server = os.getenv("MASTER_ADDRESS", "127.0.0.1:8470")
    condition = os.getenv("CONDITION", "exp1")
    robot = osw.OnlineSACRobot(args.client_id)
    info = restore_on_start(robot, r, os.getenv("RESTORE_DIR", ""),
                            os.getenv("EXP1_RELAUNCH", "0") == "1",
                            PolicyProbe.action_mse)
    osw.logger.info(f"[{robot.robot_id}] exp1 worker start tag={args.run_tag} "
                    f"relaunch_info={info}")

    # In Exp1 the source never resumes after a migration request (the runner
    # kills it), so the cold_restart flag of the task2 client is irrelevant.
    client = osw._make_flower_client(robot, r, condition, False)
    retry = 0
    while retry < 10 and not osw.shutdown_requested:
        try:
            fl.client.start_numpy_client(server_address=server, client=client)
            osw.logger.info(f"[{robot.robot_id}] all FL rounds complete")
            break
        except Exception as e:                           # noqa: BLE001
            retry += 1
            osw.logger.warning(f"[{robot.robot_id}] connect failed ({e}); "
                               f"retry {retry}/10")
            time.sleep(5)
    r.set(f"robot_done:{robot.robot_id}", "1")


if __name__ == "__main__":
    main()
