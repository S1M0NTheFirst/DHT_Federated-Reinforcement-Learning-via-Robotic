#!/usr/bin/env python3
"""Exp1 robot worker = task2's online-SAC worker (imported unchanged from
/cluster_app/task2/worker) plus two things a real relocation needs:

  - restore-on-start: a robot relaunched on the destination after its source was
    killed loads the bundle the runner fetched (RESTORE_DIR), or starts cold if
    none arrived, and reports what it got (probe_metrics:<robot>) to the runner;
  - orphan safety: exits if its launcher (apptainer / the job's SSH session)
    disappears, or after EXP1_MAX_LIFETIME seconds.

  - Exp1 client behaviour (see exp1_client): robot_id in metrics, and a
    migration source that never resumes.

Env (besides task2's): RESTORE_DIR, EXP1_RELAUNCH=1, EXP1_MAX_LIFETIME,
EXP1_SOURCE_GRACE, EXP1_PRECOPY_ROUNDS (app_warm only, else 0).
The --run-tag argument only makes this job's processes uniquely matchable
by the runner's pkill (a SIGKILL must never hit another job's robot)."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import socket
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


def bundle_files(d: str) -> dict:
    files = {}
    for name in ("sac_state.pt", "replay_buffer.pkl", "manifest.json"):
        try:
            files[name] = os.path.getsize(os.path.join(d, name))
        except OSError:
            pass
    return files


def exp1_client(osw, robot, r, condition: str, grace_s: float,
                precopy_rounds: int = 0):
    """task2's Flower client, adapted to Exp1:
      - fit/evaluate metrics carry robot_id, so the server can tell which proxy
        belongs to which robot (ghost pruning after a kill);
      - a migration source NEVER resumes: task2's _maybe_migrate waits 600 s for
        migration_done and then carries on training, which after a failed kill
        leaves two copies of the robot in FedAvg. In Exp1 the runner kills the
        source, so the source saves + requests, then waits to be killed and
        exits by itself after grace_s (loudly) if the kill never comes;
      - app_warm (precopy_rounds > 0): at the START of the fit round that is
        precopy_rounds before a migration round, before that round's training,
        write a pre-copy bundle to <chk_dir>.pre and tell the runner, which
        copies it to the destination in the background. The migration's final
        bundle is therefore precopy_rounds of training newer than it."""
    base = osw._make_flower_client(robot, r, condition, False)
    Base = type(base)

    class Exp1Client(Base):
        def _maybe_migrate(self, fl_round: int):
            if fl_round not in self.forced_rounds:
                return
            rid = self.robot.robot_id
            a_pre = self.robot.probe_actions()
            info = self.robot.save_bundle(self.chk_dir)
            # File sizes go with the request, so the runner needs no ssh stat on
            # the source (that ssh hung once for minutes).
            files = bundle_files(self.chk_dir)
            r.set(f"probe_pre:{rid}", pickle.dumps(a_pre).hex(), ex=3600)
            r.set(f"ready_for_criu:{rid}", "1", ex=3600)
            r.set(f"migration_request:{rid}", json.dumps({
                "robot_id": rid, "fl_round": fl_round,
                "success_rate": self.robot.last_eval_success,
                "eval_return_pre": self.robot.last_eval_return,
                "task_counter": self.robot.total_env_steps,
                "bundle_mb": info["bundle_mb"], "files": files,
                "replay_entries": info["replay_entries"],
            }), ex=3600)
            osw.logger.info(f"[{rid}] migration requested at fl_round={fl_round} "
                            f"(bundle={info['bundle_mb']:.2f}MB) - source waits "
                            f"to be killed")
            time.sleep(grace_s)
            osw.logger.error(f"[{rid}] source NOT killed within {grace_s:.0f}s - "
                             f"exiting so the robot never runs twice")
            r.set(f"exp1_source_selfexit:{rid}", str(time.time()), ex=86400)
            os._exit(4)

        def fit(self, params, config):
            rd = int(config.get("round", 0))
            if precopy_rounds > 0 and rd + precopy_rounds in self.forced_rounds:
                rid = self.robot.robot_id
                pre_dir = self.chk_dir + ".pre"
                info = self.robot.save_bundle(pre_dir)
                r.set(f"exp1_precopy:{rid}", json.dumps({
                    "robot_id": rid, "round": rd, "for_round": rd + precopy_rounds,
                    "files": bundle_files(pre_dir),
                    "replay_entries": info["replay_entries"], "t": time.time(),
                }), ex=3600)
                osw.logger.info(f"[{rid}] pre-copy bundle written at round {rd} for "
                                f"the migration at round {rd + precopy_rounds}")
            arrays, n, metrics = super().fit(params, config)
            return arrays, n, {**metrics, "robot_id": self.robot.robot_id}

        def evaluate(self, params, config):
            loss, n, metrics = super().evaluate(params, config)
            return loss, n, {**metrics, "robot_id": self.robot.robot_id}

    return Exp1Client()


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
    # Host + PID (Apptainer shares the host PID namespace) let the runner kill
    # exactly this process and verify it is gone.
    r.set(f"exp1_pid:{robot.robot_id}", json.dumps({
        "host": socket.gethostname().split(".")[0], "pid": os.getpid(),
        "tag": args.run_tag, "t": time.time()}), ex=86400)
    info = restore_on_start(robot, r, os.getenv("RESTORE_DIR", ""),
                            os.getenv("EXP1_RELAUNCH", "0") == "1",
                            PolicyProbe.action_mse)
    osw.logger.info(f"[{robot.robot_id}] exp1 worker start tag={args.run_tag} "
                    f"pid={os.getpid()} relaunch_info={info}")

    # The cold_restart flag of the task2 client is irrelevant: in Exp1 the
    # source never resumes after a migration request.
    client = exp1_client(osw, robot, r, condition,
                         float(os.environ.get("EXP1_SOURCE_GRACE", "900")),
                         int(os.environ.get("EXP1_PRECOPY_ROUNDS", "0")))
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
