"""
Shared ONLINE-SAC Flower worker — used by ALL 6 task2 conditions.

Do NOT fork this per condition. Behavior differences are gated by env vars:
  CONDITION           — dht_frl | app_cold | app_warm | tcp_scp | dmtcp | cold_restart
  COLD_RESTART=1      — on resume, DISCARD the local bundle (empty replay +
                        reset optimizers); keep only the global actor weights.
                        This is the negative control that must dip (Issue #2).
Everything else (env stepping, SAC updates, eval rollout, bundle write, probe,
logging) is identical across conditions so the outputs are directly comparable.
Only the checkpoint/transport tool differs, and that lives in the per-condition
runner's trigger_fn — NOT here.

Each robot:
  (a) steps its OWN gymnasium.make(TASK2_ENV) (default Hopper-v4, MuJoCo),
      collecting transitions into a LOCAL replay buffer, running SAC updates on
      CPU (device forced to cpu in sac.py),
  (b) federates ONLY the actor weights each FL round (FedAvg),
  (c) runs a periodic eval rollout with exploration OFF and logs mean episode
      return + success,
  (d) writes the migration bundle (full SAC local state + replay buffer) to
      /checkpoints/<robot>/,
  (e) logs per-round eval return + success to Redis `task_logs` (server persists
      task_logs.csv).

CPU forced. Real self-generated MuJoCo data. No Minari, no synthetic fallback.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import signal
import sys
import time
from collections import OrderedDict

import numpy as np
import torch

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

# worker/ is on sys.path so `import sac` etc. resolve both in-container
# (/cluster_app/task2/worker bound + PYTHONPATH) and in a local smoke test.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sac import SAC, OBS_DIM, ACT_DIM, DEVICE          # noqa: E402
from replay_buffer import ReplayBuffer                 # noqa: E402
from probe import PolicyProbe                          # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(message)s")
logger = logging.getLogger("online_sac")

# ---- knobs (env-tunable so the cluster run can be calibrated w/o code edits) --
ENV_NAME          = os.environ.get("TASK2_ENV", "Hopper-v4")
STEPS_PER_ROUND   = int(os.environ.get("STEPS_PER_ROUND", "1000"))
MIN_BUFFER_FILL   = int(os.environ.get("MIN_BUFFER_FILL", "1000"))
UPDATES_PER_STEP  = int(os.environ.get("UPDATES_PER_STEP", "1"))
BATCH_SIZE        = int(os.environ.get("SAC_BATCH", "256"))
BUFFER_CAPACITY   = int(os.environ.get("BUFFER_CAPACITY", "100000"))
EVAL_EPISODES     = int(os.environ.get("EVAL_EPISODES", "3"))
EVAL_SUCCESS_RETURN = float(os.environ.get("EVAL_SUCCESS_RETURN", "250"))
TOTAL_FL_ROUNDS   = int(os.environ.get("TOTAL_FL_ROUNDS", "150"))
# Shared start: all robots/conditions seed identical initial actor weights.
SHARED_SEED       = int(os.environ.get("SHARED_SEED", "12345"))


class OnlineSACRobot:
    """Env-driven online SAC for one robot. No Flower/redis here so the smoke
    test can drive it directly (step, eval, wipe)."""

    def __init__(self, client_id: int, *, env_name: str = ENV_NAME,
                 capacity: int = BUFFER_CAPACITY, seed: int | None = None):
        import gymnasium as gym
        self.client_id = client_id
        self.robot_id = f"robot_{client_id:03d}"
        # Per-robot env seed → naturally heterogeneous rollouts across the fleet.
        self.env_seed = (client_id * 1000 + 7) if seed is None else seed
        self.env = gym.make(env_name)
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        self.obs_dim, self.act_dim = obs_dim, act_dim

        # Shared initial ACTOR weights across the whole fleet + all conditions:
        # seed torch identically before constructing the SAC so every robot's
        # actor starts from the same point (PLAN: shared start for fair
        # comparison). Critics differ only by this same seed → also identical.
        torch.manual_seed(SHARED_SEED)
        np.random.seed(SHARED_SEED)
        self.sac = SAC(obs_dim, act_dim)

        self.replay = ReplayBuffer(obs_dim, act_dim, capacity=capacity,
                                   seed=self.env_seed)
        self.probe = PolicyProbe(obs_dim)

        self._obs, _ = self.env.reset(seed=self.env_seed)
        self.global_step = 0
        self.total_env_steps = 0
        # Most recent eval, so a migration mid-fit can report the real
        # pre-migration success/return without a fresh (expensive) rollout.
        self.last_eval_return = 0.0
        self.last_eval_success = 0.0

    # ---- online interaction ----
    def collect_and_train(self, n_steps: int) -> dict:
        """Step the live env n_steps, storing transitions and running SAC
        updates once the buffer passes MIN_BUFFER_FILL. Returns round stats."""
        ep_return = 0.0
        returns = []
        losses = []
        for _ in range(n_steps):
            if len(self.replay) < MIN_BUFFER_FILL:
                # Warm-up: random actions to fill the buffer (standard SAC).
                a = self.env.action_space.sample()
            else:
                a = self.sac.select_action(self._obs, deterministic=False)
            no, r, term, trunc, _ = self.env.step(a)
            done = term  # bootstrap on truncation (time-limit) — don't treat as terminal
            self.replay.add(self._obs, a, r, no, done)
            self._obs = no
            ep_return += r
            self.global_step += 1
            self.total_env_steps += 1

            # Gate SAC updates on min buffer fill so a WIPED buffer causes a real
            # setback (Issue #2 continuity contrast depends on this).
            if len(self.replay) >= MIN_BUFFER_FILL:
                for _ in range(UPDATES_PER_STEP):
                    stats = self.sac.update(self.replay.sample(BATCH_SIZE))
                losses.append(stats["critic_loss"])

            if term or trunc:
                returns.append(ep_return)
                ep_return = 0.0
                self._obs, _ = self.env.reset()
        return {
            "train_return_mean": float(np.mean(returns)) if returns else 0.0,
            "critic_loss": float(np.mean(losses)) if losses else 0.0,
            "alpha": self.sac.alpha,
            "buffer": len(self.replay),
        }

    @torch.no_grad()
    def eval_return(self, n_episodes: int = EVAL_EPISODES) -> dict:
        """Real eval return: run the current policy with exploration OFF."""
        import gymnasium as gym
        eval_env = gym.make(ENV_NAME)
        rets, lens = [], []
        for ep in range(n_episodes):
            o, _ = eval_env.reset(seed=self.env_seed + 10_000 + ep)
            done = False
            ret, steps = 0.0, 0
            while not done:
                a = self.sac.select_action(o, deterministic=True)
                o, r, term, trunc, _ = eval_env.step(a)
                ret += r
                steps += 1
                done = term or trunc
            rets.append(ret)
            lens.append(steps)
        eval_env.close()
        mean_ret = float(np.mean(rets))
        success = float(np.mean([1.0 if x >= EVAL_SUCCESS_RETURN else 0.0
                                 for x in rets]))
        self.last_eval_return = mean_ret
        self.last_eval_success = success
        return {"eval_return": mean_ret,
                "eval_episode_len": float(np.mean(lens)),
                "eval_success": success}

    # ---- probe ----
    def probe_actions(self) -> np.ndarray:
        return self.probe.actions(self.sac.actor)

    # ---- bundle (policy + optimizer + local replay) ----
    def save_bundle(self, chk_dir: str) -> dict:
        os.makedirs(chk_dir, exist_ok=True)
        sac_path = os.path.join(chk_dir, "sac_state.pt")
        rb_path = os.path.join(chk_dir, "replay_buffer.pkl")
        manifest_path = os.path.join(chk_dir, "manifest.json")

        tmp = sac_path + ".tmp"
        torch.save(self.sac.state_dict(), tmp)
        os.replace(tmp, sac_path)
        with open(rb_path + ".tmp", "wb") as f:
            pickle.dump(self.replay.state_dict(), f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(rb_path + ".tmp", rb_path)

        sizes = {p: (os.path.getsize(p) if os.path.exists(p) else 0)
                 for p in (sac_path, rb_path)}
        total_mb = sum(sizes.values()) / (1024 * 1024)
        with open(manifest_path, "w") as f:
            json.dump({"robot_id": self.robot_id,
                       "train_step": self.sac.train_step,
                       "replay_entries": len(self.replay),
                       "bundle_mb": round(total_mb, 3)}, f)
        return {"bundle_mb": total_mb, "replay_entries": len(self.replay)}

    def load_bundle(self, chk_dir: str) -> dict:
        sac_path = os.path.join(chk_dir, "sac_state.pt")
        rb_path = os.path.join(chk_dir, "replay_buffer.pkl")
        self.sac.load_state_dict(torch.load(sac_path, map_location=DEVICE,
                                            weights_only=False))
        with open(rb_path, "rb") as f:
            self.replay.load_state_dict(pickle.load(f))
        return {"replay_entries": len(self.replay)}

    def wipe_local_state(self) -> None:
        """cold_restart resume (and the smoke test's simulated cold restart):
        empty replay buffer + reset optimizers/temperature, keep only the global
        actor weights the robot already holds. Critics are re-initialized so
        Q-values must be relearned — a real setback."""
        actor_arrays = self.sac.get_actor_arrays()   # keep global weights
        fresh = SAC(self.obs_dim, self.act_dim)
        fresh.set_actor_arrays(actor_arrays)
        self.sac = fresh
        self.replay.clear()

    # ---- actor exchange ----
    def get_actor_arrays(self):
        return self.sac.get_actor_arrays()

    def set_actor_arrays(self, arrays):
        self.sac.set_actor_arrays(arrays)


# =============================================================================
# Flower client + redis migration protocol (cluster path).
# =============================================================================
_MIGRATION_SCHEDULE_ROUNDS = [
    int(x) for x in os.environ.get(
        "MIGRATION_ROUNDS", "30,60,90,120,140").split(",") if x.strip()
]
_MIGRATION_OFFSET = int(os.environ.get("MIGRATION_OFFSET", "0"))


def _forced_rounds_for(client_id: int) -> set:
    # Stagger per robot so migrations don't all fire on the same FL round; keep
    # the schedule identical across conditions so markers line up on the figure.
    off = (client_id % 5)  # small spread; keep within round budget
    return {min(rd + off, TOTAL_FL_ROUNDS - 1)
            for rd in _MIGRATION_SCHEDULE_ROUNDS}


shutdown_requested = False


def _on_signal(s, f):
    global shutdown_requested
    shutdown_requested = True


signal.signal(signal.SIGTERM, _on_signal)
signal.signal(signal.SIGINT, _on_signal)


def _make_flower_client(robot: OnlineSACRobot, r, condition: str,
                        cold_restart: bool):
    import flwr as fl

    class SACClient(fl.client.NumPyClient):
        def __init__(self):
            self.robot = robot
            self.forced_rounds = _forced_rounds_for(robot.client_id)
            self.chk_dir = os.path.join("/checkpoints", robot.robot_id)

        def get_parameters(self, config):
            return self.robot.get_actor_arrays()

        def _maybe_migrate(self, fl_round: int):
            if fl_round not in self.forced_rounds:
                return
            rid = self.robot.robot_id
            # 1. record pre-migration probe actions + write bundle.
            a_pre = self.robot.probe_actions()
            actor_pre = self.robot.get_actor_arrays()
            info = self.robot.save_bundle(self.chk_dir)
            r.set(f"probe_pre:{rid}", pickle.dumps(a_pre).hex(), ex=3600)
            r.set(f"ready_for_criu:{rid}", "1", ex=3600)
            r.set(f"migration_request:{rid}", json.dumps({
                "robot_id": rid, "fl_round": fl_round,
                "success_rate": self.robot.last_eval_success,
                "eval_return_pre": self.robot.last_eval_return,
                "task_counter": self.robot.total_env_steps,
                "bundle_mb": info["bundle_mb"],
            }), ex=3600)
            logger.info(f"[{rid}] migration requested at fl_round={fl_round} "
                        f"(bundle={info['bundle_mb']:.2f}MB) — waiting for runner")

            # 2. wait for runner to complete transport.
            deadline = time.time() + 600
            while time.time() < deadline and not shutdown_requested:
                if r.get(f"migration_done:{rid}"):
                    r.delete(f"migration_done:{rid}")
                    break
                time.sleep(0.5)

            # 3. resume. State-preserving conditions reload the bundle the runner
            #    placed at load_policy; cold_restart discards it.
            t_load = time.perf_counter()
            if cold_restart:
                self.robot.wipe_local_state()
                logger.info(f"[{rid}] COLD RESTART — wiped local replay+optimizer")
            else:
                load_dir = r.get(f"load_policy:{rid}") or self.chk_dir
                try:
                    self.robot.load_bundle(load_dir)
                    logger.info(f"[{rid}] bundle restored from {load_dir}")
                except Exception as e:
                    logger.error(f"[{rid}] BUNDLE LOAD FAILED from {load_dir}: {e!r}")
                r.delete(f"load_policy:{rid}")
            load_ms = (time.perf_counter() - t_load) * 1000

            # 4. post-migration probe + losslessness numbers → runner reads these.
            a_post = self.robot.probe_actions()
            actor_post = self.robot.get_actor_arrays()
            mse = PolicyProbe.action_mse(a_pre, a_post)
            wl2 = PolicyProbe.weight_l2(actor_pre, actor_post)
            r.set(f"probe_metrics:{rid}", json.dumps({
                "policy_action_mse": mse,
                "policy_weight_l2": wl2,
                "policy_load_ms": round(load_ms, 2),
                "replay_entries_post": len(self.robot.replay),
            }), ex=3600)
            r.set(f"first_bid_after_migration:{rid}",
                  json.dumps({"policy_load_ms": round(load_ms, 2)}), ex=120)
            logger.info(f"[{rid}] resumed: action_mse={mse:.3e} weight_l2={wl2:.3e} "
                        f"replay={len(self.robot.replay)}")

        def fit(self, params, config):
            self.robot.set_actor_arrays(params)
            fl_round = int(config.get("round", 0))
            self._maybe_migrate(fl_round)
            stats = self.robot.collect_and_train(STEPS_PER_ROUND)
            return (self.robot.get_actor_arrays(), STEPS_PER_ROUND, {
                "train_loss": stats["critic_loss"],
                "mean_reward": stats["train_return_mean"],
                "policy_entropy": stats["alpha"],
                "train_time": 0.0,
                "cpu_usage": _cpu(),
                "gpu_usage": 0.0,
                "network_mb": 0.0,
            })

        def evaluate(self, params, config):
            self.robot.set_actor_arrays(params)
            fl_round = int(config.get("round", 0))
            ev = self.robot.eval_return()
            # Push the per-round time series row (server persists task_logs.csv).
            r.lpush("task_logs", json.dumps({
                "robot_id": self.robot.robot_id,
                "fl_round": fl_round,
                "training_step": self.robot.sac.train_step,
                "reward": round(ev["eval_return"], 3),
                "success_rate_rolling10": ev["eval_success"],
                "policy_entropy": round(self.robot.sac.alpha, 4),
                "status": "eval",
                "eval_return": round(ev["eval_return"], 3),
                "eval_episode_len": round(ev["eval_episode_len"], 1),
                "eval_success": ev["eval_success"],
            }))
            r.ltrim("task_logs", 0, 199999)
            return (float(max(0.0, -ev["eval_return"])),
                    EVAL_EPISODES, {
                        "mean_reward": ev["eval_return"],
                        "success_rate": ev["eval_success"],
                        "eval_return": ev["eval_return"],
                        "cpu_usage": _cpu(),
                        "gpu_usage": 0.0,
                    })

    return SACClient()


def _cpu() -> float:
    try:
        import psutil
        return float(psutil.cpu_percent())
    except Exception:
        return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--client-id", type=int, required=True)
    p.add_argument("--num-clients", type=int, default=20)
    p.add_argument("--container-type", type=str, default="cpu_specialist")
    args = p.parse_args()

    import redis
    import flwr as fl

    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    server = os.getenv("MASTER_ADDRESS", "127.0.0.1:8470")
    condition = os.getenv("CONDITION", "dht_frl")
    cold_restart = os.getenv("COLD_RESTART", "0") == "1"

    r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    robot = OnlineSACRobot(args.client_id)
    logger.info(f"[{robot.robot_id}] online SAC start — env={ENV_NAME} "
                f"condition={condition} cold_restart={cold_restart} "
                f"server={server} device={DEVICE}")

    client = _make_flower_client(robot, r, condition, cold_restart)

    retry, MAX_RETRIES = 0, 10
    while retry < MAX_RETRIES and not shutdown_requested:
        try:
            fl.client.start_numpy_client(server_address=server, client=client)
            logger.info(f"[{robot.robot_id}] all FL rounds complete")
            break
        except Exception as e:
            retry += 1
            logger.warning(f"[{robot.robot_id}] connect failed ({e}); "
                           f"retry {retry}/{MAX_RETRIES}")
            time.sleep(5)

    r.set(f"robot_done:{robot.robot_id}", "1")
    logger.info(f"[{robot.robot_id}] shutdown complete")


if __name__ == "__main__":
    main()
