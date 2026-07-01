"""
Pre-train a shared BidPolicyMLP so every condition starts from the same
competent policy instead of random weights.

Why: the recovery-curve and regression metrics are far less noisy when each
condition starts from a policy that already knows roughly when to accept vs.
decline. A random start spends the first few hundred tasks flailing, which
swamps the post-migration signal we actually care about.

The output checkpoint is in RobotPPOAgent.save_checkpoint() format
(policy_state_dict / optimizer_state_dict / training_step / robot_id) so it
can be loaded by:
  - the cluster copy of the DHT worker (Condition A), and
  - worker_app_checkpoint.py (Conditions C/D/E/F/G)

This trains OFFLINE against a synthetic load model — no Redis, no cluster, no
GPU required. It is deterministic given --seed so the committed policy is
reproducible. The file itself is gitignored (regenerate with this script).

Run (locally or on the cluster head node):
    python3 cluster/common/pretrain_ppo.py
    python3 cluster/common/pretrain_ppo.py --steps 8000 --out /path/to/policy.pt
"""
import argparse
import os
import sys

import numpy as np
import torch


def _locate_robot_module() -> str:
    """Return the dir containing policy.py / sensor.py, searching the usual
    cluster + local layouts. swiftbot_rl/ stays frozen — we only import it."""
    candidates = []
    env_root = os.environ.get("SWIFTBOT_RL_ROOT")
    if env_root:
        candidates.append(os.path.join(env_root, "dht_frl", "robot"))
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))  # cluster/common -> repo
    candidates.append(os.path.join(repo_root, "swiftbot_rl", "dht_frl", "robot"))
    # Inside the apptainer container the robot modules are bound at /robot_lib.
    candidates.append("/robot_lib")
    for c in candidates:
        if os.path.exists(os.path.join(c, "policy.py")):
            return c
    raise FileNotFoundError(
        "Could not find policy.py in any of: " + ", ".join(candidates))


ROBOT_DIR = _locate_robot_module()
sys.path.insert(0, ROBOT_DIR)
from policy import RobotPPOAgent  # noqa: E402

STATE_DIM = 15


def _synthetic_state(load: float, complexity: float, rng: np.random.Generator
                     ) -> np.ndarray:
    """Build a 15-dim sensor-shaped state for a given system load.

    Mirrors the dimension layout in swiftbot_rl sensor.py. We don't need every
    dimension to be physically accurate — only that 'load' and 'complexity'
    carry the signal the policy must learn (accept when slack, decline when
    saturated)."""
    s = rng.random(STATE_DIM).astype(np.float32) * 0.1
    s[0] = load                          # cpu_util
    s[2] = load                          # gpu_util
    s[3] = load * 0.8                    # gpu_mem_util
    s[4] = min(load * 1.2, 1.0)          # active_tasks_norm
    s[8] = complexity                    # task_complexity
    s[9] = 0.5                           # deadline_norm
    return s


PRESSURE_THRESHOLD = 0.65


def _reward_for_decision(bid: float, load: float, complexity: float,
                         threshold: float = 0.5) -> float:
    """Reward model for offline pre-training.

    A state-aware policy (accept when there's slack, decline when saturated)
    must be the optimum, otherwise PPO collapses to a state-blind 'always
    accept'. We give accepting under high pressure a large penalty (it both
    times out and, in the real system, spills contention onto neighbours) and
    declining only a small penalty. With these magnitudes:
        always-accept   E[r] ~= +0.03
        state-aware     E[r] ~= +0.50   <- clear global optimum
        always-decline  E[r]  = -0.20
    """
    accept = bid >= threshold
    pressure = 0.6 * load + 0.4 * complexity
    if accept:
        return +1.5 if pressure < PRESSURE_THRESHOLD else -1.0
    else:
        return -0.2


def pretrain(steps: int, seed: int) -> RobotPPOAgent:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    agent = RobotPPOAgent(state_dim=STATE_DIM, robot_id="pretrained")

    for _ in range(steps):
        load = float(rng.random())          # 0..1 system load
        complexity = float(rng.uniform(0.5, 1.0))
        state = _synthetic_state(load, complexity, rng)
        bid = agent.get_bid(state)
        reward = _reward_for_decision(bid, load, complexity)
        next_state = _synthetic_state(float(rng.random()), complexity, rng)
        agent.record_reward(reward, next_state)

    return agent


def _evaluate(agent: RobotPPOAgent, trials: int = 2000, seed: int = 999) -> float:
    """Fraction of CORRECT decisions: accept when there's slack
    (pressure < threshold), decline when saturated. Random ~0.50; a competent
    state-aware policy should be well above 0.65."""
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(trials):
        load = float(rng.random())
        complexity = float(rng.uniform(0.5, 1.0))
        state = _synthetic_state(load, complexity, rng)
        bid = agent.policy.get_bid(state)
        accept = bid >= 0.5
        should_accept = (0.6 * load + 0.4 * complexity) < PRESSURE_THRESHOLD
        if accept == should_accept:
            correct += 1
    return correct / trials


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000,
                    help="number of synthetic task decisions to train on")
    ap.add_argument("--seed", type=int, default=20260527)
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "pretrained_policy.pt")
    ap.add_argument("--out", type=str, default=default_out)
    args = ap.parse_args()

    print(f"Robot modules: {ROBOT_DIR}")
    print(f"Pre-training BidPolicyMLP for {args.steps} steps (seed={args.seed})")
    agent = pretrain(args.steps, args.seed)
    score = _evaluate(agent)
    print(f"Post-training correct-decision rate: {score:.3f} "
          f"(random ~0.50; a competent policy should be >0.65)")

    agent.save_checkpoint(args.out)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"Saved pretrained policy -> {args.out} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
