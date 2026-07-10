"""
Policy-equivalence probe (behavioral losslessness) — PLAN Logging #5.

Direct numerical proof that the RESTORED policy is the SAME policy, not just
that reward looks continuous. A fixed batch of observations is sampled ONCE at
startup, identical across all robots/conditions (seeded). Immediately BEFORE
migration we record the pre-migration deterministic actions (a_pre); immediately
AFTER resume we recompute on the same batch (a_post).

Expected: policy_action_mse ≈ 0 for all state-preserving conditions
(dht_frl/app_cold/app_warm/tcp_scp/dmtcp) → bit-for-bit lossless; LARGE for
cold_restart. Cheap: a few CPU forward passes on a small batch.
"""
from __future__ import annotations

import numpy as np
import torch

from sac import DEVICE

PROBE_SEED = 20260710      # identical across ALL robots/conditions
PROBE_SIZE = 256


class PolicyProbe:
    def __init__(self, obs_dim: int, size: int = PROBE_SIZE,
                 seed: int = PROBE_SEED):
        rng = np.random.default_rng(seed)
        # Hopper observations are roughly unit-scaled; a standard-normal probe
        # batch spans the region the policy actually sees. The point is only
        # that the batch is FIXED and shared, not that it matches the exact
        # state distribution.
        self.obs = torch.as_tensor(
            rng.standard_normal((size, obs_dim)).astype(np.float32),
            device=DEVICE,
        )

    @torch.no_grad()
    def actions(self, actor) -> np.ndarray:
        """Deterministic actions on the fixed batch (exploration off)."""
        return actor.act_deterministic(self.obs).cpu().numpy()

    @staticmethod
    def action_mse(a_pre: np.ndarray, a_post: np.ndarray) -> float:
        return float(np.mean((a_post - a_pre) ** 2))

    @staticmethod
    def weight_l2(sac_pre_actor_arrays, sac_post_actor_arrays) -> float:
        total = 0.0
        for wp, wq in zip(sac_pre_actor_arrays, sac_post_actor_arrays):
            total += float(np.sum((wq - wp) ** 2))
        return float(np.sqrt(total))
