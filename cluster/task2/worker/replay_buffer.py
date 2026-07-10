"""
Local per-robot replay buffer — the robot's OWN online experience, collected
from its OWN live MuJoCo rollouts. This is the state migration must preserve:
unlike task1's static Minari dataset (a reloadable public file), this buffer is
hard-won and genuinely valuable. cold_restart discards it → the SAC updates
starve until it refills → the eval-return curve dips and re-climbs (Issue #2).

Rides in the migration bundle. Sized meaningfully (default 100k) and paired with
a min-fill gate in the worker so a wiped buffer causes a REAL, visible learning
setback rather than an instant recovery.
"""
from __future__ import annotations

import numpy as np
import torch

from sac import DEVICE  # CPU


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 100_000,
                 seed: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, o, a, r, no, d) -> None:
        i = self.ptr
        self.obs[i] = o
        self.act[i] = a
        self.rew[i] = r
        self.next_obs[i] = no
        self.done[i] = float(d)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], device=DEVICE),
            "act": torch.as_tensor(self.act[idx], device=DEVICE),
            "rew": torch.as_tensor(self.rew[idx], device=DEVICE),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=DEVICE),
            "done": torch.as_tensor(self.done[idx], device=DEVICE),
        }

    # ---- bundle (de)serialization ----
    def state_dict(self) -> dict:
        """Serialize only the filled prefix so the bundle stays as small as the
        real experience (a cold buffer ships nothing)."""
        n = self.size
        return {
            "capacity": self.capacity,
            "ptr": self.ptr,
            "size": self.size,
            "obs": self.obs[:n].copy(),
            "act": self.act[:n].copy(),
            "rew": self.rew[:n].copy(),
            "next_obs": self.next_obs[:n].copy(),
            "done": self.done[:n].copy(),
            "rng": self.rng.bit_generator.state,
        }

    def load_state_dict(self, sd: dict) -> None:
        n = int(sd["size"])
        self.obs[:n] = sd["obs"]
        self.act[:n] = sd["act"]
        self.rew[:n] = sd["rew"]
        self.next_obs[:n] = sd["next_obs"]
        self.done[:n] = sd["done"]
        self.ptr = int(sd["ptr"])
        self.size = n
        try:
            self.rng.bit_generator.state = sd["rng"]
        except Exception:
            pass

    def clear(self) -> None:
        """Wipe local experience — cold_restart on resume, or the smoke-test's
        simulated cold restart."""
        self.ptr = 0
        self.size = 0
