"""
BidPolicyMLP — PPO policy for robot task bidding.
State: 15-dim vector (see sensor.py)
Action: bid confidence [0.0, 1.0]
Trained with simplified PPO via replay buffer.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import pickle
import os


class BidPolicyMLP(nn.Module):
    """Small MLP. Fast inference < 1ms. Fits in < 1MB for policy transfer."""

    def __init__(self, state_dim: int = 15, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, 1),        nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_bid(self, state: np.ndarray) -> float:
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0)
            return float(self.net(t).squeeze())

    def get_entropy(self, state: np.ndarray) -> float:
        """Policy entropy — lower = more confident. Used as a training signal."""
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0)
            p = float(self.net(t).squeeze())
            p = max(1e-6, min(1.0 - 1e-6, p))
            return -(p * np.log(p) + (1 - p) * np.log(1 - p))


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            float(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
        ))

    def sample(self, batch_size: int = 64):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        s, a, r, ns = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(ns)

    def tail(self, n: int = 1000) -> list:
        return list(self.buffer)[-n:]

    def load_tail(self, entries: list):
        for e in entries:
            self.buffer.append(e)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(list(self.buffer), f)

    def load(self, path: str):
        with open(path, "rb") as f:
            entries = pickle.load(f)
        self.buffer = deque(entries, maxlen=self.buffer.maxlen)

    def __len__(self):
        return len(self.buffer)


class RobotPPOAgent:
    """Wraps BidPolicyMLP with PPO training loop."""

    def __init__(self, state_dim: int = 15, lr: float = 3e-4,
                 robot_id: str = "robot_000"):
        self.robot_id = robot_id
        self.policy = BidPolicyMLP(state_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.training_step = 0
        self._last_state = None
        self._last_action = None

    def get_bid(self, state: np.ndarray) -> float:
        bid = self.policy.get_bid(state)
        self._last_state = state.copy()
        self._last_action = bid
        return bid

    def record_reward(self, reward: float, next_state: np.ndarray):
        if self._last_state is not None:
            self.replay_buffer.add(
                self._last_state, self._last_action, reward, next_state
            )
        if len(self.replay_buffer) >= 64 and len(self.replay_buffer) % 32 == 0:
            self._ppo_update()

    def _ppo_update(self, batch_size: int = 64, epochs: int = 4):
        states, actions, rewards, _ = self.replay_buffer.sample(batch_size)
        s_t = torch.FloatTensor(states)
        r_t = torch.FloatTensor(rewards)
        if r_t.std() > 1e-8:
            r_t = (r_t - r_t.mean()) / (r_t.std() + 1e-8)
        for _ in range(epochs):
            bids = self.policy(s_t).squeeze()
            loss = -torch.mean(r_t * bids)
            entropy = -torch.mean(
                bids * torch.log(bids + 1e-8) +
                (1 - bids) * torch.log(1 - bids + 1e-8)
            )
            total_loss = loss - 0.01 * entropy
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        self.training_step += 1

    def get_entropy(self) -> float:
        if len(self.replay_buffer) < 10:
            return 1.0
        states, *_ = self.replay_buffer.sample(32)
        return float(np.mean([
            self.policy.get_entropy(s) for s in states
        ]))

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step": self.training_step,
            "robot_id": self.robot_id,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.training_step = ckpt.get("training_step", 0)

    def get_weights(self) -> dict:
        return {k: v.clone() for k, v in self.policy.state_dict().items()}

    def set_weights(self, weights: dict):
        self.policy.load_state_dict(weights)
