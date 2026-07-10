"""
Proper online SAC — tanh-squashed bounded actor + twin critics + automatic
entropy tuning. CPU-forced (task2 runs entirely on CPU; DMTCP can't dump a live
CUDA process, and Hopper + tiny SAC is light on CPU).

Network SHAPES are the same as swiftbot_rl/motivation/hopper_agent.py
(OBS_DIM=11, ACT_DIM=3, HIDDEN=256) so the offline motivation agent and this
online agent are directly comparable — but the crude offline update there
(`loss_a = -critic1(o,a_pi).mean()` with unbounded actions) is REPLACED here by
a real SAC actor: reparameterized tanh-squashed sampling + entropy term. That
offline update does not learn online; this one does.

What gets FEDERATED (FedAvg over the fleet) is the ACTOR only — that's the
"global policy". Critics, the actor/critic optimizers, the entropy temperature,
and the replay buffer are LOCAL per-robot state (they ride in the migration
bundle). This is the split the PLAN relies on: cold_restart keeps whatever
global actor weights it pulls but loses the local buffer + optimizers → dip.
"""
from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# task2 is CPU-only, always. Do NOT change to cuda — see module docstring.
DEVICE = torch.device("cpu")

OBS_DIM = 11
ACT_DIM = 3
HIDDEN = 256

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6


class Actor(nn.Module):
    """Tanh-squashed Gaussian policy. Shares the mu/log_std head shape with
    hopper_agent.Actor, but adds the squashing + log-prob correction needed for
    a bounded-action online SAC."""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs):
        """Return (action, log_prob) with reparameterized tanh squashing."""
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x = normal.rsample()
        y = torch.tanh(x)
        action = y
        log_prob = normal.log_prob(x) - torch.log(1 - y.pow(2) + EPS)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

    @torch.no_grad()
    def act_deterministic(self, obs):
        """Exploration OFF: tanh(mu). Used for eval rollouts and the policy
        probe."""
        mu, _ = self.forward(obs)
        return torch.tanh(mu)


class Critic(nn.Module):
    """Single Q(s,a). Same concat-MLP shape as hopper_agent.Critic."""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: int = HIDDEN):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1))


class SAC:
    """Twin-critic SAC with automatic entropy tuning. Everything here except the
    actor is LOCAL per-robot state that migrates in the bundle."""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: int = HIDDEN, *, gamma: float = 0.99,
                 tau: float = 0.005, lr: float = 3e-4,
                 target_entropy: float | None = None):
        self.gamma = gamma
        self.tau = tau
        self.act_dim = act_dim

        self.actor = Actor(obs_dim, act_dim, hidden).to(DEVICE)
        self.critic1 = Critic(obs_dim, act_dim, hidden).to(DEVICE)
        self.critic2 = Critic(obs_dim, act_dim, hidden).to(DEVICE)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        for p in self.critic1_target.parameters():
            p.requires_grad_(False)
        for p in self.critic2_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=lr,
        )

        # Automatic entropy temperature.
        self.target_entropy = (
            float(-act_dim) if target_entropy is None else target_entropy
        )
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        self.train_step = 0
        self.last_critic_loss = 0.0
        self.last_actor_loss = 0.0

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    @torch.no_grad()
    def select_action(self, obs_np: np.ndarray, *, deterministic: bool = False):
        obs = torch.as_tensor(obs_np, dtype=torch.float32,
                              device=DEVICE).unsqueeze(0)
        if deterministic:
            a = self.actor.act_deterministic(obs)
        else:
            a, _ = self.actor.sample(obs)
        return a.squeeze(0).cpu().numpy()

    def policy_entropy(self, obs_batch: torch.Tensor) -> float:
        with torch.no_grad():
            _, log_prob = self.actor.sample(obs_batch)
            return float(-log_prob.mean().item())

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        o, a, r = batch["obs"], batch["act"], batch["rew"]
        no, d = batch["next_obs"], batch["done"]

        # --- critics ---
        with torch.no_grad():
            na, nlogp = self.actor.sample(no)
            q1_n = self.critic1_target(no, na)
            q2_n = self.critic2_target(no, na)
            q_n = torch.min(q1_n, q2_n) - self.log_alpha.exp() * nlogp
            tgt = r + self.gamma * (1.0 - d) * q_n

        q1, q2 = self.critic1(o, a), self.critic2(o, a)
        critic_loss = F.mse_loss(q1, tgt) + F.mse_loss(q2, tgt)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # --- actor ---
        a_pi, logp = self.actor.sample(o)
        q1_pi = self.critic1(o, a_pi)
        q2_pi = self.critic2(o, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.log_alpha.exp().detach() * logp - q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # --- temperature ---
        alpha_loss = -(self.log_alpha
                       * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # --- target soft update ---
        with torch.no_grad():
            for p, pt in zip(self.critic1.parameters(),
                             self.critic1_target.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
            for p, pt in zip(self.critic2.parameters(),
                             self.critic2_target.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)

        self.train_step += 1
        self.last_critic_loss = float(critic_loss.item())
        self.last_actor_loss = float(actor_loss.item())
        return {"critic_loss": self.last_critic_loss,
                "actor_loss": self.last_actor_loss,
                "alpha": self.alpha}

    # ---- actor param exchange (FedAvg federates ONLY the actor) ----
    def get_actor_arrays(self):
        return [v.detach().cpu().numpy()
                for v in self.actor.state_dict().values()]

    def set_actor_arrays(self, arrays):
        from collections import OrderedDict
        keys = self.actor.state_dict().keys()
        sd = OrderedDict(
            {k: torch.as_tensor(v, dtype=torch.float32, device=DEVICE)
             for k, v in zip(keys, arrays)}
        )
        self.actor.load_state_dict(sd, strict=True)

    # ---- full local-state (de)serialization for the migration bundle ----
    def state_dict(self) -> dict:
        """Everything needed to resume identically: actor + critics + targets +
        all three optimizers + temperature + counters."""
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "train_step": self.train_step,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic1.load_state_dict(sd["critic1"])
        self.critic2.load_state_dict(sd["critic2"])
        self.critic1_target.load_state_dict(sd["critic1_target"])
        self.critic2_target.load_state_dict(sd["critic2_target"])
        self.actor_opt.load_state_dict(sd["actor_opt"])
        self.critic_opt.load_state_dict(sd["critic_opt"])
        self.alpha_opt.load_state_dict(sd["alpha_opt"])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"].to(DEVICE))
        self.train_step = int(sd.get("train_step", 0))
