"""
D4RL hopper-medium-v2 offline-RL agent — motivation experiment.

Loads hopper-medium transitions (via Minari, the Farama-maintained port of
D4RL — `mujoco/hopper/medium-v0` is the same dataset as d4rl's
`hopper-medium-v2`), trains a small SAC-style actor for a few hundred steps,
saves the policy state_dict, then idles so the orchestrator can CRIU-dump
the live process.

Runs on GPU (cuda:0) inside the swiftbot-motivation Docker image so the
CRIU dump captures the same runtime footprint as the criu_cold baseline:
CUDA context + model weights resident in VRAM + Python/torch RSS.
"""
import argparse, os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


OBS_DIM = 11
ACT_DIM = 3
HIDDEN  = 256


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        )
        self.mu      = nn.Linear(HIDDEN, ACT_DIM)
        self.log_std = nn.Linear(HIDDEN, ACT_DIM)

    def forward(self, obs):
        h = self.net(obs)
        return self.mu(h), self.log_std(h).clamp(-5, 2)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(OBS_DIM + ACT_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1))


def load_transitions(n_target: int = 50_000):
    """Return (obs, act, rew, next_obs, done) numpy arrays.

    Tries Minari `mujoco/hopper/medium-v0`. Falls back to synthetic data
    if the download fails — the motivation point (checkpoint size) does not
    depend on data quality.
    """
    try:
        import minari
        ds_name = "mujoco/hopper/medium-v0"
        try:
            ds = minari.load_dataset(ds_name)
        except (FileNotFoundError, ValueError):
            print(f"[hopper_agent] downloading {ds_name} ...", flush=True)
            minari.download_dataset(ds_name)
            ds = minari.load_dataset(ds_name)

        obs, act, rew, nxt, done = [], [], [], [], []
        for ep in ds.iterate_episodes():
            o = np.asarray(ep.observations, dtype=np.float32)
            a = np.asarray(ep.actions,      dtype=np.float32)
            r = np.asarray(ep.rewards,      dtype=np.float32)
            obs.append(o[:-1]); nxt.append(o[1:])
            act.append(a);      rew.append(r)
            d = np.zeros(len(r), dtype=np.float32); d[-1] = 1.0
            done.append(d)
            if sum(len(x) for x in obs) >= n_target:
                break
        obs  = np.concatenate(obs)[:n_target]
        act  = np.concatenate(act)[:n_target]
        rew  = np.concatenate(rew)[:n_target]
        nxt  = np.concatenate(nxt)[:n_target]
        done = np.concatenate(done)[:n_target]
        print(f"[hopper_agent] loaded {len(obs)} real hopper transitions",
              flush=True)
        return obs, act, rew, nxt, done
    except Exception as e:
        print(f"[hopper_agent] minari load failed ({e}); using synthetic",
              flush=True)
        rng = np.random.default_rng(0)
        obs  = rng.standard_normal((n_target, OBS_DIM)).astype(np.float32)
        act  = rng.uniform(-1, 1, (n_target, ACT_DIM)).astype(np.float32)
        rew  = rng.standard_normal(n_target).astype(np.float32)
        nxt  = rng.standard_normal((n_target, OBS_DIM)).astype(np.float32)
        done = (rng.random(n_target) < 0.01).astype(np.float32)
        return obs, act, rew, nxt, done


def train(steps: int, batch: int, save_path: str, ready_path: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[hopper_agent] device={device}", flush=True)

    obs, act, rew, nxt, done = load_transitions()
    obs_t  = torch.from_numpy(obs).to(device)
    act_t  = torch.from_numpy(act).to(device)
    rew_t  = torch.from_numpy(rew).unsqueeze(-1).to(device)
    nxt_t  = torch.from_numpy(nxt).to(device)
    done_t = torch.from_numpy(done).unsqueeze(-1).to(device)
    N = len(obs_t)

    actor   = Actor().to(device)
    critic1 = Critic().to(device)
    critic2 = Critic().to(device)
    opt_a   = torch.optim.Adam(actor.parameters(),   lr=3e-4)
    opt_c   = torch.optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()), lr=3e-4)

    for step in range(steps):
        idx = torch.randint(0, N, (batch,))
        o, a, r, no, d = obs_t[idx], act_t[idx], rew_t[idx], nxt_t[idx], done_t[idx]

        with torch.no_grad():
            mu_n, log_std_n = actor(no)
            a_n = mu_n + log_std_n.exp() * torch.randn_like(mu_n)
            q_n = torch.min(critic1(no, a_n), critic2(no, a_n))
            tgt = r + 0.99 * (1 - d) * q_n

        q1, q2 = critic1(o, a), critic2(o, a)
        loss_c = F.mse_loss(q1, tgt) + F.mse_loss(q2, tgt)
        opt_c.zero_grad(); loss_c.backward(); opt_c.step()

        mu, log_std = actor(o)
        a_pi = mu + log_std.exp() * torch.randn_like(mu)
        loss_a = -critic1(o, a_pi).mean()
        opt_a.zero_grad(); loss_a.backward(); opt_a.step()

        if step % 100 == 0:
            print(f"[hopper_agent] step={step:5d} "
                  f"loss_c={loss_c.item():.3f} loss_a={loss_a.item():.3f}",
                  flush=True)

    # App-level checkpoint: just the policy parameters. Move to CPU first
    # so the file is portable and the size matches what FRL would actually
    # ship across nodes.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({k: v.cpu() for k, v in actor.state_dict().items()}, save_path)
    size_kb = os.path.getsize(save_path) / 1024.0
    print(f"[hopper_agent] saved policy state_dict -> {save_path} "
          f"({size_kb:.1f} KB)", flush=True)

    # Drop a ready marker so the orchestrator knows the process is in its
    # steady-state RAM footprint (model + dataset resident, training done).
    with open(ready_path, "w") as fh:
        json.dump({"pid": os.getpid(),
                   "policy_path": save_path,
                   "policy_kb": size_kb}, fh)
    print(f"[hopper_agent] ready marker written -> {ready_path} "
          f"(pid={os.getpid()})", flush=True)

    # Idle so CRIU can dump us.
    while True:
        time.sleep(5)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--save-path", required=True)
    p.add_argument("--ready-path", required=True)
    args = p.parse_args()
    train(args.steps, args.batch, args.save_path, args.ready_path)
