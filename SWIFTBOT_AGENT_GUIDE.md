# SwiftBot-RL — Agent Working Guide

Concise guide for AI agents. Read `swiftbot_rl/<condition>/` for implementation.
This folder is the **workstation (Ubuntu + Docker)** setup and is frozen — new
work goes under `cluster/`.

## What this project is

Empirical comparison of **5 migration mechanisms** for federated robotic RL
agents. Paper claim:

> We enable zero-regression migration for learning agents via state-complete
> transfer, *independent of the underlying checkpoint mechanism*. This is
> application-level migration, not a GPU-core or low-level technique.

| ID | Folder | What migrates | Engine |
|---|---|---|---|
| **A** | `dht_frl/` | policy + replay summary bundle (~160 KB) | DHT (Kademlia) + FedAvg |
| B | `docker_checkpoint/` | full process image | `docker checkpoint` (CRIU) |
| C | `criu_cold/` | full process image | direct `criu dump`/`restore` |
| D | `criu_warm/` | iterative pre-copy + stop-and-copy | CRIU pre-dump |
| E | `cold_restart/` | nothing — kill + relaunch | Docker only |

A is the proposed system; B–E are baselines.

## Design

- **Workload (no dataset):** `SyntheticTaskGenerator` emits fake GPU/CPU tasks.
  Work is pure matrix math — `torch.matmul` (perception) or `np.linalg.eigvalsh`
  (planning). Random size/deadline create load + timing pressure. UCF101/
  LibriSpeech are cited only to motivate task types; they are not processed.
- **Bid gating:** robot bids per task; bid < 0.5 → `declined`, else execute and
  `success` iff latency ≤ deadline. Random policy ≈ 50% decline; trained PPO
  learns to decline under contention.
- **Migration:** 5 forced events per robot at `[200,400,600,800,950]` + offset
  `client_id*25`. 8 robots → ~40 events/run. Each event = one CSV row in
  `<condition>/results/` via `shared/metrics_collector.py`.

## Setup & run

1. Ubuntu 22.04 (real kernel, not WSL); CRIU 3.x; Docker w/ experimental;
   NVIDIA Container Toolkit; `cuda-checkpoint` on host; Redis (`redis-cli ping`).
2. Build image: `docker build -f <condition>/Dockerfile -t <tag> swiftbot_rl/`
   (A → `swiftbot-robot`; B/C/D → `swiftbot-baseline`; E → `swiftbot-cold-restart`).
3. Start Redis. For A also start `dht_frl/flower_server.py`.
4. Run: `python swiftbot_rl/<condition>/<condition>_runner.py` — launches 8
   containers, drives ~1000 tasks each with forced migrations, writes CSV.
5. Run conditions one at a time. `evaluation/compare_all.py` builds figures.

## Results so far (Ubuntu, 8 robots)

Headline = per-migration **checkpoint size**:

| Mechanism | Checkpoint size | Post-mig regression |
|---|---|---|
| CRIU warm (pre-copy image) | **~2.16 GB** | 14–75% |
| CRIU cold | **~547 MB** | 0–75% |
| Docker checkpoint | **~250 MB** | mixed |
| Cold restart | 0 (no state) | high |
| **DHT+FRL (ours)** | **~160 KB** | **~0%** |

Takeaway: container-level checkpointing ships hundreds of MB–GBs (Python runtime,
CUDA libs, full VRAM), while the semantically necessary state to resume on-policy
updates is only ~160 KB — a >3000× reduction with near-zero success regression.
