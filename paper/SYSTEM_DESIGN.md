# DHT-FRL System Design

This document describes the architecture of **DHT-FRL** (DHT-based Federated
Reinforcement Learning for edge robotic fleets), with a focus on the
**migration mechanism** — the core contribution of the paper.

---

## 1. Overview

DHT-FRL is a system for running on-device reinforcement learning across a
fleet of edge robots, where individual robots are periodically **migrated**
between compute nodes (for load balancing, maintenance, thermal throttling,
or failure recovery). The central design goal is to make migration **fast,
lightweight, and hardware-independent**, so that a migration event does not
interrupt the robot's productive work or discard its learned policy.

The key idea: instead of snapshotting the *entire process* (as CRIU does),
DHT-FRL transfers only the **semantically necessary agent state** — a small
"unified state bundle" of roughly **160 KB** — through a distributed hash
table (DHT) overlay that the federated-learning layer already maintains.

---

## 2. System Components

The system runs across **three physical nodes** connected by Ethernet.

### 2.1 Coordinator Node

Hosts the control- and learning-plane services:

| Component | Role |
|-----------|------|
| **Flower FL Server** (port 8470) | Runs FedAvg aggregation. Collects per-robot weight updates each round, averages them, and broadcasts the global model back. Seeds the initial global model from a pretrained policy. |
| **DHT Overlay** (Kademlia) | Content-addressed key–value store, co-resident with the FL server. Holds the latest federated weights (keyed by round/hash) and the per-event migration bundle. |
| **Redis Bus** (port 6379) | Coordination signal bus. Carries `migration_request`, `ready_for_criu`, `load_policy`, `first_bid_after_migration`, `resume_counter`, `task_logs`, etc. |
| **Experiment Runner** | Orchestrates migration events: detects triggers, drives the migration protocol, records per-event metrics to CSV. |

### 2.2 Worker Nodes A and B

Each worker node hosts **10 robot workers** (robots 0–9 on Node A, robots
10–19 on Node B), each running inside its own **Apptainer container** built
from a single shared image. Each robot worker runs:

- **`RobotPPOAgent`** — a PPO agent built on `BidPolicyMLP` (state_dim = 15,
  hidden = 64, sigmoid bid output). PPO updates are driven by task rewards.
- **`SyntheticTaskGenerator`** — produces a stream of bidding-style compute
  tasks (GPU-heavy / CPU-heavy matmul and eigendecomposition).
- **`SensorHistoryBuffer`** — tracks recent task outcomes to construct the
  agent's state vector.
- **Flower FL client** — each round, the robot sends its weight delta to the
  FL server and receives the updated global model.

A robot's task loop is simple: **read state → bid on task → execute task →
receive reward → PPO update → repeat**, while participating in FedAvg rounds
in the background.

Robots migrate **only between Node A and Node B** — never to or from the
Coordinator Node.

---

## 3. What Migrates (and Why It's Small)

Traditional checkpoint-based migration (CRIU) ships the union of three things:

1. The **OS process image** — file descriptors, sockets, memory mappings.
2. The **ML training state** — policy weights, optimizer moments, replay buffer.
3. The **CUDA driver state** — allocator caches, contexts, streams.

Only **(2)** is semantically necessary for an RL agent to resume productive
work. Items (1) and (3) are re-creatable on the destination from the
application image plus a fresh CUDA context.

DHT-FRL transfers **only (2)**, packaged as a **unified state bundle**:

```
State Bundle  (≈ 160 KB)
├── policy_weights.pt   — PPO policy network parameters
├── replay_buffer.pkl   — compact replay / reward summary
└── manifest.json       — robot_id, training_step, RNG seed, round number
```

Because the federated weights are already replicated to every peer through
the FL aggregator between rounds, the migration mechanism does not even have
to ship the full weights in the steady state — it points the destination at
the DHT and lets the overlay deliver the current model.

This is the heart of the contribution: **~160 KB vs. 547 MB (CRIU cold) /
2.16 GB (CRIU warm)** for the same agent — a roughly **3,000× reduction** in
migration payload.

---

## 4. The Migration Protocol (Core Mechanism)

A migration event for robot `R` moving from a **source node** to a
**destination node** executes five phases. The phase names match the metric
columns reported in the evaluation.

### Phase ① TRIGGER
The Experiment Runner sets `migration_request:<robot_id>` in Redis, naming
the destination host. This is the externally-driven signal (load balancing,
node drain, failure recovery).

### Phase ② DUMP
The source robot finishes its current task and flushes its state bundle to
its local `/checkpoints/<robot_id>/` directory — policy weights, replay
summary, and manifest (`training_step`, RNG seed, round number). It signals
completion via the `ready_for_criu:<robot_id>` Redis key. The bundle is
published into the DHT under a per-event key.

To preserve correctness, the source agent's task loop is gated on a
per-robot `migrating` flag set **before** the bundle is published, so the
source does not push further weight deltas after the snapshot is taken
(the **no double-write** property).

### Phase ③ TRANSFER
The runner `rsync`s the bundle directory from the source node to the
destination node (`/checkpoints/<robot_id>_dest/`) over an SSH connection
multiplexed per robot (`ControlMaster`). The destination peer fetches the
bundle by key from the DHT and fetches the latest federated weights by hash.

The destination must load weights **at least as recent** as the source's
last completed FedAvg round (the **weight freshness** property): weights are
keyed by round number, and the destination refuses to resume until it has
fetched a key ≥ the source's last round.

This phase touches **no disk I/O on the critical path** and ships no full
process image.

### Phase ④ RESTORE
The destination worker loads the bundle (`torch.load()`, ~2.8 ms), seeds the
RNG from the manifest, and rebuilds a **fresh CUDA context** from scratch.
Because no CUDA state crosses hosts, the migration is **independent of GPU
model** — a context captured on an RTX 3090 does not need to restore on a
Tesla P100. The source instance is then torn down.

### Phase ⑤ RESUME
The destination robot resumes its task loop from the restored `training_step`,
rejoins the FedAvg round as a client, and confirms via
`first_bid_after_migration`. The runner sets `resume_counter` in Redis;
migration is complete. The robot is back to **productive work with zero
policy regression**.

```
①TRIGGER ──► ②DUMP ──► ③TRANSFER ──► ④RESTORE ──► ⑤RESUME
   │                                       │            │
   └──────────── downtime_ms ─────────────┘            │
   └──────────────────── total_MTT_ms ─────────────────┘
```

- **`downtime_ms`** — time the robot is offline (phases ①–④). Median ≈ 1.20 s.
- **`total_MTT_ms`** — trigger to first task on the destination (phases ①–⑤).
  Median ≈ 2.95 s.

---

## 5. Warm and Cold Variants

To match the warm/cold distinction used in CRIU evaluation, DHT-FRL defines
two operating points:

- **Warm DHT** (steady state) — the destination host is already running a DHT
  peer and has cached the latest federated weights from normal FedAvg
  participation. Migration cost reduces to the per-event bundle publish plus a
  local weight load.
- **Cold DHT** (recovery case) — the destination host joins the DHT at
  migration time, adding DHT bootstrap and a full weight fetch over the
  overlay. This is the worst case for our mechanism and is what we compare
  against CRIU's cold variant.

---

## 6. Correctness Properties

Two properties guarantee that migration preserves learning:

1. **Weight freshness** — the destination loads weights at least as recent as
   the source's last completed FedAvg round. Enforced by keying weights by
   round number and refusing to resume until a key ≥ the source's last round
   is fetched.
2. **No double-write** — the source must not push further deltas after the
   bundle is published. Enforced by a per-robot `migrating` flag set before
   bundle publish and cleared only on the destination after resume.

---

## 7. Why This Beats Checkpoint-Based Migration

| Property | CRIU checkpoint | DHT-FRL |
|----------|-----------------|---------|
| Migration payload | 547 MB – 2.16 GB | ~160 KB |
| Critical-path disk I/O | Yes (full image dump/restore) | No |
| Restore phase | ~5 s (process reconstruction) | None (load weights, 2.8 ms) |
| GPU-model portability | Fails on heterogeneous hosts | Hardware independent |
| Median downtime | ~6.4 s | ~1.2 s |
| Policy regression | Possible | Zero |

The advantage is **structural**, not a matter of tuning: by representing the
migrating state as a small federated bundle rather than as a process snapshot,
DHT-FRL eliminates the restore phase that dominates CRIU's migration time and
removes the disk I/O and GPU-context dependencies entirely.

---

## 8. Implementation Notes

- ~3.2 KLOC of Python. Agent uses PyTorch PPO; FL via Flower 1.x with a
  standard FedAvg strategy; DHT is a Kademlia implementation co-resident with
  each agent process; orchestration and per-event logging via Redis.
- The same agent and DHT code is deployed on both testbeds; the only
  environment difference is the container runtime — **Docker** on the Ubuntu
  workstation (where real kernel CRIU is measured), and **Apptainer** on the
  HPC2 cluster (where scale-out and tail-latency are measured).
- On the HPC2 cluster, `/tmp` is node-local, so the bundle is replicated to
  the destination's local checkpoint path before the worker reads it. This is
  faithfully described in the paper as application-level bundle-transfer
  migration.
