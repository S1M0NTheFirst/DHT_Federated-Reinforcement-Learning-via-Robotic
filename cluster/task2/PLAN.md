# Task2 — Learning Continuity + Direct-Transfer Baseline

## Goal
A second cluster experiment on a **real RL task** (not the saturated deadline
workload of task1) that closes two reviewer objections:
1. **Issue #1** — add a direct TCP/SCP baseline; show DHT transport overhead is
   trivial vs. direct transfer (single-event latency is transport-independent).
2. **Issue #2** — show learning continuity: reward/success continues smoothly
   across migration for state-preserving conditions, and drops for the
   stateless `cold_restart` control.

## RL task — ONLINE RL (mandatory; NOT offline)
The agent MUST learn **online**: each robot interacts with its OWN live
environment, collects its OWN experience, and trains on it. This is required —
an offline/static-dataset setup (like the current `hopper_agent.py`) produces
NO rising learning curve AND leaves nothing meaningful for `cold_restart` to
lose (the "replay buffer" would just be a reloadable public dataset), so the
issue #2 continuity contrast would collapse. Online is what makes BOTH the
reward *rise* and the cold-restart *drop* real.

- **Env:** each robot runs its own `gymnasium.make("Hopper-v4")` (MuJoCo). This
  is NEW — `hopper_agent.py` never steps an env (it only trains on a frozen
  Minari dataset). Reuse `hopper_agent.py` ONLY for the `Actor`/`Critic` network
  shapes, not its offline training loop.
- **Online loop per robot:** reset env → {observe → act → env.step → store
  transition in a LOCAL per-robot replay buffer → SAC update} repeated; the
  buffer grows from the robot's OWN rollouts (naturally heterogeneous across the
  fleet). This local, hard-won buffer is the state migration must preserve.
- **Proper SAC actor:** upgrade the crude actor (add tanh-squashed bounded
  actions + entropy term) so online Hopper learns stably; the current
  `loss_a = -critic1(o,a_pi).mean()` with unbounded actions will NOT learn online.
- **CPU-ONLY, forced (all 6 conditions).** Hard-override device to `cpu`.
  Rationale: DMTCP (like CRIU) cannot checkpoint a live CUDA process, so a heavy
  real checkpoint baseline is only possible on CPU; Hopper + tiny SAC is light on
  CPU; task1 ran GPU-idle so CPU keeps both experiments comparable. (MuJoCo sim
  is the main new CPU cost — fine for 20 light robots.)
- **Data is REAL, self-generated:** transitions come from live MuJoCo rollouts,
  not a downloaded dataset. No Minari, no synthetic fallback.
- Federated with Flower FedAvg over policy weights, same structure as task1
  Condition A (`cluster/condition_A_dht_frl/`).
- **Shared start for fair comparison:** all 6 conditions (and all robots) start
  from the SAME initial policy weights + same seed, so the learning curves are
  directly comparable and any divergence is attributable to the migration
  mechanism, not initialization. (task1 seeded FedAvg from a shared pretrained
  policy for this reason.)
- **Reward curve = REAL eval return:** each FL round, run the current policy in
  the env for a few episodes with exploration OFF and log the mean **episode
  return**. This must RISE over rounds and sit below the env ceiling (genuine
  learning, no saturation like task1).
- `success_rate` = fraction of eval episodes whose return exceeds a fixed target
  threshold (define per env; must not peg at 1.0).

## Migration = unified state bundle
Bundle = `policy_weights.pt` + optimizer state + **local replay buffer**
(the robot's own online experience — now genuinely valuable, unlike a static
dataset). Same bundle concept as task1 Condition A. Only the
**transport / checkpoint tool** differs per condition. `cold_restart` discards
this bundle → loses the hard-won replay buffer + optimizer → curve drops.

## Conditions (6 total, each run ONCE on 2 worker nodes)
| # | Condition   | Checkpoint tool            | Transport            | Preserves state? | Purpose |
|---|-------------|----------------------------|----------------------|------------------|---------|
| 1 | `dht_frl`   | torch.save bundle          | DHT (Kademlia)       | yes | our system |
| 2 | `app_cold`  | torch.save bundle          | rsync (stop-world)   | yes | light app-checkpoint baseline |
| 3 | `app_warm`  | torch.save bundle          | rsync + bg pre-copy  | yes | pre-copy variant of app_cold |
| 4 | `tcp_scp`   | torch.save bundle          | direct TCP socket / scp | yes | direct-transfer control → Issue #1 |
| 5 | `dmtcp`     | **DMTCP** full-process img | rsync/scp            | yes | heavy real checkpoint DHT beats on latency/restore |
| 6 | `cold_restart` | none (fresh from pretrained) | —              | **no** | negative control → Issue #2 |

Notes:
- `dmtcp` is COLD only (no warm variant in this plan).
- All 6 run identical RL training + identical migration schedule + identical
  logging. Only the checkpoint/transport box changes.

### CRITICAL — what each condition preserves vs discards at migration (get this right)
In FL the **policy weights are global** (they live in the FedAvg server), so
weights alone are NOT what distinguishes the conditions. What differs is the
**LOCAL state**: the per-robot **replay buffer** + **optimizer state**.
- State-preserving conditions (`dht_frl`, `app_cold`, `app_warm`, `tcp_scp`,
  `dmtcp`) carry the bundle → resume with **local replay + optimizer intact** →
  learning continues smoothly, no dip.
- `cold_restart` resumes with an **empty replay buffer + reset optimizer** (it
  keeps only whatever global weights it pulls). It must then **refill the buffer
  from scratch** and rebuild optimizer momentum → the eval-return curve
  **dips and re-climbs**. This dip is the WHOLE point of issue #2.
- Implementation must therefore make the **local replay buffer matter**: size it
  meaningfully, and gate SAC updates on a minimum buffer fill so a wiped buffer
  causes a real, visible learning setback (not an instant recovery). If a wiped
  buffer recovers instantly, the continuity contrast disappears — verify in the
  smoke test.
- For the **policy-equivalence probe**: `cold_restart`'s post-resume policy is
  whatever it reloads; expect its action divergence to be LARGE relative to the
  state-preserving conditions (which are ~0). If cold_restart also pulls fresh
  global weights, still expect divergence from the degraded local trajectory.

## What each condition proves
- **Issue #1:** `tcp_scp` ≈ `app_cold` ≈ `dht_frl` on single-event MTT/downtime
  → bundle win is transport-independent; DHT overhead small.
- **Issue #2:** curves of `dht_frl`/`app_cold`/`app_warm`/`dmtcp` stay continuous
  across migration markers; `cold_restart` drops to floor and re-climbs.
- **Winning margin:** `dht_frl` beats `dmtcp` on latency/restore (real heavy
  checkpoint tool, not a stand-in) → retires task1's "faked CRIU" criticism.
- **Losslessness:** policy-equivalence probe (see Logging #5) ≈0 for all
  state-preserving conditions, LARGE for `cold_restart` → migration is
  behaviorally bit-for-bit lossless, proven numerically not just via reward.

## Repository layout (create under `cluster/task2/`)
ONE shared online worker + ONE Flower server + a thin runner per condition
(the runner only swaps the migration `trigger_fn`; the RL loop is identical).
```
cluster/task2/
  PLAN.md                       # this file
  worker/
    online_sac_worker.py        # the online RL Flower client (SHARED by all conditions)
    sac.py                      # Actor/Critic (shapes from hopper_agent.py) + SAC update
    replay_buffer.py            # local per-robot replay buffer (part of the bundle)
    probe.py                    # fixed seeded probe batch + a_pre/a_post capture
  flower_server.py              # FedAvg + writes fl_convergence/fl_* for ALL conditions
  common/                       # import task1 cluster/common/* ; task2 config overrides only
  condition_dht_frl/     runner.py  run.sh
  condition_app_cold/    runner.py  run.sh
  condition_app_warm/    runner.py  run.sh
  condition_tcp_scp/     runner.py  run.sh
  condition_dmtcp/       runner.py  run.sh
  condition_cold_restart/runner.py  run.sh
  smoke_test.py                 # single-robot online gate (see Smoke-test gate)
  results/<condition>/          # migration_events.csv, task_logs.csv, fl_*.csv
  evaluation/                   # task2 figure/table scripts (or reuse ../evaluation2)
```
- **Shared worker:** all 6 conditions run `worker/online_sac_worker.py`. Only
  `cold_restart` and `dht_frl` need small behavior flags (cold_restart discards
  the bundle on resume; dht_frl joins the DHT) — gate via env var, do NOT fork
  the worker.
- **Per-condition runner:** clone task1's structure; each `runner.py` supplies
  condition name + `trigger_fn`. Transport is the ONLY difference between
  dht_frl / app_cold / app_warm / tcp_scp / dmtcp.

## Implementation (reuse task1 harness)
- Reuse `cluster/common/` scaffolding: `run_baseline(...)`,
  `MigrationMetricsWriter`, migration monitor, SSH-master pooling, launch/kill.
- Each condition = a `runner_<name>.py` supplying: condition name, worker
  script, and a `trigger_fn(cfg, r, robot_id, sr_pre, tc_pre) -> metrics dict`.
  Clone from the closest task1 condition:
  - `dht_frl` ← `condition_A_dht_frl/runner_A.py`
  - `app_cold`/`app_warm` ← `condition_C`/`condition_D` (rename, drop "CRIU")
  - `tcp_scp` ← new trigger: point-to-point socket/scp of the 3 bundle files
  - `dmtcp` ← new trigger: `dmtcp_launch`/`dmtcp_command --checkpoint` dump +
    transfer + restart on dst
  - `cold_restart` ← `condition_E_cold_restart/runner_E.py`
- Worker: a Flower client running the ONLINE loop — (a) steps its own
  `Hopper-v4` env collecting transitions into a local replay buffer and runs SAC
  updates on CPU (device forced to `cpu`), (b) federates policy weights each
  round, (c) runs a periodic eval rollout (exploration off) in the env,
  (d) writes the bundle (policy + optimizer + local replay buffer) to
  `/checkpoints/<robot>/`, (e) logs per-round eval return + success to Redis
  `task_logs` AND persists `task_logs.csv` per condition (this file was MISSING
  in task1 — must persist it here). Reuse ONLY the `Actor`/`Critic` shapes from
  `hopper_agent.py`; replace its offline dataset training with the online loop
  and a proper SAC actor (tanh-squashed actions + entropy).
- Metrics schema = task1 `MigrationMetricsWriter.FIELDNAMES` (latency, network
  bytes, hardware, phase breakdown, regression) + `task_logs.csv` time series.
- **Smoke-test gate (before any cluster run):** run ONE robot online locally for
  the full round count; confirm (i) eval return RISES and plateaus below ceiling,
  and (ii) simulating a cold restart (wipe local replay + optimizer, keep global
  weights) makes the return visibly DROP then recover. If both don't hold, the
  cluster run won't either — do not submit until they do.

## Logging (ALL 6 conditions log the SAME full set — no exceptions)
Every condition records, for every migration event and every FL round:
- **Migration/latency:** total MTT, downtime, 4-phase breakdown
  (dump / transfer / restore / load) — task1 `MigrationMetricsWriter.FIELDNAMES`.
- **Network:** bytes transferred, checkpoint/bundle size.
- **Hardware:** CPU util pre/during/post (GPU fields present but 0 — CPU-only).
- **Continuity:** `success_rate_pre/post`, `regression_pct`, `recovery_*`.
- **Learning time series → `task_logs.csv`:** per-round eval return + success
  per robot (the file MISSING in task1 — MUST be persisted here for every
  condition, including `cold_restart`).
Identical schema across conditions so all outputs/figures are directly
comparable and drop into the `evaluation2` plotting.

### Log EVERYTHING task1 logged, for EVERY condition (log-first, pick later)
Task1 produced these files; task2 must produce all of them for all 6 conditions
(task1 only wrote the `fl_*` files for Condition A — fix that: write them for
all conditions from the Flower server / worker).

1. **`migration_events.csv`** (per migration event) — full task1 schema
   (`MigrationMetricsWriter.FIELDNAMES`):
   `condition, robot_id, migration_event_id, timestamp, src_node, dst_node,
   trigger_to_dump_ms, dump_to_transfer_ms, transfer_to_restore_ms,
   policy_load_ms, downtime_ms, total_MTT_ms, success_rate_pre,
   success_rate_post, regression_pct, fl_rounds_to_recover,
   replay_buffer_entries_restored, gpu_util_{pre,during,post}_migration,
   cpu_util_{pre,during,post}_migration, network_bytes_transferred,
   checkpoint_size_mb, criu_mode, throughput_post_60s, recovery_tasks_to_pre,
   background_bandwidth_mb, concurrency_level, fault_injected, retry_count,
   total_recovery_ms`.
   (`criu_mode` → repurpose as `checkpoint_mode`: dht_bundle / app / tcp / dmtcp / none.)
2. **`task_logs.csv`** (per eval step / per round, per robot — MUST persist,
   missing in task1). Task1 fields to keep + RL adaptations:
   `robot_id, fl_round, training_step, reward, success_rate_rolling10,
   policy_entropy, status` + NEW `eval_return, eval_episode_len,
   eval_success` (real Hopper rollout). Drop task1-only fields that don't apply
   to Hopper (`task_type, complexity, deadline_ms, bid_value`) or keep as null.
3. **`fl_convergence.csv`** (per FL round, from Flower server — write for ALL
   conditions): `round, train_loss, mean_reward, success_rate, policy_entropy,
   cpu_usage, gpu_usage, network_mb, train_time, total_latency`.
4. **`fl_network.csv`** `round, network_mb`; **`fl_hardware.csv`**
   `round, cpu_usage, gpu_usage`; **`fl_latency.csv`**
   `round, total_latency, train_time` (subsets of fl_convergence — keep for
   drop-in compatibility with `evaluation2/make_figs.py`).
5. **Policy-equivalence probe (behavioral losslessness) — per migration event.**
   Direct, numerical proof that the restored policy is the SAME policy, not just
   that reward looks continuous.
   - **Mechanism:** the worker keeps a FIXED probe batch of observations
     (e.g. 256 obs sampled once at startup, identical across all robots/conditions,
     seeded). Immediately BEFORE migration it records the pre-migration policy's
     deterministic actions on the probe batch (`a_pre`). Immediately AFTER resume
     it recomputes actions on the same batch (`a_post`).
   - **Metrics (add as columns to `migration_events.csv`):**
     `policy_action_mse` = mean((a_post − a_pre)²),
     `policy_action_kl` = KL(π_post‖π_pre) over the probe batch (optional, if
     using the stochastic actor's mean+log_std),
     `policy_weight_l2` = L2 norm of (weights_post − weights_pre).
   - **Expected result:** ≈0 for all state-preserving conditions (`dht_frl`,
     `app_cold`, `app_warm`, `tcp_scp`, `dmtcp`) → migration is bit-for-bit
     lossless; LARGE for `cold_restart` (reset to generic policy). This turns
     "we preserve learned state" from a claim into a hard number a reviewer
     cannot dispute. Cheap: a few CPU forward passes on a small batch.
   - Log the raw per-event value (not pre-aggregated) so it can be aggregated later.

## Setup / scale
- 2 worker nodes, ~20 robots (10/node), ~5 migrations/robot, N FL rounds long
  enough to show a clear rising curve (tune N in the smoke test; likely more
  rounds than task1's 100 since online learning is slower). MOAB/Apptainer like task1.
- **Migration trigger = reuse task1's mechanism:** the runner's migration monitor
  watches Redis `migration_request:robot_*` keys; the worker emits them on a
  schedule via `MIGRATION_OFFSET` (see `worker_robot_client.py` / `ClusterConfig`).
  Keep the SAME schedule across all 6 conditions so migration markers line up on
  the figures. Each migration moves a robot node1↔node2.
- **Migration markers:** record each event's `fl_round` (and wall-clock) so the
  continuity figure can draw vertical lines at the exact rounds migrations fired.
- `run.sh` per condition, cloned from task1 `run_A.sh` (env: REDIS, Flower bind,
  N_ROUNDS, MIGRATION_OFFSET, forced CPU, condition name).

## Outputs / figures
1. **Learning-continuity figure:** eval return vs FL round, one line/condition,
   vertical migration markers. State-preserving = continuous; `cold_restart` = drops.
2. **Continuity table:** per condition mean `regression_pct`, `recovery_*`.
3. **Overhead/latency table (Issue #1):** MTT, downtime, network bytes, phase
   breakdown for `dht_frl` vs `tcp_scp` vs `app_cold`/`app_warm` vs `dmtcp`.
4. **Losslessness table:** per condition mean `policy_action_mse` /
   `policy_weight_l2` — ≈0 for state-preserving, large for `cold_restart`.

## Pre-flight checks (before building)
- [ ] `gymnasium` + **MuJoCo** (`Hopper-v4`) install and run in the Apptainer
      image on CSULB HPC (CPU). This is the core online-RL dependency. If MuJoCo
      is painful, fall back to a lighter online env (e.g. `LunarLander-v2` /
      classic-control) that still gives a rising, perturbable curve.
- [ ] DMTCP installed/installable in the image; can dump the **CPU** Hopper/torch
      process (GPU dumps NOT supported — CPU-only).
- [ ] Single-robot ONLINE smoke test passes: eval return rises + plateaus below
      ceiling, AND a simulated cold restart makes it visibly drop then recover.
- [ ] 20 online robots (each a MuJoCo sim) fit the node CPU budget without
      thrashing (check per-worker thread caps as in task1 `WORKER_MATH_THREADS`).
