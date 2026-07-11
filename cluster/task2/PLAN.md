# Task2 — Learning Continuity + Direct-Transfer Baseline

## Goal
A second cluster experiment on a **real online RL task** that closes two reviewer objections:
1. **Issue #1** — direct TCP/SCP baseline shows DHT transport overhead is trivial vs. direct transfer.
2. **Issue #2** — learning continuity: reward continues smoothly across migration for state-preserving conditions, and drops for the stateless `cold_restart` control.

## Design (locked — for reference)
- **Online RL, CPU-only, forced.** Each robot steps its own `Hopper-v4` (MuJoCo), collects its OWN experience into a LOCAL replay buffer, runs SAC updates (tanh-squashed actions + entropy). Reward = real eval-rollout return (exploration off), must rise below ceiling. `success_rate` = fraction of eval episodes above a fixed target.
- **Shared start:** all conditions/robots start from the SAME initial weights + seed. Federated with Flower FedAvg over policy weights.
- **Migration bundle** = `policy_weights.pt` + optimizer state + **local replay buffer**. Only the transport/checkpoint tool differs per condition.

### CRITICAL — what each condition preserves (the whole point)
Policy weights are GLOBAL (live in FedAvg server), so weights alone don't distinguish conditions. What differs is **LOCAL state**: per-robot replay buffer + optimizer.
- State-preserving (`dht_frl`, `app_cold`, `app_warm`, `tcp_scp`, `dmtcp`): carry the bundle → resume with local replay + optimizer intact → no dip.
- `cold_restart`: empty replay + reset optimizer → must refill from scratch → curve **dips and re-climbs**.
- Implementation must make the buffer MATTER: size it meaningfully, gate SAC updates on a minimum buffer fill, so a wiped buffer causes a visible setback (verified in smoke test).

## Conditions (6, each run once on 2 worker nodes)
| # | Condition | Checkpoint tool | Transport | State? |
|---|-----------|-----------------|-----------|--------|
| 1 | `dht_frl` | torch.save bundle | DHT (Kademlia) | yes |
| 2 | `app_cold` | torch.save bundle | rsync (stop-world) | yes |
| 3 | `app_warm` | torch.save bundle | rsync + bg pre-copy | yes |
| 4 | `tcp_scp` | torch.save bundle | direct socket/scp | yes |
| 5 | `dmtcp` | **DMTCP** full-process img | rsync/scp | yes |
| 6 | `cold_restart` | none (fresh) | — | **no** |

## Status
**DONE:**
- `worker/` — `online_sac_worker.py`, `sac.py`, `replay_buffer.py`, `probe.py`
- `smoke_test.py`
- Conditions built: `dht_frl`, `app_cold`, `app_warm`, `tcp_scp`, `cold_restart`
- Full results (all log files) for `dht_frl` and `cold_restart`

**REMAINING:**
1. **Build `dmtcp` condition** — new trigger: `dmtcp_launch` / `dmtcp_command --checkpoint` dump + transfer + restart on dst. Needs DMTCP userspace build on HPC (see [[project_dmtcp_userspace_install]]; healthy nodes n005/n016/n024/n027/n035).
2. **Run remaining conditions** — `app_cold`, `app_warm`, `tcp_scp` (and `dmtcp`) → produce the same full log set as dht_frl/cold_restart.
3. **Evaluation/figures** (create `evaluation/` or reuse task1 `evaluation2/make_figs.py`):
   - Learning-continuity figure: eval return vs FL round, one line/condition, vertical migration markers.
   - Continuity table: mean `regression_pct`, `recovery_*` per condition.
   - Overhead/latency table (Issue #1): MTT, downtime, network bytes, phase breakdown.
   - Losslessness table: mean `policy_action_mse` / `policy_weight_l2` — ≈0 for state-preserving, large for `cold_restart`.

## Logging schema (all 6 conditions, identical — reference)
- **`migration_events.csv`** — task1 `MigrationMetricsWriter.FIELDNAMES` (MTT, downtime, 4-phase breakdown, network bytes, checkpoint size, cpu util pre/during/post, regression, recovery). `criu_mode` → `checkpoint_mode` (dht_bundle/app/tcp/dmtcp/none). Plus policy-equivalence columns below.
- **`task_logs.csv`** — per round/robot: `reward, eval_return, eval_episode_len, eval_success, success_rate_rolling10, policy_entropy, fl_round, status`.
- **`fl_convergence.csv` / `fl_network.csv` / `fl_hardware.csv` / `fl_latency.csv`** — from Flower server, for ALL conditions.
- **Policy-equivalence probe** (per migration event): fixed seeded 256-obs batch shared across robots/conditions. Record deterministic actions `a_pre` before migration, `a_post` after resume. Columns: `policy_action_mse` = mean((a_post−a_pre)²), `policy_action_kl` (optional), `policy_weight_l2`. ≈0 for state-preserving, LARGE for `cold_restart`.

## Setup / scale
- 2 worker nodes, ~20 robots (10/node), ~5 migrations/robot, N FL rounds (tune for a clear rising curve). CPU forced.
- **Migration trigger** = reuse task1 mechanism: runner's monitor watches Redis `migration_request:robot_*`; worker emits on schedule via `MIGRATION_OFFSET`. SAME schedule across all conditions so markers line up. Record each event's `fl_round` for figure markers.
