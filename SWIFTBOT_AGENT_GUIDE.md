# SwiftBot-RL — Agent Working Guide

Concise operational guide for AI agents working on this repo. Instructions only — no code listings. Read `swiftbot_rl/<condition>/` for actual implementation.

---

## What this project is

Empirical comparison of **5 migration mechanisms** for federated robotic RL agents. Paper claim:

> We enable zero-regression migration for learning agents via state-complete transfer, *independent of the underlying checkpoint mechanism*. This is application-level migration, not a GPU-core or low-level technique.

The 5 conditions:

| ID | Folder | What migrates | Engine |
|---|---|---|---|
| **A** | `dht_frl/` | policy weights + replay buffer + metadata bundle (~5 MB) | DHT (Kademlia) over loopback |
| B | `docker_checkpoint/` | full process memory image | `docker checkpoint create` (CRIU under the hood) |
| C | `criu_cold/` | full process memory image | direct `criu dump` / `criu restore` |
| D | `criu_warm/` | iterative pre-copy + final stop-and-copy | CRIU pre-dump |
| E | `cold_restart/` | nothing — kill and re-launch container | Docker only |

A is the proposed system. B-E are baselines.

---

## Repo layout

```
swiftbot_rl/
  dht_frl/             # Condition A (proposed)
  criu_cold/           # Condition C
  criu_warm/           # Condition D
  docker_checkpoint/   # Condition B
  cold_restart/        # Condition E
  evaluation/          # compare_all.py + figures/
  shared/              # metrics_collector.py
```

Each condition folder contains:
- `<condition>_runner.py` — host-side orchestrator (the only thing you run by hand)
- `worker_*_client.py` — script copied into the container; runs the agent loop
- `Dockerfile` — image definition
- `results/` — per-event CSV output

---

## Host environment requirements

- Ubuntu 22.04 (NOT WSL — CRIU needs a real kernel)
- CRIU 3.x+ installed on host (`criu --version`)
- Docker with experimental features enabled (`docker info | grep Experimental` → true)
- NVIDIA Container Toolkit; `docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi` must work
- Redis running on host (`redis-cli ping` → PONG)
- Python 3.10+ with: `docker`, `redis`, `psutil`, `pynvml`, `numpy`, `pandas`, `torch`

A consumer GPU (e.g. RTX 4080) is fine. CRIU's `cuda-checkpoint` plugin will fail PAUSE_DEVICES — runners auto-fall-back to SIMULATE mode. This is expected and is part of the paper's finding.

---

## How to run an experiment

For **every** condition the loop is the same:

1. **Build the image** for that condition: `docker build -f <condition>/Dockerfile -t <image-tag> swiftbot_rl/`
   - A → `swiftbot-robot:latest`
   - B, C, D → `swiftbot-baseline:latest` (shared)
   - E → `swiftbot-cold-restart:latest`
2. **Start Redis** on the host (`redis-server` or `systemctl start redis`).
3. **(Condition A only)** Start the Flower FedAvg server: `python swiftbot_rl/dht_frl/flower_server.py` in a separate terminal.
4. **Run the runner** in a second terminal: `python swiftbot_rl/<condition>/<condition>_runner.py`.
5. The runner launches 8 containers, drives them through ~1000 tasks each with forced migrations, and writes a CSV to `<condition>/results/`.
6. Stop with Ctrl+C if needed; the runner cleans up its containers on exit.

Run conditions one at a time. Don't share container names across runs.

---

## Migration schedule (identical across conditions)

`_MIGRATION_SCHEDULE = [200, 400, 600, 800, 950]` with per-client offset `client_id * 25`.

8 clients × 5 events ≈ 40 migration events per experiment (some are skipped if a robot finishes early or a container dies).

---

## What gets logged

Each migration event is one CSV row in `<condition>/results/`. Key columns:

- `downtime_ms` — wall-clock time agent cannot serve tasks. **The headline metric.**
- `total_MTT_ms` — full migration time including any background work
- `trigger_to_dump_ms`, `dump_to_transfer_ms`, `transfer_to_restore_ms` — phase breakdown
- `checkpoint_size_mb` — bytes shipped per migration
- `network_bytes_transferred`
- `success_rate_pre`, `success_rate_post`, `regression_pct` — learning quality
- `replay_buffer_entries_restored` — A only
- `policy_load_ms` — A only
- `gpu_util_pre/during/post`, `cpu_util_pre/during/post`

---

## Comparing results across conditions

Run `python swiftbot_rl/evaluation/compare_all.py` after all 5 conditions have produced result CSVs. It loads every `results/*.csv` and emits comparison plots into `evaluation/figures/`.

For paper figures, the four claims are:
1. **Lower service interruption** → `downtime_ms`, `dump_to_transfer_ms`, `transfer_to_restore_ms`
2. **Zero learning regression** → `regression_pct`, `success_rate_post`, `replay_buffer_entries_restored`
3. **Smaller migration footprint** → `checkpoint_size_mb` (use log-scale)
4. **Cleaner resource handoff** → `gpu_util_post_migration`, `cpu_util_post_migration`

---

## Known operational gotchas (read before touching runners)

These were learned the hard way — don't undo them.

- **Container launch must be serialized.** Each runner uses an `asyncio.Lock` around `docker create + start`. Parallel launches cause ghost containers stuck in "Created".
- **Docker SDK timeout must be ≥ 300 s.** Default 60 s is too short under nvidia-container-runtime contention. All runners set `docker.from_env(timeout=300)`.
- **Container start has a retry loop** (10 attempts, 2 s backoff) plus a **shell fallback** to `subprocess docker run -d --gpus all` if the SDK keeps failing. Keep both paths.
- **Ghost-robot detection** in the main loop: if `TOTAL_CLIENTS - alive - done > 0`, log and exit. Don't hang waiting forever.
- **Cold restart workers must NOT exit cleanly on migration request.** They set `resume_counter` + `migration_request` in Redis and then loop sleeping until the runner SIGKILLs them. Clean exit caused double-migration races.
- **Cold restart runner must clear stale Redis keys before relaunch** (`migration_request:`, `migration_done:`). Otherwise the new worker inherits the request and re-migrates immediately.
- **CRIU + CUDA fails on consumer GPUs** with `criu failed: type DUMP errno 0`. Runners detect this and fall back to SIMULATE mode (probe RSS via `docker stats`, synthesize dump time at 600 MB/s, transfer at 400 MB/s). Don't try to "fix" CRIU — the failure is the paper's finding.
- **Docker Checkpoint uses the same CRIU engine as Condition C** — same failure mode, same fallback. Document this; don't treat them as independent successes/failures.

---

## Coding conventions

- Runners use `asyncio` and `docker` SDK. Worker scripts are plain blocking Python with `redis-py`.
- Keep runner files self-contained per condition — don't refactor shared logic into a base class. The conditions diverge in subtle ways; readability beats DRY here.
- All timing uses `time.perf_counter()`. Convert to ms at the boundary.
- Per-event CSV rows are written immediately, not batched. A crashed run still produces partial data.
- Don't add new metrics columns without updating `compare_all.py` and all 5 runners in the same change.

---

## Adding a new condition

1. Create `swiftbot_rl/<new>/` mirroring `cold_restart/` (simplest baseline).
2. Implement the runner: container launch loop, migration trigger, per-event CSV writer with the standard column set.
3. Implement the worker: same task loop as `cold_restart/worker_random_client.py`, with whatever pre/post-migration hook your mechanism needs.
4. Write a Dockerfile (reuse `swiftbot-baseline:latest` if no special deps).
5. Add the condition to `evaluation/compare_all.py`'s condition list.
6. Run it; verify `results/<new>_*.csv` has rows for ≥7 of 8 robots.

---

## Quick troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "Exiting with N ghost robot(s)" | container stuck in Created | already handled by retry + shell fallback; if persistent, check Docker daemon health |
| `Read timed out (read timeout=60)` | SDK default timeout | confirm runner uses `timeout=300` |
| `criu failed: type DUMP errno 0` | cuda-checkpoint can't pause GPU | expected; SIMULATE fallback should engage automatically |
| Robot does two migrations 1 s apart | stale `migration_request:` in Redis | confirm runner clears the key before relaunch |
| Empty `checkpoint_size_mb` for A | logging bug — bundle path not measured | re-run after fix in `dht_frl_runner.py` |
| Worker resumes at task 0 after cold restart | `resume_counter:<robot>` missing | runner must SET before kill |

---

## When in doubt

- Read the runner for the condition you're working on; it's the authoritative spec for that experiment.
- Check `results/` for the most recent CSV to see whether the run completed cleanly.
- Don't modify `_MIGRATION_SCHEDULE` or `TOTAL_TASKS` without re-running every condition — the comparison breaks.
