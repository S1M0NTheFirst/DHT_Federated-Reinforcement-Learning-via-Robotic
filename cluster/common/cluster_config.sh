#!/bin/bash
# Sourceable config — every run_X.sh sources this first.
# Edit paths here; do not duplicate them in the per-condition scripts.

# --- account / job ---
export GROUP_LIST="hpc2-coe-users"
export BEACH_ID="029822154"

# --- paths on the cluster ---
# Layout assumed:
#   /home/<beach_id>/cluster/        — this folder (orchestration scripts)
#   /home/<beach_id>/swiftbot_rl/    — original code (workers, robot/ modules,
#                                       flower_server.py); bind-mounted into
#                                       apptainer at /app at runtime.
export CLUSTER_ROOT="/home/${BEACH_ID}/cluster"
export SWIFTBOT_RL_ROOT="/home/${BEACH_ID}/swiftbot_rl"
export PROJECT_ROOT="/home/${BEACH_ID}"   # kept for backward-compat with any path that referenced it
export IMG_DIR="${CLUSTER_ROOT}/apptainer"
export LOG_ROOT="${CLUSTER_ROOT}/logs"
export RESULTS_ROOT="${CLUSTER_ROOT}/results"

# --- conda env (carries python+redis CLI for the host-side runner) ---
export CONDA_BASE="/home/${BEACH_ID}/miniconda3"
export CONDA_ENV="base"

# --- experiment knobs (cluster-only — workstation values live in the workers) ---
export NUM_CLIENTS=20             # 10 robots × 2 client nodes
export ROBOTS_PER_NODE=10
export MIGRATION_OFFSET=10        # client_id * 10 (was 25 on workstation)
export TOTAL_TASKS=2000           # 100 rounds × 20 tasks/round
export TOTAL_FL_ROUNDS=100        # bumped from 60 — more samples after the 5 migrations

# --- network ports (chosen to avoid common HPC services) ---
export FLOWER_PORT=8470
export REDIS_PORT=6379            # standard port — avoids needing worker-side patches
export DHT_BASE_PORT=8480

# --- CRIU mode (overridden per condition if needed) ---
export SIMULATE_CRIU="${SIMULATE_CRIU:-0}"
