#!/bin/bash
# task2 config — sourced by every task2 run.sh AFTER cluster/common/cluster_config.sh.
# Overrides task1 knobs for the online-RL experiment. Do NOT duplicate paths that
# already live in cluster/common/cluster_config.sh.

# task2 lives under the same cluster root.
export TASK2_ROOT="${CLUSTER_ROOT}/task2"
export TASK2_PYLIBS2="${TASK2_ROOT}/apptainer/pylibs2"   # mujoco layer (bind alongside pylibs)

# --- online-RL knobs (env-tunable; calibrate in the smoke test first) ---
export TOTAL_FL_ROUNDS=150            # online learning is slower than task1's 100
export STEPS_PER_ROUND=1000           # env steps per robot per FL round
export MIN_BUFFER_FILL=1000           # gate SAC updates → wiped buffer = real dip
export BUFFER_CAPACITY=100000         # meaningful local replay
export SAC_BATCH=256
export EVAL_EPISODES=3
export EVAL_SUCCESS_RETURN=800        # success threshold (must not peg at 1.0)
export TASK2_ENV="Hopper-v4"
export SHARED_SEED=12345              # identical global-actor start for all conditions

# --- migration schedule in FL ROUNDS (identical across all 6 conditions) ---
export MIGRATION_ROUNDS="30,60,90,120,140"

# --- FORCE CPU for all conditions (DMTCP can't dump live CUDA; keeps all
#     conditions comparable). Set on host; propagated into containers. ---
export CUDA_VISIBLE_DEVICES=""

# task2 uses the same fleet size as task1.
export NUM_CLIENTS=20
export ROBOTS_PER_NODE=10

# task2 ports — offset from task1 so a task2 run can coexist with a task1 run.
export FLOWER_PORT=8570
export REDIS_PORT=6579
