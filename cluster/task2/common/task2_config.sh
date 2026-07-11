#!/bin/bash
# task2 config — sourced by every task2 run.sh AFTER cluster/common/cluster_config.sh.
# Overrides task1 knobs for the online-RL experiment. Do NOT duplicate paths that
# already live in cluster/common/cluster_config.sh.

# task2 lives under the same cluster root.
export TASK2_ROOT="${CLUSTER_ROOT}/task2"
export TASK2_PYLIBS2="${TASK2_ROOT}/apptainer/pylibs2"   # mujoco layer (bind alongside pylibs)

# Keep ALL task2 logs + results INSIDE the task2 folder (not cluster/logs,
# cluster/results). Overrides cluster_config.sh; must be set before
# setup_run_dirs runs in run.sh.
export LOG_ROOT="${TASK2_ROOT}/logs"
export RESULTS_ROOT="${TASK2_ROOT}/results"

# --- online-RL knobs (env-tunable; calibrate in the smoke test first) ---
export TOTAL_FL_ROUNDS=150            # online learning is slower than task1's 100
export STEPS_PER_ROUND=1000           # env steps per robot per FL round
export MIN_BUFFER_FILL=1000           # gate SAC updates → wiped buffer = real dip
export BUFFER_CAPACITY=100000         # meaningful local replay
export SAC_BATCH=256
export EVAL_EPISODES=3
export EVAL_SUCCESS_RETURN=250        # success threshold; ~mid-plateau so success
                                      # sits between 0 and 1 (smoke plateau ~300).
                                      # Raise if the full 150-round run climbs higher.
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

# --- quick-test overrides (for smoke-testing a condition at tiny scale) ---
# Pass via msub -v, e.g. for a fast dmtcp checkpoint sanity run:
#   ... bash tools/submit_free.sh task2/condition_dmtcp/run.sh \
#         -v OVR_NUM_CLIENTS=2,OVR_FL_ROUNDS=8,OVR_MIGRATION_ROUNDS=3
[ -n "${OVR_NUM_CLIENTS:-}" ]     && export NUM_CLIENTS="$OVR_NUM_CLIENTS"
[ -n "${OVR_FL_ROUNDS:-}" ]       && export TOTAL_FL_ROUNDS="$OVR_FL_ROUNDS"
[ -n "${OVR_MIGRATION_ROUNDS:-}" ] && export MIGRATION_ROUNDS="$OVR_MIGRATION_ROUNDS"
