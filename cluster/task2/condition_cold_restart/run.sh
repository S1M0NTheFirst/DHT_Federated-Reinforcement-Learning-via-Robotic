#!/bin/bash
#MSUB -N task2_cold_restart
#MSUB -W group_list=hpc2-coe-users
#MSUB -l walltime=12:00:00
#MSUB -j oe
# Submit with:
#   cluster/tools/submit_free.sh cluster/task2/condition_cold_restart/run.sh

set -uo pipefail
CONDITION_DIR="condition_cold_restart"
CLUSTER_ROOT="${CLUSTER_ROOT:-$HOME/cluster}"
HERE="$CLUSTER_ROOT/task2/$CONDITION_DIR"

source "$CLUSTER_ROOT/common/cluster_config.sh"
source "$CLUSTER_ROOT/common/cluster_lib.sh"
source "$CLUSTER_ROOT/task2/common/task2_config.sh"

# Non-DHT conditions use a 2-node footprint: server co-located on client node 1
# (robots 0..9 + server on node0, robots 10..19 on node1). Still cross-node
# migration, one fewer node than DHT.
export MIN_ALIVE_NODES=2

setup_run_dirs "task2_cold_restart"
cleanup_and_exit() { cleanup_all_nodes; exit "${1:-0}"; }
trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_all_nodes' EXIT

pick_alive_nodes || exit 1
start_redis_on_server || exit 1

source "$CONDA_BASE/bin/activate" "$CONDA_ENV"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$CLUSTER_ROOT:${PYTHONPATH:-}"
export SERVER_NODE CLIENT_NODE_1 CLIENT_NODE_2
export REDIS_HOST="$SERVER_NODE"
export CONDITION="cold_restart"
export RESULTS_DIR="$RESULTS_ROOT/task2_cold_restart"
mkdir -p "$RESULTS_DIR"

echo ">>> Launching task2 cold_restart runner" | tee -a "$RUNNER_LOG"
python3 -u "$HERE/runner.py" 2>&1 | tee -a "$RUNNER_LOG"
RC=${PIPESTATUS[0]}
echo ">>> runner exited rc=$RC" | tee -a "$RUNNER_LOG"
exit "$RC"
