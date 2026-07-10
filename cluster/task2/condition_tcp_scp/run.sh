#!/bin/bash
#MSUB -N task2_tcp_scp
#MSUB -W group_list=hpc2-coe-users
#MSUB -l walltime=12:00:00
#MSUB -j oe
# Submit with (2-node footprint):
#   WORKING_NODES="nAAA nBBB" NEED=2 bash tools/submit_free.sh task2/condition_tcp_scp/run.sh

set -uo pipefail
CONDITION_DIR="condition_tcp_scp"
CLUSTER_ROOT="${CLUSTER_ROOT:-$HOME/cluster}"
HERE="$CLUSTER_ROOT/task2/$CONDITION_DIR"

source "$CLUSTER_ROOT/common/cluster_config.sh"
source "$CLUSTER_ROOT/common/cluster_lib.sh"
source "$CLUSTER_ROOT/task2/common/task2_config.sh"

# Non-DHT conditions use a 2-node footprint (server co-located on client node 1).
export MIN_ALIVE_NODES=2

setup_run_dirs "task2_tcp_scp"
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
export CONDITION="tcp_scp"
export RESULTS_DIR="$RESULTS_ROOT/task2_tcp_scp"
mkdir -p "$RESULTS_DIR"

echo ">>> Launching task2 tcp_scp runner" | tee -a "$RUNNER_LOG"
python3 -u "$HERE/runner.py" 2>&1 | tee -a "$RUNNER_LOG"
RC=${PIPESTATUS[0]}
echo ">>> runner exited rc=$RC" | tee -a "$RUNNER_LOG"
exit "$RC"
