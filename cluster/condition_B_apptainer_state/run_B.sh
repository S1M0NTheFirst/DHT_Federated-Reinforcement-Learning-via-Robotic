#!/bin/bash
#MSUB -N SwiftBot_B_AptState
#MSUB -W group_list=hpc2-coe-users
#MSUB -l walltime=03:00:00
#MSUB -l nodes=n005.cluster.pssclabs.com:ppn=8+n006.cluster.pssclabs.com:ppn=8+n016.cluster.pssclabs.com:ppn=8
#MSUB -j oe

set -uo pipefail
CONDITION="condition_B_apptainer_state"
CLUSTER_ROOT="${CLUSTER_ROOT:-$HOME/cluster}"
HERE="$CLUSTER_ROOT/$CONDITION"
source "$CLUSTER_ROOT/common/cluster_config.sh"
source "$CLUSTER_ROOT/common/cluster_lib.sh"

setup_run_dirs "$CONDITION"
cleanup_and_exit() {
    cleanup_all_nodes
    exit "${1:-0}"
}
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
export RESULTS_DIR="$RESULTS_ROOT/$CONDITION"
export CONDITION="apptainer_state"
mkdir -p "$RESULTS_DIR"

echo ">>> Launching runner_B.py" | tee -a "$RUNNER_LOG"
python3 -u "$HERE/runner_B.py" 2>&1 | tee -a "$RUNNER_LOG"
RC=${PIPESTATUS[0]}
echo ">>> runner_B.py exited rc=$RC" | tee -a "$RUNNER_LOG"
exit "$RC"
