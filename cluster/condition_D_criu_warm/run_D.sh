#!/bin/bash
#MSUB -N SwiftBot_D_CriuWarm
#MSUB -W group_list=hpc2-coe-users
#MSUB -l walltime=06:00:00
#MSUB -l nodes=n020.cluster.pssclabs.com:ppn=8+n027.cluster.pssclabs.com:ppn=8+n033.cluster.pssclabs.com:ppn=8
#MSUB -j oe

set -uo pipefail
CONDITION="condition_D_criu_warm"
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

# Condition D uses app-level checkpointing + warm pre-copy, not kernel CRIU.

source "$CONDA_BASE/bin/activate" "$CONDA_ENV"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$CLUSTER_ROOT:${PYTHONPATH:-}"
export SERVER_NODE CLIENT_NODE_1 CLIENT_NODE_2
export REDIS_HOST="$SERVER_NODE"
export RESULTS_DIR="$RESULTS_ROOT/$CONDITION"
export CONDITION="criu_warm"
mkdir -p "$RESULTS_DIR"

echo ">>> Launching runner_D.py" | tee -a "$RUNNER_LOG"
python3 -u "$HERE/runner_D.py" 2>&1 | tee -a "$RUNNER_LOG"
RC=${PIPESTATUS[0]}
echo ">>> runner_D.py exited rc=$RC" | tee -a "$RUNNER_LOG"
exit "$RC"
