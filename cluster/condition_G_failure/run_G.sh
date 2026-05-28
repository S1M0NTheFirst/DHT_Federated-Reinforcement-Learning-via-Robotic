#!/bin/bash
#MSUB -N SwiftBot_G_Failure
#MSUB -W group_list=hpc2-coe-users
#MSUB -l walltime=06:00:00
#MSUB -l nodes=n030.cluster.pssclabs.com:ppn=8+n031.cluster.pssclabs.com:ppn=8+n032.cluster.pssclabs.com:ppn=8
#MSUB -j oe

# Condition G — failure injection (destination killed mid-migration).
# Override the fault timing with: msub -v FAULT_DELAY_MS=500 condition_G_failure/run_G.sh

set -uo pipefail
CONDITION_DIR="condition_G_failure"
CLUSTER_ROOT="${CLUSTER_ROOT:-$HOME/cluster}"
HERE="$CLUSTER_ROOT/$CONDITION_DIR"
source "$CLUSTER_ROOT/common/cluster_config.sh"
source "$CLUSTER_ROOT/common/cluster_lib.sh"

# Per-condition Redis port so G can run alongside A/C/D/E/F.
export REDIS_PORT=6384
export FAULT_DELAY_MS="${FAULT_DELAY_MS:-500}"
export CONDITION="failure_injection"

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
mkdir -p "$RESULTS_DIR"

echo ">>> Launching runner_G.py (fault_delay_ms=$FAULT_DELAY_MS)" | tee -a "$RUNNER_LOG"
python3 -u "$HERE/runner_G.py" 2>&1 | tee -a "$RUNNER_LOG"
RC=${PIPESTATUS[0]}
echo ">>> runner_G.py exited rc=$RC" | tee -a "$RUNNER_LOG"
exit "$RC"
