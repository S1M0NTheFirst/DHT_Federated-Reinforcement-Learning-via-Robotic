#!/bin/bash
#MSUB -N SwiftBot_C_CriuCold
#MSUB -W group_list=hpc2-coe-users
#MSUB -l walltime=05:00:00
#MSUB -l nodes=n020.cluster.pssclabs.com:ppn=8+n027.cluster.pssclabs.com:ppn=8+n033.cluster.pssclabs.com:ppn=8
#MSUB -j oe

set -uo pipefail
CONDITION="condition_C_criu_cold"
CLUSTER_ROOT="${CLUSTER_ROOT:-$HOME/cluster}"
HERE="$CLUSTER_ROOT/$CONDITION"
source "$CLUSTER_ROOT/common/cluster_config.sh"
source "$CLUSTER_ROOT/common/cluster_lib.sh"

# Condition C uses application-level checkpointing (torch.save/load), not
# kernel CRIU — see cluster/README.md "Known limitations". No criu probe
# needed.

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
export CONDITION="criu_cold"   # disk path component (kept for git continuity)
mkdir -p "$RESULTS_DIR"

echo ">>> Launching runner_C.py (app-level checkpointing)" | tee -a "$RUNNER_LOG"
python3 -u "$HERE/runner_C.py" 2>&1 | tee -a "$RUNNER_LOG"
RC=${PIPESTATUS[0]}
echo ">>> runner_C.py exited rc=$RC" | tee -a "$RUNNER_LOG"
exit "$RC"
