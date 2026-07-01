#!/bin/bash
#MSUB -N SwiftBot_F_Concurrent
#MSUB -W group_list=hpc2-coe-users
#MSUB -l walltime=06:00:00
#MSUB -l nodes=n023.cluster.pssclabs.com:ppn=8+n024.cluster.pssclabs.com:ppn=8+n025.cluster.pssclabs.com:ppn=8
#MSUB -j oe

# Condition F — concurrent-migration stress test.
# Override these when submitting, e.g.:
#   msub -v MECHANISM=cold,MIGRATION_CONCURRENCY=5 condition_F_concurrent/run_F.sh
# Sweep by submitting once per level (1,2,5,10) for each mechanism (cold,warm).

set -uo pipefail
CONDITION_DIR="condition_F_concurrent"
CLUSTER_ROOT="${CLUSTER_ROOT:-$HOME/cluster}"
HERE="$CLUSTER_ROOT/$CONDITION_DIR"
source "$CLUSTER_ROOT/common/cluster_config.sh"
source "$CLUSTER_ROOT/common/cluster_lib.sh"

# Per-condition Redis port so F can run alongside A/C/D/E.
export REDIS_PORT=6383

# Experiment knobs (overridable via msub -v).
export MECHANISM="${MECHANISM:-cold}"
export MIGRATION_CONCURRENCY="${MIGRATION_CONCURRENCY:-5}"
# Synchronized waves: all robots migrate at the same task counters so a wave
# is large enough to fill a batch of MIGRATION_CONCURRENCY.
export MIGRATION_OFFSET=0

# Unique condition tag per (mechanism, level) so results/checkpoints don't mix.
export CONDITION="concurrent_${MECHANISM}_c${MIGRATION_CONCURRENCY}"

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

echo ">>> Launching runner_F.py (mechanism=$MECHANISM concurrency=$MIGRATION_CONCURRENCY)" \
    | tee -a "$RUNNER_LOG"
python3 -u "$HERE/runner_F.py" 2>&1 | tee -a "$RUNNER_LOG"
RC=${PIPESTATUS[0]}
echo ">>> runner_F.py exited rc=$RC" | tee -a "$RUNNER_LOG"
exit "$RC"
