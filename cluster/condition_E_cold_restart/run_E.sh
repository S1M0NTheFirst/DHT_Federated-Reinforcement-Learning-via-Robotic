#!/bin/bash
#MSUB -N SwiftBot_E_ColdRestart
#MSUB -W group_list=hpc2-coe-users
#MSUB -l walltime=05:00:00
# Pinned nodes. SINGLETASK is not honored on this MOAB install, so generic
# "nodes=3:ppn=8" consolidates onto one host. n034 currently rejects SSH
# from outside an active job (head→n034 fails), so we use n035 + n036
# (RTX 3090) + n021 (Tesla P100) instead. Swap n021 for n022 if it queues.
#MSUB -l nodes=n023.cluster.pssclabs.com:ppn=8+n024.cluster.pssclabs.com:ppn=8+n030.cluster.pssclabs.com:ppn=8
#MSUB -j oe

set -uo pipefail
CONDITION="condition_E_cold_restart"
# Torque/MOAB copies this script to /var/spool/torque/mom_priv/jobs/ before
# running it, so $0 / $(dirname $0) is NOT the original location. Use a
# fixed path under $HOME instead.
CLUSTER_ROOT="${CLUSTER_ROOT:-$HOME/cluster}"
HERE="$CLUSTER_ROOT/$CONDITION"
source "$CLUSTER_ROOT/common/cluster_config.sh"
source "$CLUSTER_ROOT/common/cluster_lib.sh"

# Per-condition Redis port so A/C/D/E can run in parallel without colliding.
export REDIS_PORT=6382

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
export CONDITION="cold_restart"
mkdir -p "$RESULTS_DIR"

echo ">>> Launching runner_E.py" | tee -a "$RUNNER_LOG"
python3 -u "$HERE/runner_E.py" 2>&1 | tee -a "$RUNNER_LOG"
RC=${PIPESTATUS[0]}
echo ">>> runner_E.py exited rc=$RC" | tee -a "$RUNNER_LOG"
exit "$RC"
