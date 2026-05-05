#!/bin/bash
#MSUB -N SwiftBot_E_ColdRestart
#MSUB -W group_list=hpc2-coe-users
#MSUB -l walltime=02:00:00
#MSUB -l nodes=n034.cluster.pssclabs.com:ppn=8+n035.cluster.pssclabs.com:ppn=8+n036.cluster.pssclabs.com:ppn=8
#MSUB -j oe

set -uo pipefail
CONDITION="condition_E_cold_restart"
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../common/cluster_config.sh"
source "$HERE/../common/cluster_lib.sh"

setup_run_dirs "$CONDITION"
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
