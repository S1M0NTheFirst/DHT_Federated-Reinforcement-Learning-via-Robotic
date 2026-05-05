#!/bin/bash
#MSUB -N SwiftBot_C_CriuCold
#MSUB -W group_list=hpc2-coe-users
#MSUB -l walltime=03:00:00
#MSUB -l nodes=n034.cluster.pssclabs.com:ppn=8+n035.cluster.pssclabs.com:ppn=8+n036.cluster.pssclabs.com:ppn=8
#MSUB -j oe

set -uo pipefail
CONDITION="condition_C_criu_cold"
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../common/cluster_config.sh"
source "$HERE/../common/cluster_lib.sh"

# Probe whether real CRIU is usable on a compute node. If not, force
# SIMULATE so the runner doesn't waste 100 events failing.
detect_criu() {
    if ssh -n -o ConnectTimeout=5 "$CLIENT_NODE_1" "command -v criu >/dev/null 2>&1 && criu check --extra >/dev/null 2>&1"; then
        echo ">>> criu OK on $CLIENT_NODE_1" | tee -a "$RUNNER_LOG"
        export SIMULATE_CRIU=0
    else
        echo ">>> criu NOT usable on $CLIENT_NODE_1 — forcing SIMULATE_CRIU=1" | tee -a "$RUNNER_LOG"
        export SIMULATE_CRIU=1
    fi
}

setup_run_dirs "$CONDITION"
trap 'cleanup_all_nodes' EXIT

pick_alive_nodes || exit 1
start_redis_on_server || exit 1
detect_criu

source "$CONDA_BASE/bin/activate" "$CONDA_ENV"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$CLUSTER_ROOT:${PYTHONPATH:-}"
export SERVER_NODE CLIENT_NODE_1 CLIENT_NODE_2
export REDIS_HOST="$SERVER_NODE"
export RESULTS_DIR="$RESULTS_ROOT/$CONDITION"
export CONDITION="criu_cold"
mkdir -p "$RESULTS_DIR"

echo ">>> Launching runner_C.py (SIMULATE_CRIU=$SIMULATE_CRIU)" | tee -a "$RUNNER_LOG"
python3 -u "$HERE/runner_C.py" 2>&1 | tee -a "$RUNNER_LOG"
RC=${PIPESTATUS[0]}
echo ">>> runner_C.py exited rc=$RC" | tee -a "$RUNNER_LOG"
exit "$RC"
