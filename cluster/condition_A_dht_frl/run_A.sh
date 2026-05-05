#!/bin/bash
#MSUB -N SwiftBot_A_DHT_FRL
#MSUB -W group_list=hpc2-coe-users
#MSUB -l walltime=06:00:00
#MSUB -l nodes=n034.cluster.pssclabs.com:ppn=8+n035.cluster.pssclabs.com:ppn=8+n036.cluster.pssclabs.com:ppn=8
#MSUB -j oe

# Condition A — DHT+FRL.
# Requests 3 GPU nodes (RTX 3090 by default; swap to n021/n022 for P100):
#   n034 → server (Redis + Flower + runner)
#   n035 → client #1 (robots 0..9)
#   n036 → client #2 (robots 10..19)

set -uo pipefail

CONDITION="condition_A_dht_frl"
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../common/cluster_config.sh"
source "$HERE/../common/cluster_lib.sh"

setup_run_dirs "$CONDITION"
trap 'cleanup_all_nodes' EXIT

pick_alive_nodes || exit 1
start_redis_on_server || exit 1

# Activate conda for the runner host (this script runs on the head/master node).
source "$CONDA_BASE/bin/activate" "$CONDA_ENV"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$CLUSTER_ROOT:${PYTHONPATH:-}"

export SERVER_NODE CLIENT_NODE_1 CLIENT_NODE_2
export REDIS_HOST="$SERVER_NODE"
export RESULTS_DIR="$RESULTS_ROOT/$CONDITION"
export CONDITION="dht_frl"
mkdir -p "$RESULTS_DIR"

echo ">>> Launching runner_A.py" | tee -a "$RUNNER_LOG"
python3 -u "$HERE/runner_A.py" 2>&1 | tee -a "$RUNNER_LOG"
RC=${PIPESTATUS[0]}
echo ">>> runner_A.py exited rc=$RC" | tee -a "$RUNNER_LOG"
exit "$RC"
