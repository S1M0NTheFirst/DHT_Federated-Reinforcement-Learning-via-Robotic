#!/bin/bash
# Diagnostic: run ONE online-SAC worker under dmtcp_launch with output to the
# terminal, to see WHERE it stalls (the real dmtcp run produced empty worker
# logs). No Redis/Flower needed — the worker prints its startup + builds the
# MuJoCo env BEFORE trying to connect, so if dmtcp_launch runs it we'll see
# "online SAC start ...". Auto-killed after 45s.
#
# Run on the cluster:  bash cluster/task2/apptainer/diag_dmtcp_worker.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"     # .../cluster
SIF="$ROOT/apptainer/robot.sif"

PROBE='
print("start", flush=True)
import torch; print("torch ok", flush=True)
import numpy; print("numpy", numpy.__version__, flush=True)
import gymnasium as gym; print("gymnasium ok", flush=True)
import mujoco; print("mujoco ok", flush=True)
e = gym.make("Hopper-v4"); print("env made ok", flush=True)
o,_ = e.reset(seed=0); e.step(e.action_space.sample()); print("env step ok", flush=True)
import flwr; print("flwr ok", flush=True)
print("ALL IMPORTS OK", flush=True)
'

run() {  # $1 = label   $2 = extra exports   $3 = launcher prefix (e.g. dmtcp_launch ...)
    echo; echo "========== $1 =========="
    timeout 60 apptainer exec \
        --bind "$ROOT":/cluster_app \
        --bind "$ROOT/apptainer/pylibs":/pylibs \
        --bind "$ROOT/task2/apptainer/pylibs2":/pylibs2 \
        --bind "$HOME/dmtcp":"$HOME/dmtcp" \
        "$SIF" bash -lc "
            export PATH=\$HOME/dmtcp/bin:\$PATH
            export DMTCP_DL_PLUGIN=0 PYTHONUNBUFFERED=1
            export PYTHONPATH=/pylibs2:/pylibs:/cluster_app/task2/worker
            export TASK2_ENV=Hopper-v4
            mkdir -p /tmp/dmtcp_diag
            $2
            $3 python3 -u -c '$PROBE'
        "
    echo ">>> exit $?"
}

# A. baseline: NO dmtcp (proves the imports themselves are fine w/ this PYTHONPATH).
run "A) NO dmtcp (baseline)" "true" ""

# B. dmtcp with single-threaded math libs (the likely fix for the torch hang).
run "B) dmtcp + OMP/MKL=1" \
    "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1" \
    "dmtcp_launch --new-coordinator --coord-port 7851 --ckptdir /tmp/dmtcp_diag"

echo ">>> cleanup: rm -rf /tmp/dmtcp_diag"
