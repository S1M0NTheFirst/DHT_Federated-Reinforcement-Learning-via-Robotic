#!/bin/bash
# Step 2: verify the host-built DMTCP runs INSIDE the container and can
# checkpoint + restore a live python+torch CPU process (proves glibc compat and
# that DMTCP works before we wire it into the dmtcp condition).
#
# Run on the cluster:  bash cluster/task2/apptainer/test_dmtcp.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SIF="$HERE/../../apptainer/robot.sif"
DMTCP="$HOME/dmtcp"
WORK="/tmp/dmtcp_test_$$"
mkdir -p "$WORK"

# A tiny long-running torch process that prints a counter every second.
cat > "$WORK/counter.py" <<'PY'
import time, torch
x = torch.zeros(1)
i = 0
while True:
    x += 1; i += 1
    print(f"counter={i} tensor={x.item():.0f}", flush=True)
    time.sleep(1)
PY

apptainer exec --bind "$DMTCP":"$DMTCP" --bind "$WORK":"$WORK" \
    "$SIF" bash -lc "
    export PATH='$DMTCP/bin':\"\$PATH\"   # prepend, keep the image's conda PATH
    export DMTCP_DL_PLUGIN=0              # silence harmless CPU-only libcuda dlopen warnings
    cd '$WORK'
    echo '>>> which python3:'; command -v python3
    echo '>>> dmtcp_launch --version:'; dmtcp_launch --version | head -1
    echo '>>> launching counter under DMTCP (self-managed coordinator, port 7779)'
    dmtcp_launch --new-coordinator --coord-port 7779 --ckptdir '$WORK' \
        python3 counter.py > run.log 2>&1 &
    sleep 12
    echo '>>> counter before checkpoint:'; grep counter= run.log | tail -2
    echo '>>> checkpointing...'
    dmtcp_command --coord-port 7779 --bcheckpoint    # blocking: waits until ckpt done
    echo '>>> checkpoint image(s):'
    ls -lh '$WORK'/ckpt_*.dmtcp 2>/dev/null | awk '{print \"   \", \$5, \$9}'
    echo '>>> stopping original (coordinator --quit kills launched peer):'
    dmtcp_command --coord-port 7779 --quit 2>/dev/null || true
    pkill -f counter.py 2>/dev/null || true
    sleep 2
    echo '>>> restarting from checkpoint image:'
    setsid bash '$WORK'/dmtcp_restart_script.sh > restart.log 2>&1 &
    sleep 10
    echo '>>> counter AFTER restore (should CONTINUE from ~12, NOT reset to 1):'
    grep counter= restart.log | tail -3
    echo '>>> tearing down all test processes'
    '$DMTCP'/bin/dmtcp_command --coord-port 7779 --quit 2>/dev/null || true
    pkill -f counter.py 2>/dev/null || true
    pkill -f dmtcp_coordinator 2>/dev/null || true
    sleep 1
" || true
echo ">>> test dir: $WORK (rm -rf when done)"
