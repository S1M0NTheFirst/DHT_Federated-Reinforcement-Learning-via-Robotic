#!/bin/bash
# Verify direct `criu dump --leave-running` works on a CUDA container.
# Run with: sudo bash test_direct_criu.sh
set -e

echo "=== 1. Spin up a CUDA container ==="
docker rm -f cudatest 2>/dev/null || true

# Write the workload to a temp file and bind-mount it so python3 can be the
# container's PID 1 directly (no shell wrapper). cuda-checkpoint operates on
# the PID it's given — if PID 1 is bash, it has no CUDA contexts and reports
# "initialization error". The actual experiment workers run python3 as PID 1
# already, so this matches their behaviour.
cat > /tmp/cuda_workload.py <<'PYEOF'
import torch, time
x = torch.randn(2000, 2000, device="cuda")
y = torch.randn(2000, 2000, device="cuda")
print("CUDA tensors allocated", flush=True)
while True:
    z = x @ y
    torch.cuda.synchronize()
    time.sleep(0.5)
PYEOF

docker run -d --name cudatest --gpus all --security-opt seccomp:unconfined \
    -v /tmp/cuda_workload.py:/workload.py:ro \
    swiftbot-robot:latest \
    python3 /workload.py
sleep 6   # extra time for torch + CUDA init
docker logs cudatest 2>&1 | tail -3
PID=$(docker inspect -f '{{.State.Pid}}' cudatest)
echo "  container PID = $PID"
[ "$PID" -gt 0 ] || { echo "container did not start"; docker logs cudatest; exit 1; }

# Sanity: confirm PID is actually python3
COMM=$(cat /proc/$PID/comm 2>/dev/null)
echo "  /proc/$PID/comm = $COMM (expect python3)"

echo ""
echo "=== 2. Toggle CUDA OFF ==="
/usr/local/bin/cuda-checkpoint --toggle --pid $PID
echo "  state: $(/usr/local/bin/cuda-checkpoint --get-state --pid $PID)"

echo ""
echo "=== 3. THE REAL TEST — direct criu dump --leave-running ==="

echo "  Discovering external mounts via /proc/$PID/mountinfo..."
# CRIU refuses to dump mounts it considers "unreachable sharing" or that
# have "no proper root mount". This happens for:
# 1. NVIDIA bind mounts (which have a master:N propagation field).
# 2. Docker bind mounts of individual files/dirs (like /etc/hosts) where the
#    mount root (field 4) is not "/".
NV_MOUNTS=$(awk '
    {
        has_master = 0
        for (i = 7; i <= NF; i++) {
            if ($i == "-") break
            if ($i ~ /^master:/) { has_master = 1; break }
        }
        
        root = $4
        mp = $5
        
        if (has_master || (root != "/" && mp !~ /^\/(proc|sys|dev)/)) {
            print mp
        }
    }
' /proc/$PID/mountinfo | sort -u)
echo "$NV_MOUNTS" | sed 's/^/    /'
echo "  count: $(echo "$NV_MOUNTS" | wc -l)"

# Build --external args, one per nvidia mount (CRIU 3.16.1 has no
# --enable-external-masters CLI flag — use --external mnt[X]:label instead).
EXT_ARGS=()
i=0
while IFS= read -r m; do
    [ -z "$m" ] && continue
    EXT_ARGS+=(--external "mnt[$m]:nv$i")
    i=$((i+1))
done <<< "$NV_MOUNTS"
echo "  --external args: ${EXT_ARGS[@]}"

rm -rf /tmp/direct_chk && mkdir -p /tmp/direct_chk
T0=$(date +%s.%N)
if /usr/sbin/criu dump \
    --tree $PID \
    --images-dir /tmp/direct_chk \
    --tcp-established \
    --shell-job \
    --ext-unix-sk \
    --manage-cgroups=soft \
    --leave-running \
    "${EXT_ARGS[@]}" 2>/tmp/direct_chk_err; then
    T1=$(date +%s.%N)
    DT=$(echo "$T1 - $T0" | bc)
    SIZE=$(du -sh /tmp/direct_chk | cut -f1)
    echo "  *** SUCCESS — dump took ${DT}s, size $SIZE ***"
    echo "  Files:"
    ls /tmp/direct_chk | head -10
else
    echo "  *** FAILED — last 30 lines of criu output: ***"
    tail -30 /tmp/direct_chk_err
fi

echo ""
echo "=== 4. Toggle CUDA back ON ==="
/usr/local/bin/cuda-checkpoint --toggle --pid $PID
echo "  state: $(/usr/local/bin/cuda-checkpoint --get-state --pid $PID)"

echo ""
echo "=== 5. Container still alive? ==="
docker ps --filter name=cudatest --format '{{.Names}}\t{{.Status}}'
docker logs cudatest 2>&1 | tail -3

echo ""
echo "=== 6. Cleanup ==="
docker rm -f cudatest 2>/dev/null || true
echo "  done."
