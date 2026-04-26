#!/bin/bash
# End-to-end test: does cuda-checkpoint + docker checkpoint actually work?
# Run with: sudo bash test_cuda_criu.sh
# (sudo because cuda-checkpoint needs root to ptrace + talk to NVIDIA driver)

set -e

echo "=== 1. Install cuda-checkpoint to /usr/local/bin ==="
if [ ! -x /usr/local/bin/cuda-checkpoint ]; then
    install -m 755 /tmp/cuda-checkpoint /usr/local/bin/cuda-checkpoint
fi
/usr/local/bin/cuda-checkpoint --help >/dev/null && echo "  OK"

echo ""
echo "=== 2. Configure CRIU options ==="
# tcp-established       — allow established TCP (Flower gRPC connections)
# enable-external-masters — allow mounts whose master lives outside the
#                           container ns (nvidia-container-runtime bind
#                           mounts /proc/driver/nvidia/* with shared
#                           propagation that CRIU otherwise refuses).
mkdir -p /etc/criu
cat > /etc/criu/runc.conf <<'EOF'
tcp-established
enable-external-masters
EOF
cp /etc/criu/runc.conf /etc/criu/criu.conf
echo "  /etc/criu/runc.conf:"
sed 's/^/    /' /etc/criu/runc.conf

echo ""
echo "=== 3. Start a CUDA-using test container ==="
docker rm -f cudatest 2>/dev/null || true
docker run -d --name cudatest --gpus all --security-opt seccomp:unconfined \
  swiftbot-robot:latest \
  python3 -c "
import torch, time
x = torch.randn(2000, 2000, device='cuda')
y = torch.randn(2000, 2000, device='cuda')
print('CUDA tensors allocated', flush=True)
while True:
    z = x @ y
    torch.cuda.synchronize()
    time.sleep(0.5)
"
sleep 5
PID=$(docker inspect -f '{{.State.Pid}}' cudatest)
echo "  container PID = $PID"

echo ""
echo "=== 4. State BEFORE toggle (should be 'running') ==="
/usr/local/bin/cuda-checkpoint --get-state --pid $PID

echo ""
echo "=== 5. Toggle CUDA OFF ==="
/usr/local/bin/cuda-checkpoint --toggle --pid $PID
echo "  state after toggle:"
/usr/local/bin/cuda-checkpoint --get-state --pid $PID

echo ""
echo "=== 6. THE REAL TEST — docker checkpoint create ==="
rm -rf /tmp/test_chk
mkdir -p /tmp/test_chk
if docker checkpoint create --checkpoint-dir=/tmp/test_chk cudatest test1 2>&1; then
    echo "  *** docker checkpoint create SUCCEEDED ***"
    echo "  Checkpoint files:"
    find /tmp/test_chk -type f | head -10
    SIZE=$(du -sh /tmp/test_chk | cut -f1)
    echo "  Total size: $SIZE"
else
    echo "  *** docker checkpoint create FAILED ***"
    echo "  Reading criu-dump.log for actual cause:"
    LOG=$(find /run/containerd/io.containerd.runtime.v2.task/moby -name 'criu-dump.log' -newer /tmp/cuda-checkpoint 2>/dev/null | head -1)
    if [ -n "$LOG" ]; then
        echo "  ---- $LOG ----"
        tail -40 "$LOG"
        echo "  ---- end ----"
    else
        echo "  (no criu-dump.log found — may be elsewhere)"
        find / -name 'criu-dump.log' 2>/dev/null | head -3
    fi
fi

echo ""
echo "=== 7. Cleanup ==="
docker rm -f cudatest 2>/dev/null || true
echo "  Done."
