#!/bin/bash
# Diagnose why /etc/criu/runc.conf isn't being honored.
# Run with: sudo bash diagnose_criu.sh
set +e

echo "=== A. CRIU version + binary path ==="
which criu
/usr/sbin/criu --version
echo ""

echo "=== B. Does CRIU support --enable-external-masters? ==="
/usr/sbin/criu dump --help 2>&1 | grep -i external
echo ""

echo "=== C. What config files does CRIU look for? ==="
/usr/sbin/criu --help 2>&1 | grep -iA1 -E "config|conf"
echo ""

echo "=== D. Does runc bundle its own criu? ==="
runc --version 2>&1
ldd $(which runc) 2>&1 | grep -i criu
echo ""

echo "=== E. Direct criu dump test (bypasses docker, runc) ==="
docker rm -f cudatest 2>/dev/null
docker run -d --name cudatest --gpus all \
    swiftbot-robot:latest \
    python3 -c "import torch,time; x=torch.randn(1000,1000,device='cuda');
                [x@x for _ in iter(int,1)] if False else (lambda: __import__('time').sleep(60))()" \
    2>&1 >/dev/null
sleep 4
PID=$(docker inspect -f '{{.State.Pid}}' cudatest)
echo "  PID: $PID"
/usr/local/bin/cuda-checkpoint --toggle --pid $PID
echo "  CUDA toggled to checkpointed"

mkdir -p /tmp/criu_direct
echo ""
echo "  Running: criu dump --tree $PID --tcp-established --enable-external-masters --shell-job"
/usr/sbin/criu dump \
    --tree $PID \
    --images-dir /tmp/criu_direct \
    --tcp-established \
    --enable-external-masters \
    --shell-job \
    --leave-running 2>&1 | tail -25
RC=$?
echo "  criu dump return code: $RC"
echo ""

if [ $RC -eq 0 ]; then
    echo "  *** DIRECT CRIU DUMP WORKS — issue is just runc not passing the flag ***"
else
    echo "  *** DIRECT CRIU DUMP FAILS TOO — different issue ***"
fi

# Resume CUDA so container keeps working
/usr/local/bin/cuda-checkpoint --toggle --pid $PID 2>/dev/null
docker rm -f cudatest 2>/dev/null

echo ""
echo "=== F. Try config file at every documented path ==="
for f in /etc/criu/criu.conf /etc/criu/default.conf /etc/criu/dump.conf \
         /etc/criu/pre-dump.conf /etc/criu/swrk.conf; do
    cat > $f <<EOF
tcp-established
enable-external-masters
EOF
    echo "  wrote $f"
done
ls -la /etc/criu/
