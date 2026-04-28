#!/usr/bin/env bash
# Build CRIU 4.0 with the CUDA plugin and install side-by-side with the
# distro CRIU (3.16.1 from apt).
#
# Result: /usr/local/sbin/criu   (criu 4.0)
#         /usr/local/lib/criu/cuda_plugin.so
#
# After this finishes, the runner will pick the new criu via
# CRIU_BIN=/usr/local/sbin/criu and load the cuda plugin via
# --plugins=cuda.
#
# References:
#   https://criu.org/CUDA_support
#   https://github.com/checkpoint-restore/criu
#   https://github.com/NVIDIA/cuda-checkpoint
#
# Run on the host (not inside a container):
#   bash install_criu_cuda.sh
set -euo pipefail

CRIU_VERSION="v4.0"
WORK=/tmp/criu_build
PREFIX=/usr/local

echo "==> Installing build deps"
sudo apt-get update
sudo apt-get install -y \
    build-essential pkg-config \
    libprotobuf-dev libprotobuf-c-dev protobuf-c-compiler protobuf-compiler \
    libnl-3-dev libnet-dev libcap-dev libbsd-dev \
    libgnutls28-dev libnftables-dev libdrm-dev \
    asciidoc xmlto python3-dev python3-protobuf python3-pip \
    libuuid1 uuid-dev \
    git curl

echo "==> Cloning CRIU $CRIU_VERSION"
rm -rf "$WORK"
git clone --depth=1 --branch "$CRIU_VERSION" \
    https://github.com/checkpoint-restore/criu.git "$WORK"

cd "$WORK"

echo "==> Building CRIU"
make -j"$(nproc)"

echo "==> Building CUDA plugin"
# The cuda plugin needs libcuda.so (from the NVIDIA driver, already on the
# system since nvidia-smi works) and the cuda-checkpoint binary
# (/usr/local/bin/cuda-checkpoint).
if [ ! -d plugins/cuda ]; then
    echo "ERROR: plugins/cuda not found in CRIU $CRIU_VERSION source. " \
         "The CUDA plugin lives in this directory in 4.0+; check the tag."
    exit 1
fi
make -C plugins/cuda

echo "==> Installing"
sudo make install PREFIX="$PREFIX"
sudo make -C plugins/cuda install PREFIX="$PREFIX"

# Ensure the plugin is in CRIU's plugin search path
sudo mkdir -p "$PREFIX/lib/criu"
sudo find "$WORK/plugins/cuda" -name "*.so" -exec cp {} "$PREFIX/lib/criu/" \;

echo "==> Verifying"
"$PREFIX/sbin/criu" --version
ls -la "$PREFIX/lib/criu/" | grep -i cuda || echo "WARN: no cuda plugin found in $PREFIX/lib/criu/"

echo
echo "==> Done. Use the new CRIU by setting:"
echo "      export CRIU_BIN=$PREFIX/sbin/criu"
echo "    in the terminal where you launch dht_frl_runner.py."
