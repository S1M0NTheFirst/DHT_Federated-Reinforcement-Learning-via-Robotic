#!/bin/bash
# Build DMTCP from source into $HOME (userspace, NO root) INSIDE the container,
# so its glibc matches the runtime. DMTCP is pure userspace — unlike CRIU it
# needs no root to install or run — which is exactly why task2's condition #5
# can use it as a REAL heavy checkpoint tool (retires task1's "faked CRIU"
# criticism). CPU-only: DMTCP cannot dump a live CUDA process, and task2 forces
# CPU, so that's fine.
#
# Run once on the cluster:
#   bash cluster/task2/apptainer/build_dmtcp.sh
# Produces:  $HOME/dmtcp/bin/{dmtcp_launch,dmtcp_command,dmtcp_restart,dmtcp_coordinator}
# Delete $HOME/dmtcp to force a clean rebuild.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SIF="$HERE/../../apptainer/robot.sif"
PREFIX="$HOME/dmtcp"
VERSION="${DMTCP_VERSION:-3.0.0}"
SRC="/tmp/dmtcp-src-$$"

if [[ -x "$PREFIX/bin/dmtcp_launch" ]]; then
    echo ">>> DMTCP already built at $PREFIX (delete it to rebuild). Version:"
    "$PREFIX/bin/dmtcp_launch" --version || true
    exit 0
fi

echo ">>> Building DMTCP $VERSION into $PREFIX (inside container, no root)"
apptainer exec "$SIF" bash -lc "
    set -euo pipefail
    rm -rf '$SRC' && mkdir -p '$SRC' && cd '$SRC'
    url='https://github.com/dmtcp/dmtcp/archive/refs/tags/$VERSION.tar.gz'
    echo '>>> downloading '\$url
    curl -fL -o dmtcp.tar.gz \"\$url\"
    tar xf dmtcp.tar.gz
    cd dmtcp-$VERSION
    ./configure --prefix='$PREFIX'
    make -j\$(nproc)
    make install
    rm -rf '$SRC'
"

echo
echo ">>> Verifying DMTCP binaries:"
apptainer exec "$SIF" bash -lc "'$PREFIX'/bin/dmtcp_launch --version"
echo ">>> DMTCP installed at $PREFIX/bin"
echo "    Add to PATH in runs:  export PATH=$PREFIX/bin:\$PATH"
