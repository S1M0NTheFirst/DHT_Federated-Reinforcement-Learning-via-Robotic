#!/bin/bash
# Build DMTCP from source into $HOME (userspace, NO root) on the HOST login node.
# The container is a minimal PyTorch runtime with NO C++ compiler, so we build
# with the host's gcc/g++ instead. DMTCP is pure userspace and its binaries are
# forward-compatible: host glibc (2.28, RHEL8) is older than the container's, so
# host-built binaries run fine inside the container at runtime (verified in the
# Step-2 test). DMTCP is why task2's condition #5 is a REAL heavy checkpoint tool
# (retires task1's "faked CRIU" criticism). CPU-only — task2 forces CPU.
#
# Run once on the cluster:
#   bash cluster/task2/apptainer/build_dmtcp.sh
# Produces:  $HOME/dmtcp/bin/{dmtcp_launch,dmtcp_command,dmtcp_restart,dmtcp_coordinator}
# Delete $HOME/dmtcp to force a clean rebuild.

set -euo pipefail
PREFIX="$HOME/dmtcp"
VERSION="${DMTCP_VERSION:-3.0.0}"
SRC="/tmp/dmtcp-src-$$"

if [[ -x "$PREFIX/bin/dmtcp_launch" ]]; then
    echo ">>> DMTCP already built at $PREFIX (delete it to rebuild). Version:"
    "$PREFIX/bin/dmtcp_launch" --version || true
    exit 0
fi

# Need a host C++ compiler. If missing, try `module load gcc` first.
if ! command -v g++ >/dev/null 2>&1; then
    echo "ERROR: no g++ on the host. Try:  module load gcc   (then re-run)." >&2
    echo "       Available gcc modules:" >&2
    module avail gcc 2>&1 | sed 's/^/         /' >&2 || true
    exit 1
fi
echo ">>> Using host compiler: $(g++ --version | head -1)"

echo ">>> Building DMTCP $VERSION into $PREFIX (host, no root)"
rm -rf "$SRC" && mkdir -p "$SRC" && cd "$SRC"
url="https://github.com/dmtcp/dmtcp/archive/refs/tags/${VERSION}.tar.gz"
echo ">>> downloading $url"
if command -v curl >/dev/null 2>&1; then
    curl -fL -o dmtcp.tar.gz "$url"
elif command -v wget >/dev/null 2>&1; then
    wget -O dmtcp.tar.gz "$url"
else
    python3 -c "import urllib.request,sys; urllib.request.urlretrieve(sys.argv[1],'dmtcp.tar.gz')" "$url"
fi
tar xf dmtcp.tar.gz
cd "dmtcp-${VERSION}"
./configure --prefix="$PREFIX"
make -j"$(nproc)"
make install
cd /tmp && rm -rf "$SRC"

echo
echo ">>> Verifying DMTCP binaries (host):"
"$PREFIX/bin/dmtcp_launch" --version
echo ">>> DMTCP installed at $PREFIX/bin"
echo "    (Step 2 will verify these run INSIDE the container.)"
