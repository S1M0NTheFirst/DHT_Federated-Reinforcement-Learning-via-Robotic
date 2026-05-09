#!/bin/bash
# Build the Apptainer image used by every condition.
#
# Why this script doesn't use `apptainer build`:
#   `apptainer build` runs the .def %post section (apt-get, pip install)
#   inside a builder rootfs that needs root privileges. On most shared HPCs
#   --fakeroot is NOT enabled (no /etc/subuid mapping for users) and Apptainer
#   recently REMOVED --remote support. So a plain user can't build from a
#   .def file at all.
#
# What we do instead:
#   1. `apptainer pull` a pre-built PyTorch+CUDA OCI image from Docker Hub.
#      Pulling needs zero privileges — it just downloads and converts to .sif.
#   2. Run pip from inside that image with `--target=$PYLIBS_DIR` to install
#      the extra Python packages into a host directory. No root, no rebuild,
#      and the dir survives across container restarts.
#   3. The runtime launcher (cluster_runner.launch_robot) bind-mounts that dir
#      and sets PYTHONPATH so `import flwr` etc. work inside the container.
#
# Output:
#   cluster/apptainer/robot.sif      (single shared image)
#   cluster/apptainer/baseline.sif   (symlink → robot.sif)
#   cluster/apptainer/pylibs/        (host-side site-packages, bind-mounted)
#
# Run once per cluster account. Re-run to add/upgrade packages (delete pylibs/
# first to force a clean reinstall).

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

IMG_URI="docker://pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime"
SIF="$HERE/robot.sif"
PYLIBS="$HERE/pylibs"

# --- 1. Pull the prebuilt PyTorch image (no root required) ----------------
if [[ -f "$SIF" ]]; then
    echo ">>> $SIF already exists, skipping pull (delete it to force re-pull)"
else
    echo ">>> Pulling $IMG_URI -> $SIF"
    # APPTAINER_TMPDIR: pulls can be huge (~6 GB unpacked), default /tmp on
    # cluster nodes is often small. Point it at $HOME/.apptainer-tmp.
    export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$HOME/.apptainer-tmp}"
    export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$HOME/.apptainer-cache}"
    mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
    apptainer pull "$SIF" "$IMG_URI"
fi

# baseline.sif is identical to robot.sif (same torch+cuda base; the extra
# python packages live in pylibs/, not in the image). Use a symlink so the
# runner code that references baseline.sif by name keeps working.
if [[ ! -e "$HERE/baseline.sif" ]]; then
    ln -sf robot.sif "$HERE/baseline.sif"
    echo ">>> Symlinked baseline.sif -> robot.sif"
fi

# --- 2. Install extra Python packages into pylibs/ -------------------------
mkdir -p "$PYLIBS"

# Sentinel file: if all packages already installed, skip. Delete to force.
SENTINEL="$PYLIBS/.installed_v1"
if [[ -f "$SENTINEL" ]]; then 
    echo ">>> $PYLIBS already populated (sentinel $SENTINEL), skipping pip install"
else
    echo ">>> Installing extra Python packages into $PYLIBS"
    # Run pip from inside the container so the wheel resolution matches the
    # container's python (3.10) and architecture. --target installs to a
    # plain directory; --no-deps would skip transitive deps so we omit it.
    apptainer exec --bind "$PYLIBS":/pylibs "$SIF" \
        python3 -m pip install --no-cache-dir --target=/pylibs --upgrade \
            "flwr==1.5.0" \
            "kademlia==2.2.3" \
            "stable-baselines3==2.2.1" \
            "gymnasium==0.29.1" \
            "psutil==5.9.8" \
            "pynvml==11.5.0" \
            "redis==5.0.1" \
            "numpy==1.24.4" \
            "pandas==2.0.3" \
            "matplotlib==3.7.2"
    touch "$SENTINEL"
    echo ">>> pylibs install done"
fi

echo
echo "Done. Image and pylibs:"
ls -lh "$HERE"/*.sif
du -sh "$PYLIBS"
echo
echo "NOTE: criu must be installed system-wide on each compute node — it can't"
echo "      be pip-installed. If \`apptainer exec robot.sif which criu\`"
echo "      returns nothing, Conditions C/D will fall back to SIMULATE mode."
