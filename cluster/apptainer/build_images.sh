#!/bin/bash
# Build the two Apptainer .sif files used by every condition. Run once per
# cluster account, or whenever the .def files change.
#
# Tries `--fakeroot` first; if that's not configured for your account, falls
# back to `--remote` (Sylabs cloud build — free tier, requires `apptainer
# remote login` once). If both fail, asks HPC support to enable user
# namespaces / fakeroot mappings.
#
# Output: cluster/apptainer/robot.sif and cluster/apptainer/baseline.sif

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

build_one() {
    local def="$1"
    local sif="$2"
    if [[ -f "$sif" ]]; then
        echo ">>> $sif already exists, skipping (delete it to force rebuild)"
        return 0
    fi
    echo ">>> Building $sif from $def"
    if apptainer build --fakeroot "$sif" "$def" 2>&1 | tee "${sif%.sif}.build.log"; then
        echo ">>> $sif built via --fakeroot"
        return 0
    fi
    echo "!!! --fakeroot build failed, trying --remote (Sylabs cloud)"
    if apptainer build --remote "$sif" "$def" 2>&1 | tee -a "${sif%.sif}.build.log"; then
        echo ">>> $sif built via --remote"
        return 0
    fi
    echo "!!! Both --fakeroot and --remote failed for $sif."
    echo "    Run 'apptainer remote login' for cloud builds, or ask HPC"
    echo "    support to enable user-namespace fakeroot for your account."
    return 1
}

build_one robot.def    robot.sif
build_one baseline.def baseline.sif

echo
echo "Done. Images:"
ls -lh "$HERE"/*.sif
