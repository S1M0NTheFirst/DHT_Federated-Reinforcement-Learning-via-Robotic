#!/bin/bash
# Build the EXTRA python layer task2 needs, WITHOUT touching task1's frozen
# image or pylibs. We can't `apptainer build` here (no --fakeroot on this HPC),
# so we reuse task1's prebuilt robot.sif and pip-install task2's additions into
# an ISOLATED dir (pylibs2) that gets bind-mounted alongside task1's pylibs.
#
# task2 additions over task1:
#   - mujoco            (pure-pip, bundles its own binaries, NO root needed)
#   - gymnasium[mujoco] extras
# torch/flwr/gymnasium/psutil/redis/numpy already live in task1's pylibs.
#
# Run once on the cluster (login node is fine — pip --target just downloads):
#   bash cluster/task2/apptainer/build_task2_env.sh
# Delete pylibs2/.installed_v1 to force a clean reinstall.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SIF="$HERE/../../apptainer/robot.sif"      # reuse task1's image
PYLIBS2="$HERE/pylibs2"

if [[ ! -f "$SIF" ]]; then
    echo "ERROR: $SIF not found. Build task1's image first "
    echo "       (cluster/apptainer/build_images.sh)." >&2
    exit 1
fi

mkdir -p "$PYLIBS2"
SENTINEL="$PYLIBS2/.installed_v1"
if [[ -f "$SENTINEL" ]]; then
    echo ">>> $PYLIBS2 already populated (sentinel present); skipping."
else
    echo ">>> Installing MuJoCo layer into $PYLIBS2 (via container python)"
    apptainer exec --bind "$PYLIBS2":/pylibs2 "$SIF" \
        python3 -m pip install --no-cache-dir --target=/pylibs2 --upgrade \
            "mujoco==3.1.6" \
            "gymnasium[mujoco]==0.29.1"
    touch "$SENTINEL"
    echo ">>> pylibs2 install done"
fi

echo
echo ">>> Verifying Hopper-v4 runs in the image (CPU)..."
apptainer exec \
    --bind "$HERE/../../apptainer/pylibs":/pylibs \
    --bind "$PYLIBS2":/pylibs2 \
    --env PYTHONPATH=/pylibs2:/pylibs \
    "$SIF" python3 - <<'PY'
import gymnasium as gym
e = gym.make("Hopper-v4")
o, _ = e.reset(seed=0)
for _ in range(5):
    o, r, term, trunc, _ = e.step(e.action_space.sample())
print("Hopper-v4 OK  obs_dim=%d act_dim=%d  sample_reward=%.3f"
      % (e.observation_space.shape[0], e.action_space.shape[0], r))
PY
echo ">>> If you saw 'Hopper-v4 OK', the online-RL dependency is satisfied."
