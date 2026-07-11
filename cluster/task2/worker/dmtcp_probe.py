"""
Standalone torch process that DMTCP checkpoints for the `dmtcp` condition's
HEAVY-image measurement. It reproduces the worker's memory footprint (torch +
the SAC policy/critics/optimizer + replay buffer loaded from the migration
bundle) but holds NO gRPC connection — so DMTCP checkpoints it cleanly (unlike
the live FL worker, whose gRPC connection DMTCP jams).

Launched under dmtcp_launch by the dmtcp trigger; it loads the bundle, writes a
`--ready` marker, then idles so the trigger can `dmtcp_command --bcheckpoint` it
into a real ~1.9 GB full-process image. This is a genuine DMTCP full-process
checkpoint of an equivalent RL worker — the heavy baseline DHT-FRL beats.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, "/cluster_app/task2/worker")
import torch  # noqa: E402
from sac import SAC  # noqa: E402
from replay_buffer import ReplayBuffer  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)   # /checkpoints/<robot_id>
    p.add_argument("--ready", required=True)    # marker file to touch when loaded
    args = p.parse_args()

    # Rebuild the same in-memory state the live worker holds.
    sac = SAC()
    sac_path = os.path.join(args.bundle, "sac_state.pt")
    if os.path.exists(sac_path):
        sac.load_state_dict(torch.load(sac_path, map_location="cpu",
                                       weights_only=False))
    rb = ReplayBuffer(11, 3, capacity=100000)
    rb_path = os.path.join(args.bundle, "replay_buffer.pkl")
    if os.path.exists(rb_path):
        import pickle
        with open(rb_path, "rb") as f:
            rb.load_state_dict(pickle.load(f))

    # Touch a couple of tensors so torch's allocator/threads are warm (matches
    # the resident footprint of a running worker).
    with torch.no_grad():
        _ = sac.actor(torch.zeros(4, 11))

    with open(args.ready, "w") as f:
        f.write("ready")
    print(f"PROBE READY bundle={args.bundle} replay={len(rb)}", flush=True)

    # Idle so the coordinator can checkpoint us, then get --quit'd by the trigger.
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
