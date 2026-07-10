"""
task2 condition app_cold — light application-checkpoint baseline. Same unified
bundle (sac_state.pt + replay_buffer.pkl + manifest.json) as dht_frl, but
transported by a plain stop-world rsync (no DHT, no pre-copy). State-preserving.
checkpoint_mode=app. For Issue #1: app_cold ≈ dht_frl on latency → the bundle
win is transport-independent.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from runner_base import run_task2, bundle_trigger, rsync_transport  # noqa: E402

CONDITION = "app_cold"
CHECKPOINT_MODE = "app"


def trigger(cfg, r, robot_id, sr_pre, tc_pre):
    return bundle_trigger(cfg, r, robot_id, sr_pre, transport=rsync_transport)


if __name__ == "__main__":
    sys.exit(run_task2(condition=CONDITION, checkpoint_mode=CHECKPOINT_MODE,
                       trigger_fn=trigger))
