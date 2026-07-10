"""
task2 condition app_warm — pre-copy variant of app_cold. Same bundle, but the
bulk is rsync'd in a background pre-copy and only a final delta is synced at
migration time, so downtime is lower than app_cold at the cost of background
bandwidth. State-preserving. checkpoint_mode=app_warm.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from runner_base import run_task2, bundle_trigger, warm_transport  # noqa: E402

CONDITION = "app_warm"
CHECKPOINT_MODE = "app_warm"


def trigger(cfg, r, robot_id, sr_pre, tc_pre):
    return bundle_trigger(cfg, r, robot_id, sr_pre, transport=warm_transport)


if __name__ == "__main__":
    sys.exit(run_task2(condition=CONDITION, checkpoint_mode=CHECKPOINT_MODE,
                       trigger_fn=trigger))
