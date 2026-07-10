"""
task2 condition tcp_scp — direct-transfer control (Issue #1). Same bundle, but
moved point-to-point with scp (direct TCP) instead of rsync/DHT. State-preserving.
checkpoint_mode=tcp. Shows single-event latency is transport-independent:
tcp_scp ≈ app_cold ≈ dht_frl → DHT overhead is trivial vs direct transfer.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from runner_base import run_task2, bundle_trigger, scp_transport  # noqa: E402

CONDITION = "tcp_scp"
CHECKPOINT_MODE = "tcp"


def trigger(cfg, r, robot_id, sr_pre, tc_pre):
    return bundle_trigger(cfg, r, robot_id, sr_pre, transport=scp_transport)


if __name__ == "__main__":
    sys.exit(run_task2(condition=CONDITION, checkpoint_mode=CHECKPOINT_MODE,
                       trigger_fn=trigger))
