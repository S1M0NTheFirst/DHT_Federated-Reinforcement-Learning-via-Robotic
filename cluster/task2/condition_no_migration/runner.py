"""
task2 condition no_migration — the LEARNING-CEILING baseline. Plain federated
SAC with ZERO migrations: robots stay on their home node the whole run, so there
is no transport, no checkpoint, no disruption. It shows the reward trajectory a
migration method would ideally match. checkpoint_mode=none.

Migrations are disabled by launching workers with MIGRATION_ROUNDS="" (set in
run.sh), so the worker never emits a migration_request and `trigger` below is
never called. It exists only to satisfy run_task2's signature.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from runner_base import run_task2  # noqa: E402

CONDITION = "no_migration"
CHECKPOINT_MODE = "none"


def trigger(cfg, r, robot_id, sr_pre, tc_pre):
    # Never invoked (no migration requests are ever issued). Return an empty
    # metrics row defensively so the run can't break if one somehow fires.
    return {"robot_id": robot_id}


if __name__ == "__main__":
    sys.exit(run_task2(condition=CONDITION, checkpoint_mode=CHECKPOINT_MODE,
                       trigger_fn=trigger))
