"""
Condition F — concurrent-migration stress test.

Not a new mechanism: F re-uses an existing checkpoint mechanism (cold = C, or
warm = D) and fires up to MIGRATION_CONCURRENCY migrations SIMULTANEOUSLY, to
measure how per-migration downtime and recovery degrade as the number of
concurrent migrations grows. The argument: checkpoint+rsync mechanisms
serialize on disk/network under concurrency, while DHT bundle transfer
(Condition A, run separately with MIGRATION_CONCURRENCY) stays flat.

Run waves of a chosen size by setting (in run_F.sh):
    MECHANISM=cold|warm          which baseline mechanism to stress
    MIGRATION_CONCURRENCY=N       migrations fired per wave (1,2,5,10)
    MIGRATION_OFFSET=0            so all robots migrate at the same task
                                  counters -> waves large enough to fill N

Each event's CSV row carries concurrency_level = the wave size it was part of.
"""
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.baseline_runner_base import run_baseline  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
LOG = logging.getLogger("condF")

IMAGE             = "baseline.sif"
WORKER            = "/cluster_app/workers/worker_app_checkpoint.py"
PRETRAINED_INSIDE = "/cluster_app/common/pretrained_policy.pt"
CKPT_INSIDE       = "/checkpoints/state.pt"
WARM_CKPT_INSIDE  = "/checkpoints/warm/state.pt"
WARM_INTERVAL     = 50


def main() -> int:
    mechanism   = os.environ.get("MECHANISM", "cold").lower()
    concurrency = int(os.environ.get("MIGRATION_CONCURRENCY", "5"))
    LOG.info("Condition F — mechanism=%s concurrency=%d", mechanism, concurrency)

    if mechanism == "cold":
        from condition_C_criu_cold.runner_C import trigger_app_cold as trigger
        condition = f"concurrent_cold_c{concurrency}"
        initial_env = {
            "APP_CHECKPOINT_PATH":    CKPT_INSIDE,
            "WORKER_PRETRAINED_PATH": PRETRAINED_INSIDE,
        }
        pre_loop = None
    elif mechanism == "warm":
        from condition_D_criu_warm.runner_D import (
            trigger_app_warm as trigger, _start_precopy, _precopy_stop,
        )
        condition = f"concurrent_warm_c{concurrency}"
        initial_env = {
            "APP_CHECKPOINT_PATH":      CKPT_INSIDE,
            "WARM_CHECKPOINT_PATH":     WARM_CKPT_INSIDE,
            "WARM_CHECKPOINT_INTERVAL": WARM_INTERVAL,
            "WORKER_PRETRAINED_PATH":   PRETRAINED_INSIDE,
        }
        pre_loop = _start_precopy
    else:
        LOG.error("Unknown MECHANISM=%r (expected cold|warm). For DHT "
                  "concurrency run Condition A with MIGRATION_CONCURRENCY set.",
                  mechanism)
        return 2

    rc = run_baseline(
        condition=condition,
        image=IMAGE,
        worker_script=WORKER,
        trigger_fn=trigger,
        initial_extra_env=initial_env,
        pre_loop=pre_loop,
        concurrency=concurrency,
    )
    if mechanism == "warm":
        from condition_D_criu_warm.runner_D import _precopy_stop
        _precopy_stop.set()
    return rc


if __name__ == "__main__":
    sys.exit(main())
