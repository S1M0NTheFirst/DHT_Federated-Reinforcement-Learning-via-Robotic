"""
Single-robot ONLINE smoke-test GATE — PLAN pre-flight. Run this INSIDE the
apptainer image (CPU) BEFORE any cluster submission. It does NOT touch Flower,
Redis, or the cluster — it drives OnlineSACRobot directly.

Two things must hold or the cluster run won't work either:
  (i)  eval return RISES over rounds and plateaus below the env ceiling
       (genuine online learning — no saturation), AND
  (ii) simulating a cold restart (wipe local replay + optimizers, keep the
       global actor weights) makes the eval return visibly DROP then recover.
If (ii) recovers instantly, the Issue-#2 continuity contrast is dead — tune
MIN_BUFFER_FILL / BUFFER_CAPACITY / STEPS_PER_ROUND until the dip is real.

Usage (inside the container, with task2 pylibs on PYTHONPATH):
  python3 smoke_test.py --rounds 40 --wipe-at 20
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "worker"))
from online_sac_worker import OnlineSACRobot, STEPS_PER_ROUND  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--steps-per-round", type=int, default=STEPS_PER_ROUND)
    ap.add_argument("--wipe-at", type=int, default=20,
                    help="round at which to simulate a cold restart")
    ap.add_argument("--client-id", type=int, default=0)
    args = ap.parse_args()

    robot = OnlineSACRobot(args.client_id)
    print(f"env={robot.env.spec.id} obs={robot.obs_dim} act={robot.act_dim} "
          f"steps/round={args.steps_per_round}", flush=True)

    history = []
    pre_wipe_peak = -1e9
    wipe_return = None
    post_wipe_min = 1e9

    for rd in range(args.rounds):
        if rd == args.wipe_at:
            ev = robot.eval_return()
            wipe_return = ev["eval_return"]
            robot.wipe_local_state()
            print(f"--- round {rd}: SIMULATED COLD RESTART "
                  f"(eval_return just before wipe = {wipe_return:.1f}) ---",
                  flush=True)

        stats = robot.collect_and_train(args.steps_per_round)
        ev = robot.eval_return()
        ret = ev["eval_return"]
        history.append(ret)
        if rd < args.wipe_at:
            pre_wipe_peak = max(pre_wipe_peak, ret)
        else:
            post_wipe_min = min(post_wipe_min, ret)
        print(f"round {rd:3d}  eval_return={ret:8.1f}  success={ev['eval_success']:.2f}  "
              f"buffer={stats['buffer']:6d}  critic_loss={stats['critic_loss']:.3f}  "
              f"alpha={stats['alpha']:.3f}", flush=True)

    final = history[-1]
    print("\n===== SMOKE-TEST VERDICT =====", flush=True)
    rose = pre_wipe_peak > (history[0] + 50)     # curve rose meaningfully
    dipped = (wipe_return is not None
              and post_wipe_min < wipe_return - 50)  # wipe caused a real drop
    recovered = final > post_wipe_min + 50            # then re-climbed
    print(f"(i)  rose:      pre-wipe peak {pre_wipe_peak:.1f} vs start {history[0]:.1f}"
          f"  -> {'PASS' if rose else 'FAIL'}")
    print(f"(ii) dipped:    post-wipe min {post_wipe_min:.1f} vs pre-wipe {wipe_return}"
          f"  -> {'PASS' if dipped else 'FAIL'}")
    print(f"(ii) recovered: final {final:.1f} vs post-wipe min {post_wipe_min:.1f}"
          f"  -> {'PASS' if recovered else 'FAIL'}")
    ok = rose and dipped and recovered
    print(f"\nGATE: {'PASS — safe to submit to cluster' if ok else 'FAIL — do NOT submit; tune knobs'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
