"""Exp1 warm-up, run once per node (inside the robot container) before a
condition's fleet starts: import the worker stack, build a robot, step its
env, and save + load a bundle, then exit. The first node reads of the image
and libraries (shared filesystem) are slow: in the first quick runs the first
condition's round 1 took 268 s against 5-34 s for the later ones, and its
relaunches took 15-57 s against 7-8.5 s. Warming every node first makes every
condition, and every full per-condition job, start equally warm."""
import os
import shutil
import sys
import tempfile
import time

t0 = time.time()
sys.path.insert(0, os.environ.get("TASK2_WORKER_DIR", "/cluster_app/task2/worker"))
import flwr  # noqa: E402
import online_sac_worker as osw  # noqa: E402

robot = osw.OnlineSACRobot(0)
robot.collect_and_train(200)
robot.eval_return(1)
d = tempfile.mkdtemp(prefix="exp1_warmup_")
try:
    robot.save_bundle(d)
    robot.load_bundle(d)
finally:
    shutil.rmtree(d, ignore_errors=True)
print(f"exp1 warm-up done in {time.time() - t0:.1f}s (flwr {flwr.__version__})", flush=True)
