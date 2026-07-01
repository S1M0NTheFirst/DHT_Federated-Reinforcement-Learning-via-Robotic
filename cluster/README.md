# SwiftBot-RL on the CSULB HPC2 cluster (Apptainer)
The CPU and GPU vary within the cluster. From the head node, you can ssh to any node (n001 to n036) and run "lscpu" to see the CPU information. The GPU nodes are n021 and n022 on NVIDIA Tesla P100-PCIE-16GB and n034, n035, and n036 on NVIDIA GeForce RTX 3090. You can ssh to those nodes and run "nvidia-smi" for more details. We do not restrict the number of GPUs you can request for a job.

Cluster port of the migration-mechanism comparison. Runs the same five
conditions as the workstation experiment, but:

- **Apptainer** instead of Docker (cluster has no rootful Docker daemon).
- **Cross-node migration**: 1 server node + 2 client nodes; every migration
  moves a robot from client #1 ↔ client #2.
- **20 robots** instead of 8 (10 per client node).
- `MIGRATION_OFFSET=10`, `TOTAL_TASKS=1200`, `N_ROUNDS=60` — keeps all 5
  forced migrations per robot inside the task budget. Workstation defaults
  (25/1000/50) are preserved when cluster env vars aren't set.

## Layout

```
cluster/
  apptainer/
    build_images.sh   robot.sif         baseline.sif (→ robot.sif)
    pylibs/           (host site-packages, bind-mounted at /pylibs)
  common/
    cluster_config.sh   cluster_lib.sh
    cluster_runner.py   baseline_runner_base.py
  condition_A_dht_frl/   runner_A.py  run_A.sh
  condition_B_apptainer_state/   runner_B.py  run_B.sh
  condition_C_criu_cold/         runner_C.py  run_C.sh
  condition_D_criu_warm/         runner_D.py  run_D.sh
  condition_E_cold_restart/      runner_E.py  run_E.sh
  logs/              filled at runtime (not in git)
  results/           per-condition migration_events.csv
```

## One-time setup (per cluster account)

Layout on the cluster (default):

```
/home/<beach_id>/
  cluster/        ← this folder (orchestration scripts, .sh files, runners)
  swiftbot_rl/    ← original Python code (dht_frl/, criu_cold/, robot/, etc.)
                    bind-mounted into the apptainer container at /app
```

The default beach ID is `029822154` — change it in
`cluster/common/cluster_config.sh` if your account differs. If your
`swiftbot_rl/` lives somewhere else, edit `SWIFTBOT_RL_ROOT` in the same
file.

1. Upload the two folders so they land at the paths above.
2. Pull the apptainer image and install Python deps (~10 min, no root needed):
   ```bash
   cd ~/cluster/apptainer
   bash build_images.sh
   ```
   This script does NOT use `apptainer build` (which would need fakeroot or
   the now-removed `--remote` cloud builds). Instead it:
   - `apptainer pull`s a prebuilt PyTorch+CUDA image from Docker Hub →
     `robot.sif` (`baseline.sif` is a symlink to it),
   - runs `pip install --target=pylibs/` from inside the container to drop
     `flwr`, `kademlia`, `stable-baselines3`, etc. into a host directory,
   - the runner bind-mounts `pylibs/` into every robot at `/pylibs` and
     prepends it to `PYTHONPATH`.

   Re-run the script anytime to refresh packages. To force a clean reinstall,
   delete `pylibs/.installed_v1` (or the whole `pylibs/` dir).

   `criu` cannot be pip-installed and must be available system-wide on the
   compute nodes. If it isn't, Conditions C/D auto-fall-back to SIMULATE.
3. Confirm Redis works on a compute node:
   ```bash
   ssh n034 "redis-cli ping"   # → PONG
   ```

## Running a condition

Each condition is one MSUB job that requests 3 nodes (1 server + 2 client).
Submit:

```bash
cd ~/cluster
msub condition_A_dht_frl/run_A.sh   # ≈ 6 h walltime requested
msub condition_B_apptainer_state/run_B.sh   # ≈ 3 h
msub condition_C_criu_cold/run_C.sh         # ≈ 3 h
msub condition_D_criu_warm/run_D.sh         # ≈ 4 h
msub condition_E_cold_restart/run_E.sh      # ≈ 2 h
```

Default node selection in every `run_X.sh` is `n034 + n035 + n036` (RTX
3090). For the P100 nodes, change the `#MSUB -l nodes=...` line to
`n021 + n022 + n034` (only two P100 nodes exist; pad with one 3090 if you
need three nodes).

Monitor a job:
```bash
showq -u $USER
checkjob <jobid>
canceljob <jobid>     # if you need to bail
```

## Real-time log streaming

While a job is running, all logs are tee'd into
`cluster/logs/<condition>/<jobid>/`. Tail any of:

```bash
LOGS=~/cluster/logs/condition_A_dht_frl/<jobid>
tail -F $LOGS/runner.log              # host-side runner
tail -F $LOGS/server.log              # server node (Redis startup, etc.)
tail -F $LOGS/flower_server.log       # Condition A only
tail -F $LOGS/client_node1.log $LOGS/client_node2.log
tail -F $LOGS/robot_*.log             # per-robot worker output
```

The `runner.log` shows the live status snapshot every 15 s — the same
table you saw on the workstation.

## Cleanup — orphan-safe by design

The cluster admin warned us once that the previous setup left orphaned
container processes that could only be cleared by rebooting the nodes. The
current code is built so this can't happen:

- **No `apptainer instance start`.** Every robot is launched with
  `apptainer exec` in the FOREGROUND of an SSH session. Apptainer is the
  immediate child of the remote `sshd` (via `exec env …`), so when the SSH
  connection closes, the kernel sends `SIGHUP` straight to apptainer-exec
  and the worker dies. No persistent "Apptainer runtime parent".
- **Every SSH connection is a tracked Popen child** of the runner Python
  process (and the Redis SSH is a tracked bash background child of the
  run_X.sh shell). When the runner exits — cleanly, on signal, or because
  MOAB killed the job — those Popens die, sshd HUPs the remote commands,
  and everything terminates.
- **`run_X.sh` traps SIGTERM and SIGINT** in addition to EXIT. MOAB sends
  SIGTERM before SIGKILL on walltime / `canceljob`; we catch it and run
  `cleanup_all_nodes` before the SIGKILL window closes.
- **`cleanup_all_nodes` is two-phase**: (1) kill local SSH children →
  remote sshd HUPs everything; (2) ssh into each node and `pkill -TERM`
  then `pkill -9` any remaining workers, plus `rm -rf /tmp/swiftbot_*`.
  Then prints a verification line so the runner.log shows the final
  process count on each node.

To verify yourself after a job ends:

```bash
for n in n034 n035 n036; do
    echo "=== $n ==="
    ssh $n "pgrep -u $USER -af 'apptainer|worker_|redis-server'"
done
```

Should return empty. If it doesn't, grep `runner.log` for the
`Post-cleanup verification:` block — it'll show what survived and on which
node.

## Known limitations / honest framing

These are documented prominently because they affect how you interpret the
CSV outputs:

- **Condition A "migration" is bundle replication, not process relocation.**
  The DHT worker has no in-place resume mechanism, and rootless apptainer
  on shared HPC can't relocate a process across nodes via CRIU. The runner
  rsyncs the policy/replay bundle to the destination node and back to the
  source so the worker can reload from a node-local path. This still
  measures the headline claim — bundle transfer is fast (~5 MB) and the
  policy load time is real — but it's not "moving the robot to another
  machine" in the strict sense. Workstation Condition A had the same
  property (containers stayed put; CRIU dump was for measurement only).

- **Condition B is "container-engine handoff" by analogy.** Apptainer has no
  `apptainer checkpoint` command. The runner approximates it by stopping
  the source apptainer instance, rsync'ing state, and starting a fresh
  instance on the destination node. With the random-policy baseline worker
  there is no learned model state to preserve, only `task_counter` (carried
  via Redis). The number that matters here is the apptainer instance
  start+stop overhead vs Condition E.

- **Conditions C / D use application-level checkpointing, not kernel CRIU.**
  CRIU isn't installed on the compute nodes (`which criu` returns nothing)
  and can't be pip-installed. Instead, the cluster-local worker
  `cluster/workers/worker_app_checkpoint.py` maintains a synthetic
  ~20 MB PyTorch state (model + Adam optimizer moments + replay buffer +
  RNG state) and `torch.save`s it on migration request. The runner rsyncs
  the file to the destination node and starts a fresh worker there with
  `APP_RESTORE_FROM` set, which `torch.load`s the state. The CSV column
  `criu_mode` is `app_cold` for Condition C and `app_warm` for D. The
  synthetic model is *not* used to make bidding decisions (random policy
  unchanged); it exists purely so dump/transfer/restore numbers reflect
  realistic state size for a small-scale RL agent.

  Condition D adds a runner-side pre-copy thread that rsyncs each robot's
  warm snapshot to the other client node every ~20 s during normal
  operation, so the migration-window rsync hits a delta-friendly target
  and ships fewer bytes. The bandwidth spent during normal operation is
  logged at the end of the run as `cumulative pre-copy bytes per robot`.

- **Cluster numbers are not directly comparable to workstation numbers.**
  20 robots vs 8, cross-node vs same-host, apptainer vs Docker — keep them
  in separate paper sections.

## Editing experiment knobs

All of these are in `cluster/common/cluster_config.sh`:

```bash
NUM_CLIENTS=20          # 10 robots × 2 client nodes
ROBOTS_PER_NODE=10
MIGRATION_OFFSET=10     # client_id × this = task offset of the 5 events
TOTAL_TASKS=1200        # raised from 1000 to fit 20-robot offsets
TOTAL_FL_ROUNDS=60      # 60×20=1200 tasks; matches TOTAL_TASKS
FLOWER_PORT=8470        # Condition A
REDIS_PORT=6379
```

The runners read all of these from environment so changing them takes
effect on the next `msub`.
