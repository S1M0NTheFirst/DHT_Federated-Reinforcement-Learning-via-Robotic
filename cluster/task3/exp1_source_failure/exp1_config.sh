#!/bin/bash
# Exp1 config — sourced by exp1_source_failure/run.sh AFTER
#   cluster/common/cluster_config.sh, task2/common/task2_config.sh (FL knobs,
#   pylibs2) and task3/common/task3_config.sh (task3 log/result roots).
# Exp1 = the SOURCE dies during a migration. Real FL fleet (task2 Hopper SAC
# workers + Flower FedAvg on 3 nodes); at a chosen kill point the source worker
# is SIGKILLed and its local bundle deleted, then the robot is relaunched on the
# destination from whatever copy of its state survived.

# Condition (pass with msub -v EXP1_CONDITION=...):
#   dht_frl      bundle replicated to EXP1_REPLICAS other nodes, replica set
#                published in a cross-node Kademlia DHT; destination resolves it
#                and pulls every file from any surviving copy
#   dht_r0       ablation: DHT pointer to the source only (task2 design, no
#                replicas) -> shows the replicas, not the lookup, save the state
#   app_cold     App-CR cold: stop-the-world rsync of the bundle from the source
#   app_warm     App-CR warm: background pre-copy EXP1_PRECOPY_ROUNDS before the
#                migration, final delta sync at the stop; a source that dies
#                before the final sync completes leaves the older pre-copy,
#                which is restored (restored_stale=1)
#   tcp_scp      App-CR direct: destination scp's the bundle from the source
#   cold_restart no state transfer; relaunch fresh (loses replay + optimizer)
#   a+b+...      a + list of the above (msub -v splits on commas), run one after another in one job
#   all          every condition above, one after another in one job (meant for
#                the quick run: validates every condition path)
export EXP1_CONDITION="${EXP1_CONDITION:-dht_frl}"

# Ports: clear of task2 (redis 6579, flower 8570) and Exp2 (redis 6679, DHT
# 23700-23742 UDP, control 8690). UDP between compute nodes is only open on
# 23000-24000 (admin rule, Sep 2026); TCP high ports are open.
export REDIS_PORT=6779
export FLOWER_PORT=8870
export EXP1_DHT_BASE_PORT=23100          # ring nodes use 23100..23100+n-1 per host (UDP)
export EXP1_CTRL_PORT=8790               # one control port per DHT host process (TCP)
export EXP1_DHT_NODES_PER_HOST=8         # 3 hosts -> 24-node Kademlia ring
export EXP1_DHT_KSIZE=3                  # record replication / lookup width
export EXP1_DHT_ALPHA=3
export EXP1_DHT_RPC_TIMEOUT=1.0

# Bundle replication (dht_frl). Replicas go to nodes other than the source; with
# EXCLUDE_DST=1 the destination is skipped too, so recovery always pulls over
# the network from a third node (no free pre-copy onto the destination). With 3
# nodes that allows 1 replica; set EXCLUDE_DST=0 to allow 2 (third node + dst).
export EXP1_REPLICAS=1
export EXP1_REPLICA_EXCLUDE_DST=1

# Kill points, assigned per event as KILL_POINTS[(client_id + migration_index)
# % n] so every robot and every migration wave sees all of them, identically in
# every condition:
#   after_save      source dies once its bundle is saved (and, for dht_frl,
#                   replicated + published); destination has nothing yet
#   mid_transfer    source dies after the first bundle file reached the
#                   destination, before the replay buffer did
#   after_transfer  source dies after the destination holds the full bundle
export EXP1_KILL_POINTS="after_save,mid_transfer,after_transfer"

export EXP1_PROBE_TIMEOUT=300            # s to wait for a relaunched worker to report
export EXP1_MAX_LIFETIME=21600           # s; DHT hosts + workers self-exit after this
export EXP1_STALL_TIMEOUT=1800           # s without a new eval row -> FL stalled, stop
# A dead client costs a round at most this long (flwr aborts it). Normal full
# rounds take 26-30 s (p95 ~40 s); kill rounds add the kill + relaunch.
export EXP1_ROUND_TIMEOUT=150
# A source that is still alive this long after its migration request exits by
# itself, so a failed kill can never leave two copies of a robot in FedAvg.
# Every remote step now has a hard deadline; the slowest possible path to the
# kill (after_transfer, every copy timing out twice) is ~15 min, typical is
# seconds. The grace is only a safety net, so it sits well above that.
export EXP1_SOURCE_GRACE=1200
# Warm every node (image + libraries) before each condition's fleet starts.
export EXP1_WARMUP=1
# app_warm: pre-copy this many rounds before the migration (taken at the start
# of that round, so the stale copy misses that much training), and how long a
# migration waits for a pre-copy that is still running before ignoring it.
export EXP1_PRECOPY_ROUNDS=1
export EXP1_PRECOPY_WAIT=30

# Same migration schedule as task2 so curves line up (worker staggers each robot
# by client_id % 5 rounds). 20 robots x 5 migrations = 100 kill events.
export TOTAL_FL_ROUNDS=150
export MIGRATION_ROUNDS="30,60,90,120,140"
export NUM_CLIENTS=20
export ROBOTS_PER_NODE=10

# Quick smoke run (~8-10 min per condition; all 6 ~1 h): -v OVR_EXP1_QUICK=1
if [[ -n "${OVR_EXP1_QUICK:-}" ]]; then
    export NUM_CLIENTS=4
    export ROBOTS_PER_NODE=2
    export TOTAL_FL_ROUNDS=12
    export MIGRATION_ROUNDS="4,8"
    export STEPS_PER_ROUND=300
    export MIN_BUFFER_FILL=300
    export EVAL_EPISODES=1
    export EXP1_PROBE_TIMEOUT=240
    export EXP1_MAX_LIFETIME=7200
    export EXP1_STALL_TIMEOUT=900
    export EXP1_ROUND_TIMEOUT=60
    export EXP1_SOURCE_GRACE=600
fi
