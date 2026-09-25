#!/bin/bash
# task3 config — sourced by every task3 run.sh AFTER cluster/common/cluster_config.sh.
# task3 = the follow-up DHT evaluations (exp1 source failure, exp2 rendezvous
# benchmark, exp3 node crash). task2 code and results are left untouched.

export TASK3_ROOT="${CLUSTER_ROOT}/task3"

# Keep ALL task3 logs + results inside the task3 folder.
export LOG_ROOT="${TASK3_ROOT}/logs"
export RESULTS_ROOT="${TASK3_ROOT}/results"

# Ports offset from task1 (redis 6379, DHT 8480) and task2 (redis 6579,
# flower 8570, DHT 8600) so a task3 job can share nodes with older jobs.
export REDIS_PORT=6679

# --------------------------------------------------------------------------- #
# Exp2 — DHT rendezvous vs a central coordinator (Redis). No robots / no FL.   #
# --------------------------------------------------------------------------- #
export EXP2_DHT_BASE_PORT=8700        # ring nodes on each host use 8700..8700+n-1
export EXP2_CTRL_PORT=8690            # one control port per ring-host process
export EXP2_ALPHA=3                   # Kademlia lookup parallelism (library default)
export EXP2_RPC_TIMEOUT=1.0           # seconds before a silent peer counts as dead
export EXP2_HOST_MAX_LIFETIME=1800    # ring hosts self-exit after this (no orphans)

# Scale sweep: lookup cost vs ring size. ksize must stay < ring size so a value
# lives on only k of N nodes and every GET has to route to find it.
export EXP2_RING_SIZES="8 16 32 64 128"
export EXP2_KSIZE=3
export EXP2_KEYS=500

# Churn: kill this many of EXP2_CHURN_N nodes (random), then look every key up.
# Each (ksize, trial, fail count) gets a fresh ring so levels are independent.
export EXP2_CHURN_N=64
export EXP2_CHURN_KSIZES="3 8"
export EXP2_CHURN_TRIALS=3
export EXP2_CHURN_FAIL_COUNTS="0 1 6 16 32 48"
export EXP2_CHURN_KEYS=500
export EXP2_CHURN_SETTLE=2            # seconds between the kills and the lookups

# Redis-down phase: lookups with the Redis server up, then after it is shut down.
export EXP2_REDIS_DOWN_KEYS=200

# Quick sanity run:  tools/submit_free.sh task3/exp2_dht_rendezvous/run.sh -v OVR_EXP2_QUICK=1
if [[ -n "${OVR_EXP2_QUICK:-}" ]]; then
    export EXP2_RING_SIZES="8 16"
    export EXP2_KEYS=50
    export EXP2_CHURN_N=16
    export EXP2_CHURN_KSIZES="3"
    export EXP2_CHURN_TRIALS=1
    export EXP2_CHURN_FAIL_COUNTS="0 8"
    export EXP2_CHURN_KEYS=50
    export EXP2_REDIS_DOWN_KEYS=20
fi
