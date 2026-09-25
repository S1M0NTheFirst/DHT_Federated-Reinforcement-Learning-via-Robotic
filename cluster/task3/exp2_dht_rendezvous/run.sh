#!/bin/bash
#MSUB -N task3_exp2_dht
#MSUB -W group_list=hpc2-coe-users
#MSUB -l walltime=04:00:00
#MSUB -j oe
# Exp2 — DHT rendezvous vs a central coordinator (Redis). No robots, no FL.
#   phase 1 scale      : lookup/publish cost vs ring size (8..128 nodes, k=3)
#   phase 2 churn      : lookup success after killing 0..48 of 64 nodes (k=3, k=8)
#   phase 3 redis_down : Redis lookups with its server up, then shut down
# One ring-host process per node (3 nodes); lookups are issued from the two
# worker nodes, Redis runs on the server node, so both backends cross the network.
#
# Submit from the cluster head node:
#   cluster/tools/submit_free.sh cluster/task3/exp2_dht_rendezvous/run.sh
# Quick sanity run (~10 min):
#   cluster/tools/submit_free.sh cluster/task3/exp2_dht_rendezvous/run.sh -v OVR_EXP2_QUICK=1
# Figures afterwards (any machine with the results copied back):
#   python3 cluster/task3/evaluation/make_exp2_figures.py

set -uo pipefail
CLUSTER_ROOT="${CLUSTER_ROOT:-$HOME/cluster}"
HERE="$CLUSTER_ROOT/task3/exp2_dht_rendezvous"

source "$CLUSTER_ROOT/common/cluster_config.sh"
source "$CLUSTER_ROOT/common/cluster_lib.sh"
source "$CLUSTER_ROOT/task3/common/task3_config.sh"

export MIN_ALIVE_NODES=3
setup_run_dirs "exp2_dht_rendezvous"

HOSTS=()
RING_PIDS=()

is_self() {
    local n="$1" f s
    f=$(hostname -f); s=$(hostname -s)
    [[ "$n" == "$f" || "$n" == "$s" || "$n" == "$s."* ]]
}

on_node() {   # on_node <node> <command...>  (this cluster can't ssh to itself)
    local n="$1"; shift
    if is_self "$n"; then
        bash -c "$*"
    else
        ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$n" "$*"
    fi
}

kill_ring_hosts() {
    local p n
    for p in "${RING_PIDS[@]:-}"; do
        [[ -n "$p" ]] && kill "$p" 2>/dev/null || true
    done
    RING_PIDS=()
    for n in "${HOSTS[@]:-}"; do
        [[ -z "$n" ]] && continue
        on_node "$n" "pkill -u \$USER -f dht_ring_host.py 2>/dev/null || true" || true
    done
}

exp2_cleanup() {
    kill_ring_hosts
    cleanup_all_nodes
}
trap 'exp2_cleanup; exit 130' INT
trap 'exp2_cleanup; exit 143' TERM
trap 'exp2_cleanup' EXIT

# start_ring <ring size> <ksize> <label>
# Splits the ring over the 3 nodes. Remote hosts run as tracked SSH children
# with `exec` (sshd HUPs them if this job dies) — same pattern as Redis.
start_ring() {
    local N="$1" K="$2" label="$3"
    local per=$((N / 3)) rem=$((N % 3)) off=0 h
    kill_ring_hosts
    sleep 1
    for h in 0 1 2; do
        local n_h=$per
        (( h < rem )) && n_h=$((per + 1))
        local node="${HOSTS[$h]}"
        local log="$RUN_LOG_DIR/ring_${label}_h${h}.log"
        local cmd="python3 -u $HERE/dht_ring_host.py --host-index $h --n-local $n_h \
--id-offset $off --base-port $EXP2_DHT_BASE_PORT --ctrl-port $EXP2_CTRL_PORT \
--bootstrap-host ${HOSTS[0]} --bootstrap-port $EXP2_DHT_BASE_PORT \
--ksize $K --alpha $EXP2_ALPHA --rpc-timeout $EXP2_RPC_TIMEOUT \
--max-lifetime $EXP2_HOST_MAX_LIFETIME"
        if is_self "$node"; then
            $cmd > "$log" 2>&1 < /dev/null &
        else
            ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
                -n "$node" \
                "source $CONDA_BASE/bin/activate $CONDA_ENV && export PYTHONUNBUFFERED=1 && exec $cmd" \
                > "$log" 2>&1 < /dev/null &
        fi
        RING_PIDS+=($!)
        off=$((off + n_h))
    done
}

stop_ring() {   # bench.py sends `quit`; give the hosts a moment, then force.
    local t p alive
    for t in {1..20}; do
        alive=0
        for p in "${RING_PIDS[@]:-}"; do
            [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null && alive=1
        done
        [[ $alive -eq 0 ]] && break
        sleep 1
    done
    kill_ring_hosts
    sleep 1
}

run_bench() {
    python3 -u "$HERE/bench.py" \
        --hosts "${HOSTS[0]}:$EXP2_CTRL_PORT,${HOSTS[1]}:$EXP2_CTRL_PORT,${HOSTS[2]}:$EXP2_CTRL_PORT" \
        --out "$RESULTS_DIR" --rpc-timeout "$EXP2_RPC_TIMEOUT" --quit-hosts "$@" \
        2>&1 | tee -a "$RUNNER_LOG"
    return "${PIPESTATUS[0]}"
}

# --------------------------------------------------------------------------- #
pick_alive_nodes || exit 1
HOSTS=("$SERVER_NODE" "$CLIENT_NODE_1" "$CLIENT_NODE_2")
start_redis_on_server || exit 1

source "$CONDA_BASE/bin/activate" "$CONDA_ENV"
export PYTHONUNBUFFERED=1

RESULTS_DIR="$RESULTS_ROOT/exp2_dht_rendezvous/${PBS_JOBID:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS_DIR"
echo ">>> Exp2 results -> $RESULTS_DIR" | tee -a "$RUNNER_LOG"

# Preflight: kademlia + redis-py importable on every node.
for n in "${HOSTS[@]}"; do
    if ! on_node "$n" "source $CONDA_BASE/bin/activate $CONDA_ENV && python3 -c 'import kademlia, redis'"; then
        echo "FATAL: kademlia/redis-py missing in conda env '$CONDA_ENV' on $n" | tee -a "$RUNNER_LOG"
        exit 1
    fi
done
kill_ring_hosts   # stale ring hosts from an earlier job on these nodes

FAILED=0

echo ">>> [1/3] scale sweep: sizes=($EXP2_RING_SIZES) k=$EXP2_KSIZE keys=$EXP2_KEYS" | tee -a "$RUNNER_LOG"
for N in $EXP2_RING_SIZES; do
    start_ring "$N" "$EXP2_KSIZE" "scale_n${N}"
    run_bench --phase scale --ring-size "$N" --ksize "$EXP2_KSIZE" \
        --keys "$EXP2_KEYS" --seed "$((17 + N))" \
        --redis-host "$SERVER_NODE" --redis-port "$REDIS_PORT" || FAILED=$((FAILED + 1))
    stop_ring
done

echo ">>> [2/3] churn: N=$EXP2_CHURN_N k=($EXP2_CHURN_KSIZES) fails=($EXP2_CHURN_FAIL_COUNTS) trials=$EXP2_CHURN_TRIALS" | tee -a "$RUNNER_LOG"
for K in $EXP2_CHURN_KSIZES; do
    for T in $(seq 1 "$EXP2_CHURN_TRIALS"); do
        for C in $EXP2_CHURN_FAIL_COUNTS; do
            start_ring "$EXP2_CHURN_N" "$K" "churn_k${K}_t${T}_f${C}"
            run_bench --phase churn --ring-size "$EXP2_CHURN_N" --ksize "$K" \
                --trial "$T" --fail-count "$C" --keys "$EXP2_CHURN_KEYS" \
                --settle "$EXP2_CHURN_SETTLE" --seed "$((1000 * T + 100 * K + C))" \
                || FAILED=$((FAILED + 1))
            stop_ring
        done
    done
done

# Last on purpose: this phase shuts the Redis server down.
echo ">>> [3/3] redis_down: keys=$EXP2_REDIS_DOWN_KEYS" | tee -a "$RUNNER_LOG"
start_ring 6 "$EXP2_KSIZE" "redis_down"
run_bench --phase redis_down --ring-size 6 --ksize "$EXP2_KSIZE" \
    --keys "$EXP2_REDIS_DOWN_KEYS" --seed 99 \
    --redis-host "$SERVER_NODE" --redis-port "$REDIS_PORT" || FAILED=$((FAILED + 1))
stop_ring

echo ">>> Exp2 finished: $FAILED failed bench invocation(s). Results: $RESULTS_DIR" | tee -a "$RUNNER_LOG"
[[ $FAILED -eq 0 ]]
