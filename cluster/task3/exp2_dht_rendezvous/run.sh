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
        return
    fi
    # sshd on these nodes sometimes drops a connection at once when several
    # arrive close together; exit 255 means ssh itself failed, so retry.
    local attempt rc
    for attempt in 1 2 3 4; do
        ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$n" "$*"
        rc=$?
        [[ $rc -ne 255 ]] && return $rc
        sleep $((attempt * 5))
    done
    return 255
}

# launch_host <node> <log> <cmd>: start <cmd> on <node> in the background and
# set LAUNCH_PID. Remote commands run as tracked SSH children with `exec`
# (sshd HUPs them if this job dies) — same pattern as Redis.
launch_host() {
    local node="$1" log="$2" cmd="$3"
    if is_self "$node"; then
        $cmd >> "$log" 2>&1 < /dev/null &
    else
        ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
            -o ConnectTimeout=10 -o StrictHostKeyChecking=no -n "$node" \
            "source $CONDA_BASE/bin/activate $CONDA_ENV && export PYTHONUNBUFFERED=1 && exec $cmd" \
            >> "$log" 2>&1 < /dev/null &
    fi
    LAUNCH_PID=$!
}

kill_ring_hosts() {
    local p n
    for p in "${RING_PIDS[@]:-}"; do
        [[ -n "$p" ]] && kill "$p" 2>/dev/null || true
    done
    RING_PIDS=()
    for n in "${HOSTS[@]:-}"; do
        [[ -z "$n" ]] && continue
        # [d]/[n]: the pattern must not match the remote shell running pkill,
        # or pkill kills that shell and ssh reports a failure. Wait until the
        # processes are really gone (their ports are reused by the next ring).
        on_node "$n" "P='[d]ht_ring_host.py|[n]et_check.py'; \
pkill -u \$USER -f \"\$P\" 2>/dev/null; \
for i in 1 2 3 4 5 6 7 8 9 10; do pgrep -u \$USER -f \"\$P\" >/dev/null || exit 0; sleep 1; done; \
pkill -9 -u \$USER -f \"\$P\" 2>/dev/null; sleep 1; true" || true
    done
}

exp2_cleanup() {
    kill_ring_hosts
    cleanup_all_nodes
}
trap 'exp2_cleanup; exit 130' INT
trap 'exp2_cleanup; exit 143' TERM
trap 'exp2_cleanup' EXIT

# wait_launched <marker> <nodes array name> <logs array name> <cmds array name>
# Waits until every log shows <marker>. A remote launch whose SSH dies before
# that (sshd dropped the connection) is relaunched, up to 4 times per host.
wait_launched() {
    local marker="$1"
    local -n _nodes="$2" _logs="$3" _cmds="$4"
    local -a tries=(0 0 0)
    local t h pending
    for t in $(seq 1 120); do
        pending=0
        for h in 0 1 2; do
            grep -q "$marker" "${_logs[$h]}" 2>/dev/null && continue
            pending=1
            kill -0 "${RING_PIDS[$h]}" 2>/dev/null && continue
            if (( tries[h] >= 4 )); then
                echo "!!! host $h (${_nodes[$h]}) failed to start; last log lines:" \
                    | tee -a "$RUNNER_LOG"
                tail -3 "${_logs[$h]}" | sed 's/^/    /' | tee -a "$RUNNER_LOG"
                return 1
            fi
            tries[h]=$((tries[h] + 1))
            echo ">>> relaunching host $h on ${_nodes[$h]} (attempt $((tries[h] + 1)))" \
                | tee -a "$RUNNER_LOG"
            sleep $((tries[h] * 5))
            launch_host "${_nodes[$h]}" "${_logs[$h]}" "${_cmds[$h]}"
            RING_PIDS[$h]=$LAUNCH_PID
        done
        [[ $pending -eq 0 ]] && return 0
        sleep 1
    done
    echo "!!! hosts did not start within 120s" | tee -a "$RUNNER_LOG"
    return 1
}

# start_ring <ring size> <ksize> <label>
# Splits the ring over the 3 nodes; returns once every ring host is up.
start_ring() {
    local N="$1" K="$2" label="$3"
    local per=$((N / 3)) rem=$((N % 3)) off=0 h
    local -a nodes=() logs=() cmds=()
    RING_PIDS=()
    for h in 0 1 2; do
        local n_h=$per
        (( h < rem )) && n_h=$((per + 1))
        nodes[$h]="${HOSTS[$h]}"
        logs[$h]="$RUN_LOG_DIR/ring_${label}_h${h}.log"
        cmds[$h]="python3 -u $HERE/dht_ring_host.py --host-index $h --n-local $n_h \
--id-offset $off --base-port $EXP2_DHT_BASE_PORT --ctrl-port $EXP2_CTRL_PORT \
--bootstrap-host ${HOSTS[0]} --bootstrap-port $EXP2_DHT_BASE_PORT \
--ksize $K --alpha $EXP2_ALPHA --rpc-timeout $EXP2_RPC_TIMEOUT \
--max-lifetime $EXP2_HOST_MAX_LIFETIME"
        : > "${logs[$h]}"
        launch_host "${nodes[$h]}" "${logs[$h]}" "${cmds[$h]}"
        RING_PIDS[$h]=$LAUNCH_PID
        off=$((off + n_h))
    done
    wait_launched "control port" nodes logs cmds
}

# net_check: before any ring starts, check every node can reach the other two
# on the ring's UDP ports and the control TCP port (plus Redis on the server).
# Extra UDP ports are only reported, to tell a port rule from a full UDP block.
net_check() {
    local udp="$EXP2_DHT_BASE_PORT,$((EXP2_DHT_BASE_PORT + 42))"
    # info only: both ends of the UDP range the admin opened (23000-24000)
    local info="23000,23999"
    local h o rc=0 targets
    local -a nodes=() logs=() cmds=()
    RING_PIDS=()
    for h in 0 1 2; do
        nodes[$h]="${HOSTS[$h]}"
        logs[$h]="$RUN_LOG_DIR/netcheck_h${h}.log"
        cmds[$h]="python3 -u $HERE/net_check.py serve --udp-ports $udp --udp-info $info \
--tcp-port $EXP2_CTRL_PORT --seconds 300"
        : > "${logs[$h]}"
        launch_host "${nodes[$h]}" "${logs[$h]}" "${cmds[$h]}"
        RING_PIDS[$h]=$LAUNCH_PID
    done
    if ! wait_launched "net_check serving" nodes logs cmds; then
        kill_ring_hosts
        return 1
    fi
    for h in 0 1 2; do
        targets=""
        for o in 0 1 2; do
            [[ $o -eq $h ]] && continue
            if [[ $o -eq 0 ]]; then
                targets="$targets --target ${HOSTS[$o]}:$EXP2_CTRL_PORT,$REDIS_PORT"
            else
                targets="$targets --target ${HOSTS[$o]}:$EXP2_CTRL_PORT"
            fi
        done
        on_node "${HOSTS[$h]}" "source $CONDA_BASE/bin/activate $CONDA_ENV && \
python3 -u $HERE/net_check.py probe --udp-ports $udp --udp-info $info $targets" \
            2>&1 | tee -a "$RUNNER_LOG"
        [[ ${PIPESTATUS[0]} -ne 0 ]] && rc=1
    done
    kill_ring_hosts
    return $rc
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

echo ">>> network check: UDP ring ports + TCP control/Redis ports between nodes" | tee -a "$RUNNER_LOG"
if ! net_check; then
    echo "FATAL: the nodes cannot reach each other on the ports above (see BLOCKED lines)." \
        | tee -a "$RUNNER_LOG"
    exit 1
fi
echo ">>> network check passed" | tee -a "$RUNNER_LOG"

FAILED=0

echo ">>> [1/3] scale sweep: sizes=($EXP2_RING_SIZES) k=$EXP2_KSIZE keys=$EXP2_KEYS" | tee -a "$RUNNER_LOG"
for N in $EXP2_RING_SIZES; do
    if start_ring "$N" "$EXP2_KSIZE" "scale_n${N}"; then
        run_bench --phase scale --ring-size "$N" --ksize "$EXP2_KSIZE" \
            --keys "$EXP2_KEYS" --seed "$((17 + N))" --concurrency "$EXP2_SCALE_CONCURRENCY" \
            --redis-host "$SERVER_NODE" --redis-port "$REDIS_PORT" || FAILED=$((FAILED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
    stop_ring
done

echo ">>> [2/3] churn: N=$EXP2_CHURN_N k=($EXP2_CHURN_KSIZES) fails=($EXP2_CHURN_FAIL_COUNTS) trials=$EXP2_CHURN_TRIALS" | tee -a "$RUNNER_LOG"
for K in $EXP2_CHURN_KSIZES; do
    for T in $(seq 1 "$EXP2_CHURN_TRIALS"); do
        for C in $EXP2_CHURN_FAIL_COUNTS; do
            if start_ring "$EXP2_CHURN_N" "$K" "churn_k${K}_t${T}_f${C}"; then
                run_bench --phase churn --ring-size "$EXP2_CHURN_N" --ksize "$K" \
                    --trial "$T" --fail-count "$C" --keys "$EXP2_CHURN_KEYS" \
                    --settle "$EXP2_CHURN_SETTLE" --seed "$((1000 * T + 100 * K + C))" \
                    || FAILED=$((FAILED + 1))
            else
                FAILED=$((FAILED + 1))
            fi
            stop_ring
        done
    done
done

# Last on purpose: this phase shuts the Redis server down.
echo ">>> [3/3] redis_down: keys=$EXP2_REDIS_DOWN_KEYS" | tee -a "$RUNNER_LOG"
if start_ring 6 "$EXP2_KSIZE" "redis_down"; then
    run_bench --phase redis_down --ring-size 6 --ksize "$EXP2_KSIZE" \
        --keys "$EXP2_REDIS_DOWN_KEYS" --seed 99 \
        --redis-host "$SERVER_NODE" --redis-port "$REDIS_PORT" || FAILED=$((FAILED + 1))
else
    FAILED=$((FAILED + 1))
fi
stop_ring

echo ">>> Exp2 finished: $FAILED failed bench invocation(s). Results: $RESULTS_DIR" | tee -a "$RUNNER_LOG"
[[ $FAILED -eq 0 ]]
