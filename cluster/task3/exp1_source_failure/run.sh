#!/bin/bash
#MSUB -N task3_exp1
#MSUB -W group_list=hpc2-coe-users
#MSUB -l walltime=03:00:00
#MSUB -j oe
# Exp1 — the SOURCE dies during a migration. Real FL fleet (task2 Hopper SAC
# workers + Flower FedAvg, 3 nodes); at each migration the source worker is
# SIGKILLed and its bundle deleted at a kill point (after_save / mid_transfer /
# after_transfer), then the robot is relaunched on the destination from any
# surviving copy of its state. One job = one condition (or all, in sequence):
#   dht_frl | dht_r0 | tcp_scp | cold_restart | all   (see exp1_config.sh)
#
# NEVER run two jobs on the same nodes: Redis start-up and the final cleanup
# kill ALL of your redis-server / apptainer processes on this job's nodes. The
# job refuses to start if your experiment processes already run on its nodes.
#
# Submit from the cluster head node (one job per condition):
#   cluster/tools/submit_free.sh cluster/task3/exp1_source_failure/run.sh -v EXP1_CONDITION=dht_frl
# Quick sanity run, every condition in one job (~1 h):
#   cluster/tools/submit_free.sh cluster/task3/exp1_source_failure/run.sh \
#       -v EXP1_CONDITION=all,OVR_EXP1_QUICK=1 -l walltime=01:45:00
# Figures afterwards (results copied back):
#   python3 cluster/task3/evaluation/make_exp1_figures.py

set -uo pipefail
CLUSTER_ROOT="${CLUSTER_ROOT:-$HOME/cluster}"
HERE="$CLUSTER_ROOT/task3/exp1_source_failure"

source "$CLUSTER_ROOT/common/cluster_config.sh"
source "$CLUSTER_ROOT/common/cluster_lib.sh"
source "$CLUSTER_ROOT/task2/common/task2_config.sh"
source "$CLUSTER_ROOT/task3/common/task3_config.sh"
source "$HERE/exp1_config.sh"

if [[ "$EXP1_CONDITION" == all ]]; then
    CONDS=(dht_frl dht_r0 tcp_scp cold_restart)
else
    CONDS=("$EXP1_CONDITION")
fi
USES_DHT=0
for c in "${CONDS[@]}"; do
    case "$c" in
        dht_frl|dht_r0|tcp_scp|cold_restart) ;;
        *) echo "FATAL: EXP1_CONDITION='$EXP1_CONDITION' (dht_frl|dht_r0|tcp_scp|cold_restart|all)"
           exit 1 ;;
    esac
    [[ "$c" == dht_* ]] && USES_DHT=1
done

export MIN_ALIVE_NODES=3
setup_run_dirs "exp1_source_failure/${EXP1_CONDITION}"
BASE_LOG_DIR="$RUN_LOG_DIR"
STAMP="${PBS_JOBID:-$(date +%Y%m%d_%H%M%S)}"

HOSTS=()
DHT_PIDS=()
# Only this experiment's helper processes (never Exp2's net_check or ring).
DHT_PAT='[t]ask3/exp1_source_failure/(exp1_dht_host|net_check)[.]py'

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
    # exit 255 = ssh itself failed (sshd drops bursts of connections): retry.
    local attempt rc
    for attempt in 1 2 3 4; do
        ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$n" "$*"
        rc=$?
        [[ $rc -ne 255 ]] && return $rc
        sleep $((attempt * 5))
    done
    return 255
}

# launch_host <node> <log> <cmd>: background <cmd> on <node>, set LAUNCH_PID.
# Remote commands run as tracked SSH children with `exec`; the python helpers
# also self-exit when their parent disappears.
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

kill_dht_hosts() {
    local p n
    for p in "${DHT_PIDS[@]:-}"; do
        [[ -n "$p" ]] && kill "$p" 2>/dev/null || true
    done
    DHT_PIDS=()
    for n in "${HOSTS[@]:-}"; do
        [[ -z "$n" ]] && continue
        on_node "$n" "P='$DHT_PAT'; pkill -u \$USER -f \"\$P\" 2>/dev/null; \
for i in 1 2 3 4 5 6 7 8 9 10; do pgrep -u \$USER -f \"\$P\" >/dev/null || exit 0; sleep 1; done; \
pkill -9 -u \$USER -f \"\$P\" 2>/dev/null; sleep 1; true" || true
    done
}

exp1_cleanup() {
    kill_dht_hosts
    cleanup_all_nodes
}

# Jobs must never share nodes (Redis start-up and cleanup_all_nodes kill ALL of
# your redis-server/apptainer there). Checked BEFORE any trap is installed, so
# refusing to start kills nothing.
nodes_free_of_my_jobs() {
    local pat='[r]edis-server|[f]lower_server[.]py|[o]nline_sac_worker[.]py|[e]xp1_worker[.]py|[d]ht_ring_host[.]py|[e]xp1_dht_host[.]py'
    local n out busy=0
    for n in $(sort -u "${PBS_NODEFILE:-/dev/null}" 2>/dev/null); do
        out=$(on_node "$n" "pgrep -u \$USER -af '$pat' 2>/dev/null | head -3; true")
        if [[ $? -eq 255 ]]; then
            echo "  WARNING: could not check $n (ssh refused)" | tee -a "$RUNNER_LOG"
        elif [[ -n "$out" ]]; then
            echo "FATAL: ${n%%.*} already runs your experiment processes (another job?):" \
                | tee -a "$RUNNER_LOG"
            echo "$out" | sed 's/^/    /' | tee -a "$RUNNER_LOG"
            busy=1
        fi
    done
    if [[ $busy -eq 1 ]]; then
        echo "This job would kill them. Resubmit with WORKING_NODES that avoid these" \
             "nodes (or, if they are leftovers of a dead job, clean them first)." \
            | tee -a "$RUNNER_LOG"
        return 1
    fi
}
nodes_free_of_my_jobs || exit 1

trap 'exp1_cleanup; exit 130' INT
trap 'exp1_cleanup; exit 143' TERM
trap 'exp1_cleanup' EXIT

# wait_launched <marker> <nodes array> <logs array> <cmds array>: wait until
# every log shows <marker>; relaunch a host whose SSH died first (up to 4x).
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
            kill -0 "${DHT_PIDS[$h]}" 2>/dev/null && continue
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
            DHT_PIDS[$h]=$LAUNCH_PID
        done
        [[ $pending -eq 0 ]] && return 0
        sleep 1
    done
    echo "!!! hosts did not start within 120s" | tee -a "$RUNNER_LOG"
    return 1
}

# ssh_mesh: the bundle copies are pulled node->node (dst from src, replica
# nodes from src, dst from replica nodes), so every ordered pair must work.
ssh_mesh() {
    local a b rc=0
    for a in "${HOSTS[@]}"; do
        for b in "${HOSTS[@]}"; do
            [[ "$a" == "$b" ]] && continue
            if on_node "$a" "for i in 1 2 3; do ssh -n -o BatchMode=yes -o ConnectTimeout=10 \
-o StrictHostKeyChecking=no $b true && exit 0; sleep 5; done; exit 1"; then
                echo "  ssh ${a%%.*} -> ${b%%.*}: ok" | tee -a "$RUNNER_LOG"
            else
                echo "  ssh ${a%%.*} -> ${b%%.*}: FAIL" | tee -a "$RUNNER_LOG"
                rc=1
            fi
        done
    done
    return $rc
}

# net_check: UDP ring ports + TCP control port between all nodes, Redis on the
# server. 23000/23999 are only reported (ends of the admin's UDP range).
net_check() {
    local last=$((EXP1_DHT_BASE_PORT + EXP1_DHT_NODES_PER_HOST - 1))
    local udp="$EXP1_DHT_BASE_PORT,$last" info="23000,23999"
    local h o rc=0 targets
    local -a nodes=() logs=() cmds=()
    DHT_PIDS=()
    for h in 0 1 2; do
        nodes[$h]="${HOSTS[$h]}"
        logs[$h]="$RUN_LOG_DIR/netcheck_h${h}.log"
        cmds[$h]="python3 -u $HERE/net_check.py serve --udp-ports $udp --udp-info $info \
--tcp-port $EXP1_CTRL_PORT --seconds 300"
        : > "${logs[$h]}"
        launch_host "${nodes[$h]}" "${logs[$h]}" "${cmds[$h]}"
        DHT_PIDS[$h]=$LAUNCH_PID
    done
    if ! wait_launched "net_check serving" nodes logs cmds; then
        kill_dht_hosts
        return 1
    fi
    for h in 0 1 2; do
        targets=""
        for o in 0 1 2; do
            [[ $o -eq $h ]] && continue
            if [[ $o -eq 0 ]]; then
                targets="$targets --target ${HOSTS[$o]}:$EXP1_CTRL_PORT,$REDIS_PORT"
            else
                targets="$targets --target ${HOSTS[$o]}:$EXP1_CTRL_PORT"
            fi
        done
        on_node "${HOSTS[$h]}" "source $CONDA_BASE/bin/activate $CONDA_ENV && \
python3 -u $HERE/net_check.py probe --udp-ports $udp --udp-info $info $targets" \
            2>&1 | tee -a "$RUNNER_LOG"
        [[ ${PIPESTATUS[0]} -ne 0 ]] && rc=1
    done
    kill_dht_hosts
    return $rc
}

# start_dht: EXP1_DHT_NODES_PER_HOST ring nodes on each of the 3 nodes.
start_dht() {
    local m="$EXP1_DHT_NODES_PER_HOST" h
    local -a nodes=() logs=() cmds=()
    DHT_PIDS=()
    for h in 0 1 2; do
        nodes[$h]="${HOSTS[$h]}"
        logs[$h]="$RUN_LOG_DIR/dht_h${h}.log"
        cmds[$h]="python3 -u $HERE/exp1_dht_host.py --host-index $h --n-local $m \
--id-offset $((h * m)) --base-port $EXP1_DHT_BASE_PORT --ctrl-port $EXP1_CTRL_PORT \
--bootstrap-host ${HOSTS[0]} --bootstrap-port $EXP1_DHT_BASE_PORT \
--ksize $EXP1_DHT_KSIZE --alpha $EXP1_DHT_ALPHA --rpc-timeout $EXP1_DHT_RPC_TIMEOUT \
--max-lifetime $EXP1_MAX_LIFETIME"
        : > "${logs[$h]}"
        launch_host "${nodes[$h]}" "${logs[$h]}" "${cmds[$h]}"
        DHT_PIDS[$h]=$LAUNCH_PID
    done
    wait_launched "control port" nodes logs cmds
}

# --------------------------------------------------------------------------- #
pick_alive_nodes || exit 1
HOSTS=("$SERVER_NODE" "$CLIENT_NODE_1" "$CLIENT_NODE_2")
start_redis_on_server || exit 1

source "$CONDA_BASE/bin/activate" "$CONDA_ENV"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$CLUSTER_ROOT:${PYTHONPATH:-}"
export SERVER_NODE CLIENT_NODE_1 CLIENT_NODE_2
export REDIS_HOST="$SERVER_NODE"
echo ">>> Exp1 conditions=(${CONDS[*]}) robots=$NUM_CLIENTS rounds=$TOTAL_FL_ROUNDS \
migrations=($MIGRATION_ROUNDS) kill_points=$EXP1_KILL_POINTS replicas=$EXP1_REPLICAS" \
    | tee -a "$RUNNER_LOG"

# Preflight: container image + pylibs, python libs on every node, ssh mesh.
for f in "$IMG_DIR/robot.sif" "$IMG_DIR/pylibs" "$TASK2_PYLIBS2"; do
    if [[ ! -e "$f" ]]; then
        echo "FATAL: missing $f (task2 container setup)" | tee -a "$RUNNER_LOG"
        exit 1
    fi
done
for n in "${HOSTS[@]}"; do
    if ! on_node "$n" "source $CONDA_BASE/bin/activate $CONDA_ENV && python3 -c 'import kademlia, redis' \
&& command -v apptainer >/dev/null && command -v rsync >/dev/null"; then
        echo "FATAL: kademlia/redis-py/apptainer/rsync missing on $n" | tee -a "$RUNNER_LOG"
        exit 1
    fi
done
echo ">>> ssh mesh check (bundle copies are pulled node to node)" | tee -a "$RUNNER_LOG"
if ! ssh_mesh; then
    echo "FATAL: some nodes cannot ssh to each other (see FAIL lines)." | tee -a "$RUNNER_LOG"
    exit 1
fi
kill_dht_hosts   # stale helpers from an earlier Exp1 job on these nodes

if [[ $USES_DHT -eq 1 ]]; then
    echo ">>> network check: DHT UDP ports + TCP control/Redis ports" | tee -a "$RUNNER_LOG"
    if ! net_check; then
        echo "FATAL: nodes cannot reach each other on the ports above (BLOCKED lines)." \
            | tee -a "$RUNNER_LOG"
        exit 1
    fi
    echo ">>> network check passed; starting the DHT ring" | tee -a "$RUNNER_LOG"
    if ! start_dht; then
        echo "FATAL: DHT ring did not start" | tee -a "$RUNNER_LOG"
        exit 1
    fi
    export EXP1_DHT_HOSTS="${HOSTS[0]}:$EXP1_CTRL_PORT,${HOSTS[1]}:$EXP1_CTRL_PORT,${HOSTS[2]}:$EXP1_CTRL_PORT"
fi

# between_conditions <cond>: `all` mode reuses the nodes, so nothing of the
# previous condition may survive (its workers, its Flower server on the same
# port); its /tmp bundles are freed too. Patterns are anchored at python3 so
# they never match apptainer (killing apptainer orphans its FUSE mounts).
between_conditions() {
    local c="$1" n
    local wpat="^python3 [^ ]*exp1_worker[.]py .* --run-tag exp1_${c}\$"
    local fpat='^python3 /cluster_app/task2/flower_server[.]py$'
    for n in "${HOSTS[@]}"; do
        on_node "$n" "pkill -9 -u \$USER -f '$wpat' 2>/dev/null; \
pkill -9 -u \$USER -f '$fpat' 2>/dev/null; \
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
pgrep -u \$USER -f '$wpat' >/dev/null || pgrep -u \$USER -f '$fpat' >/dev/null || break; \
sleep 1; done; rm -rf /tmp/swiftbot_exp1_${c}; true" || true
    done
    sleep 5
}

RC=0
for c in "${CONDS[@]}"; do
    # CONDITION names the worker/checkpoint namespace (/tmp/swiftbot_exp1_<c>),
    # the --run-tag the runner's SIGKILL matches on and the DHT key namespace.
    export EXP1_CONDITION="$c" CONDITION="exp1_${c}"
    if [[ ${#CONDS[@]} -gt 1 ]]; then
        export RUN_LOG_DIR="$BASE_LOG_DIR/$c"
        mkdir -p "$RUN_LOG_DIR"
    fi
    export RESULTS_DIR="$RESULTS_ROOT/exp1_source_failure/$c/$STAMP"
    mkdir -p "$RESULTS_DIR"
    echo ">>> [$c] launching Exp1 runner; results -> $RESULTS_DIR" | tee -a "$RUNNER_LOG"
    python3 -u "$HERE/runner.py" 2>&1 | tee -a "$RUNNER_LOG"
    rc=${PIPESTATUS[0]}
    echo ">>> [$c] Exp1 runner exited rc=$rc. Results: $RESULTS_DIR" | tee -a "$RUNNER_LOG"
    [[ $rc -ne 0 ]] && RC=$rc
    [[ ${#CONDS[@]} -gt 1 ]] && between_conditions "$c"
done
echo ">>> Exp1 finished: conditions=(${CONDS[*]}) rc=$RC" | tee -a "$RUNNER_LOG"
exit "$RC"
