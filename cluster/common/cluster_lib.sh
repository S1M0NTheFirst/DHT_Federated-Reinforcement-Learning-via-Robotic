#!/bin/bash
# Bash helpers shared by every run_X.sh.
# All functions print to stderr and write structured progress to $RUN_LOG_DIR.

set -uo pipefail

# Filled by setup_run_dirs:
#   RUN_LOG_DIR  — cluster/logs/<cond>/<jobid_or_timestamp>/
#   SERVER_LOG   — RUN_LOG_DIR/server.log
#   RUNNER_LOG   — RUN_LOG_DIR/runner.log
#   CLIENT_LOG_C1, CLIENT_LOG_C2 — per client-node log
RUN_LOG_DIR=""
SERVER_LOG=""
RUNNER_LOG=""
CLIENT_LOG_C1=""
CLIENT_LOG_C2=""
SERVER_NODE=""
CLIENT_NODE_1=""
CLIENT_NODE_2=""
ALIVE_NODES=()

setup_run_dirs() {
    local cond="$1"
    local stamp="${PBS_JOBID:-$(date +%Y%m%d_%H%M%S)}"
    # IMPORTANT: export — the Python runner reads RUN_LOG_DIR from os.environ.
    export RUN_LOG_DIR="${LOG_ROOT}/${cond}/${stamp}"
    mkdir -p "$RUN_LOG_DIR"
    export SERVER_LOG="$RUN_LOG_DIR/server.log"
    export RUNNER_LOG="$RUN_LOG_DIR/runner.log"
    export CLIENT_LOG_C1="$RUN_LOG_DIR/client_node1.log"
    export CLIENT_LOG_C2="$RUN_LOG_DIR/client_node2.log"
    : > "$SERVER_LOG"; : > "$RUNNER_LOG"
    : > "$CLIENT_LOG_C1"; : > "$CLIENT_LOG_C2"
    echo ">>> Logs: $RUN_LOG_DIR" | tee -a "$RUNNER_LOG"
    echo ">>> Tail in real time:" | tee -a "$RUNNER_LOG"
    echo "       tail -F $SERVER_LOG" | tee -a "$RUNNER_LOG"
    echo "       tail -F $CLIENT_LOG_C1 $CLIENT_LOG_C2" | tee -a "$RUNNER_LOG"
    echo "       tail -F $RUN_LOG_DIR/robot_*.log" | tee -a "$RUNNER_LOG"
}

# Test SSH on every assigned node, drop dead ones, and pick:
#   SERVER_NODE     = first alive
#   CLIENT_NODE_1   = second alive
#   CLIENT_NODE_2   = third alive
# Aborts if fewer than 3 alive nodes.
pick_alive_nodes() {
    local raw=($(cat "$PBS_NODEFILE" | sort | uniq))
    echo "Testing SSH on assigned nodes: ${raw[*]}" | tee -a "$RUNNER_LOG"
    ALIVE_NODES=()
    for n in "${raw[@]}"; do
        if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes \
               "$n" "echo alive" >/dev/null 2>&1; then
            ALIVE_NODES+=("$n")
            echo "  $n ALIVE" | tee -a "$RUNNER_LOG"
        else
            echo "  $n DEAD — dropped" | tee -a "$RUNNER_LOG"
        fi
    done
    if [[ ${#ALIVE_NODES[@]} -lt 3 ]]; then
        echo "FATAL: need ≥3 alive nodes (1 server + 2 clients), got ${#ALIVE_NODES[@]}" \
            | tee -a "$RUNNER_LOG"
        return 1
    fi
    export SERVER_NODE="${ALIVE_NODES[0]}"
    export CLIENT_NODE_1="${ALIVE_NODES[1]}"
    export CLIENT_NODE_2="${ALIVE_NODES[2]}"
    echo "Server:    $SERVER_NODE" | tee -a "$RUNNER_LOG"
    echo "Client #1: $CLIENT_NODE_1" | tee -a "$RUNNER_LOG"
    echo "Client #2: $CLIENT_NODE_2" | tee -a "$RUNNER_LOG"
}

# Start an apptainer-hosted Redis on the server node, listening on REDIS_PORT.
# Logs to $SERVER_LOG. Caller waits for redis-cli ping to return PONG.
start_redis_on_server() {
    echo ">>> Starting Redis on $SERVER_NODE:$REDIS_PORT" | tee -a "$RUNNER_LOG"
    # Apptainer Redis: pull from docker hub via apptainer's docker bootstrap.
    # We reuse baseline.sif which has redis-cli; the redis-server binary is
    # installed via the bootstrap image's apt pkg (we'll need it). To avoid
    # depending on that, run native redis-server on the head node — every
    # cluster node has redis-server installed as part of the base image
    # (verified by `redis-cli ping` in run_fldht.sh).
    ssh -n "$SERVER_NODE" "pkill -9 -u \$USER -x redis-server || true; sleep 1"
    # Note the escaped $! — we want the pid of the redis-server backgrounded
    # in the *remote* shell, not in this local shell.
    ssh -n -f "$SERVER_NODE" "
        nohup redis-server --bind 0.0.0.0 --port $REDIS_PORT --protected-mode no \
              --maxmemory 4gb --maxmemory-policy allkeys-lru \
              > $RUN_LOG_DIR/redis.log 2>&1 < /dev/null &
        echo \$! > $RUN_LOG_DIR/redis.pid
    "
    # Wait for it to be reachable. Try `redis-cli` if installed on the head
    # node; otherwise fall back to a Python ping (conda env has the redis
    # package). This avoids a hard dependency on the redis CLI being in PATH.
    local i
    for i in {1..30}; do
        if command -v redis-cli >/dev/null 2>&1; then
            if redis-cli -h "$SERVER_NODE" -p "$REDIS_PORT" ping 2>/dev/null | grep -q PONG; then
                echo ">>> Redis ready (via redis-cli)" | tee -a "$RUNNER_LOG"
                return 0
            fi
        else
            if python3 -c "import redis; r=redis.Redis(host='$SERVER_NODE',port=$REDIS_PORT,socket_connect_timeout=1); print(r.ping())" 2>/dev/null | grep -q True; then
                echo ">>> Redis ready (via python3)" | tee -a "$RUNNER_LOG"
                return 0
            fi
        fi
        sleep 1
    done
    echo "FATAL: Redis never came up on $SERVER_NODE:$REDIS_PORT" | tee -a "$RUNNER_LOG"
    return 1
}

# Kill every apptainer instance and python worker we may have started, on
# every alive node, and flush Redis. Idempotent — safe to call from EXIT trap
# even if pick_alive_nodes never ran.
cleanup_all_nodes() {
    echo ">>> Cleanup: stopping apptainer instances + workers on all nodes" \
        | tee -a "$RUNNER_LOG"
    local nodes=("${ALIVE_NODES[@]:-}")
    if [[ ${#nodes[@]} -eq 0 ]]; then
        nodes=($(cat "${PBS_NODEFILE:-/dev/null}" 2>/dev/null | sort | uniq))
    fi
    for n in "${nodes[@]}"; do
        [[ -z "$n" ]] && continue
        ssh -n -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$n" "
            apptainer instance list 2>/dev/null | awk 'NR>1 {print \$1}' | while read inst; do
                apptainer instance stop \"\$inst\" 2>/dev/null || true
            done
            pkill -9 -u \$USER -x apptainer 2>/dev/null || true
            pkill -9 -u \$USER -f worker_robot_client 2>/dev/null || true
            pkill -9 -u \$USER -f worker_random_client 2>/dev/null || true
            pkill -9 -u \$USER -x redis-server 2>/dev/null || true
            rm -rf /tmp/swiftbot_*  2>/dev/null || true
        " 2>/dev/null || true
    done
    echo ">>> Cleanup complete" | tee -a "$RUNNER_LOG"
}
