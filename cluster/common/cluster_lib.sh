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
    # Diagnostics: show exactly what PBS gave us before any sort/uniq.
    echo ">>> PBS_NODEFILE=$PBS_NODEFILE" | tee -a "$RUNNER_LOG"
    echo ">>> raw PBS_NODEFILE contents:" | tee -a "$RUNNER_LOG"
    cat "$PBS_NODEFILE" | sed 's/^/    /' | tee -a "$RUNNER_LOG"
    local self_fqdn=$(hostname -f)
    local self_short=$(hostname -s)
    echo ">>> self hostname: $self_fqdn (short: $self_short)" | tee -a "$RUNNER_LOG"
    export SELF_NODE="$self_fqdn"

    local raw=($(cat "$PBS_NODEFILE" | sort | uniq))
    echo "Testing SSH on assigned nodes: ${raw[*]}" | tee -a "$RUNNER_LOG"
    ALIVE_NODES=()
    for n in "${raw[@]}"; do
        # The local node can't SSH to itself on this cluster (sshd closes the
        # connection). Trust ourselves — we know we're running.
        if [[ "$n" == "$self_fqdn" || "$n" == "$self_short" || \
              "$n" == "${self_short}."* ]]; then
            ALIVE_NODES+=("$n")
            echo "  $n LOCAL (self) — included" | tee -a "$RUNNER_LOG"
            continue
        fi
        # Retry the probe 4 times with backoff. A single failure is usually
        # transient sshd MaxStartups rate-limiting after a previous job, not
        # a dead node. Without this, one rate-limited probe fails the whole
        # submission with "FATAL: need ≥3 alive nodes".
        local ssh_err alive="no" attempt
        for attempt in 1 2 3 4; do
            ssh_err=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
                          "$n" "echo alive" 2>&1)
            if [[ "$ssh_err" == "alive" ]]; then
                alive="yes"
                break
            fi
            echo "  $n probe attempt $attempt failed: ${ssh_err:-<empty>}" \
                | tee -a "$RUNNER_LOG"
            sleep $((attempt * 5))
        done
        if [[ "$alive" == "yes" ]]; then
            ALIVE_NODES+=("$n")
            echo "  $n ALIVE" | tee -a "$RUNNER_LOG"
        else
            echo "  $n DEAD — dropped after 4 attempts (last: ${ssh_err:-<empty>})" \
                | tee -a "$RUNNER_LOG"
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

# Start Redis on the server node as a TRACKED bash background SSH process.
#
# IMPORTANT — do not revert to `ssh -f` / `nohup ... &` patterns. Those
# double-detach the remote redis-server from the job's process tree, which
# is what caused the orphaned-container incident the cluster admin asked us
# to fix. The new pattern:
#
#   - SSH runs in the FOREGROUND of its own bash subshell.
#   - bash backgrounds the SSH (`&`), so SSH is a child of the run_X.sh shell.
#   - The remote command uses `exec redis-server …` so redis-server replaces
#     the bash that sshd spawned. When SSH dies (because run_X.sh exits or is
#     killed by MOAB), sshd sends SIGHUP to redis-server, killing it.
#   - REDIS_SSH_PID is exported so cleanup_all_nodes can `kill` it explicitly.
start_redis_on_server() {
    local self_fqdn=$(hostname -f)
    local self_short=$(hostname -s)
    local is_local="no"
    if [[ "$SERVER_NODE" == "$self_fqdn" || "$SERVER_NODE" == "$self_short" || \
          "$SERVER_NODE" == "${self_short}."* ]]; then
        is_local="yes"
    fi
    echo ">>> Starting Redis on $SERVER_NODE:$REDIS_PORT (local=$is_local)" \
        | tee -a "$RUNNER_LOG"

    # Pre-clean any stale redis on the server node.
    if [[ "$is_local" == "yes" ]]; then
        pkill -9 -u "$USER" -x redis-server 2>/dev/null || true
    else
        ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$SERVER_NODE" \
            "pkill -9 -u \$USER -x redis-server 2>/dev/null || true; sleep 1" \
            || true
    fi
    sleep 1

    if [[ "$is_local" == "yes" ]]; then
        # Local launch — backgrounded by this bash shell so it dies with the
        # job. No SSH needed (can't ssh to self on this cluster anyway).
        # Use full path to redis-server because this function is called from
        # run_*.sh BEFORE `conda activate`, so redis-server isn't on PATH.
        # Note: base env's binaries live at $CONDA_BASE/bin, while non-base
        # envs are at $CONDA_BASE/envs/<name>/bin.
        local env_bin
        if [[ "$CONDA_ENV" == "base" ]]; then
            env_bin="$CONDA_BASE/bin"
        else
            env_bin="$CONDA_BASE/envs/$CONDA_ENV/bin"
        fi
        local redis_bin="$env_bin/redis-server"
        "$redis_bin" --bind 0.0.0.0 --port "$REDIS_PORT" --protected-mode no \
                     --maxmemory 4gb --maxmemory-policy allkeys-lru \
            > "$RUN_LOG_DIR/redis.log" 2>&1 < /dev/null &
        export REDIS_SSH_PID=$!
        echo "$REDIS_SSH_PID" > "$RUN_LOG_DIR/redis_ssh.pid"
        echo ">>> Redis local PID=$REDIS_SSH_PID (bin=$redis_bin)" | tee -a "$RUNNER_LOG"
    else
        # Remote launch via tracked SSH child (foreground+exec → sshd HUPs
        # redis-server when SSH closes).
        # Non-interactive SSH does NOT source ~/.bashrc, so redis-server (in
        # the conda env) isn't on PATH. Source conda explicitly before exec.
        ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
            -n "$SERVER_NODE" \
            "source $CONDA_BASE/bin/activate $CONDA_ENV && \
             exec redis-server --bind 0.0.0.0 --port $REDIS_PORT --protected-mode no \
                              --maxmemory 4gb --maxmemory-policy allkeys-lru" \
            > "$RUN_LOG_DIR/redis.log" 2>&1 < /dev/null &
        export REDIS_SSH_PID=$!
        echo "$REDIS_SSH_PID" > "$RUN_LOG_DIR/redis_ssh.pid"
        echo ">>> Redis SSH child PID=$REDIS_SSH_PID" | tee -a "$RUNNER_LOG"
    fi

    # Wait for redis to be reachable. Use full paths because conda is not yet
    # activated when this function runs. Base env path differs from non-base.
    local _env_bin
    if [[ "$CONDA_ENV" == "base" ]]; then
        _env_bin="$CONDA_BASE/bin"
    else
        _env_bin="$CONDA_BASE/envs/$CONDA_ENV/bin"
    fi
    local redis_cli="$_env_bin/redis-cli"
    local py_bin="$_env_bin/python3"
    local i
    for i in {1..30}; do
        if [[ -x "$redis_cli" ]]; then
            if "$redis_cli" -h "$SERVER_NODE" -p "$REDIS_PORT" ping 2>/dev/null | grep -q PONG; then
                echo ">>> Redis ready (via redis-cli)" | tee -a "$RUNNER_LOG"
                return 0
            fi
        elif [[ -x "$py_bin" ]]; then
            if "$py_bin" -c "import redis; r=redis.Redis(host='$SERVER_NODE',port=$REDIS_PORT,socket_connect_timeout=1); print(r.ping())" 2>/dev/null | grep -q True; then
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
    echo ">>> Cleanup: tearing down all SSH children + remote processes" \
        | tee -a "$RUNNER_LOG" 2>/dev/null
    # 1. Kill local tracked SSH children FIRST. As each one dies, sshd on the
    # remote node sends SIGHUP to its child (apptainer-exec / redis-server),
    # which terminates cleanly. This is the orderly shutdown path.
    if [[ -n "${REDIS_SSH_PID:-}" ]]; then
        kill "$REDIS_SSH_PID" 2>/dev/null || true
    fi
    # Any other SSH children of this script (e.g. backgrounded probes).
    pkill -P $$ -f '^ssh ' 2>/dev/null || true

    # Give sshd a moment to deliver SIGHUP to remote commands.
    sleep 2

    # 2. Belt-and-suspenders: pkill anything that survived the HUP on each
    # node. We do this for every node we know about, not just nodes in our
    # job — covers the case where a previous job on the same nodes left
    # debris that this job inherited.
    local nodes=("${ALIVE_NODES[@]:-}")
    if [[ ${#nodes[@]} -eq 0 ]]; then
        nodes=($(cat "${PBS_NODEFILE:-/dev/null}" 2>/dev/null | sort | uniq))
    fi
    local self_fqdn=$(hostname -f)
    local self_short=$(hostname -s)
    local cleanup_script='
        apptainer instance list 2>/dev/null | awk "NR>1 {print \$1}" | while read inst; do
            apptainer instance stop -s KILL "$inst" 2>/dev/null || true
        done
        pkill -TERM -u $USER -f worker_robot_client 2>/dev/null || true
        pkill -TERM -u $USER -f worker_random_client 2>/dev/null || true
        pkill -TERM -u $USER -f worker_app_checkpoint 2>/dev/null || true
        pkill -TERM -u $USER -f flower_server.py 2>/dev/null || true
        sleep 2
        pkill -9 -u $USER -f worker_robot_client 2>/dev/null || true
        pkill -9 -u $USER -f worker_random_client 2>/dev/null || true
        pkill -9 -u $USER -f worker_app_checkpoint 2>/dev/null || true
        pkill -9 -u $USER -f flower_server.py 2>/dev/null || true
        pkill -9 -u $USER -x apptainer 2>/dev/null || true
        pkill -9 -u $USER -x redis-server 2>/dev/null || true
        # Apptainer mounts the SIF via a squashfuse (FUSE) helper, and overlay
        # via fuse-overlayfs. We launch with `apptainer exec`, so a SIGKILLed
        # apptainer ORPHANS these helpers: the image stays mounted and pinned
        # in RAM outside the scheduler (the n017 leak the admin flagged).
        # Unmount each FUSE mount, then kill any helper that survives.
        for mp in $(grep -E "squashfuse|fuse-overlayfs|fuse.squash" /proc/mounts 2>/dev/null | awk "{print \$2}"); do
            fusermount -u "$mp" 2>/dev/null || fusermount -uz "$mp" 2>/dev/null || true
        done
        pkill -TERM -u $USER -f "squashfuse|fuse-overlayfs" 2>/dev/null || true
        sleep 1
        pkill -9 -u $USER -f "squashfuse|fuse-overlayfs" 2>/dev/null || true
        rm -rf /tmp/swiftbot_* 2>/dev/null || true
    '
    for n in "${nodes[@]}"; do
        [[ -z "$n" ]] && continue
        # Self can't SSH to itself on this cluster — run cleanup locally.
        if [[ "$n" == "$self_fqdn" || "$n" == "$self_short" || \
              "$n" == "${self_short}."* ]]; then
            bash -c "$cleanup_script" 2>/dev/null || true
        else
            ssh -n -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
                -o BatchMode=yes "$n" "$cleanup_script" 2>/dev/null || true
        fi
    done

    # 3. Final verification — log what's left so the admin doesn't have to ask.
    echo ">>> Post-cleanup verification:" | tee -a "$RUNNER_LOG" 2>/dev/null
    for n in "${nodes[@]}"; do
        [[ -z "$n" ]] && continue
        local leftover
        if [[ "$n" == "$self_fqdn" || "$n" == "$self_short" || \
              "$n" == "${self_short}."* ]]; then
            leftover=$(pgrep -u "$USER" -af 'apptainer|worker_|redis-server|squashfuse|fuse-overlayfs' \
                2>/dev/null | head -5)
        else
            leftover=$(ssh -n -o ConnectTimeout=5 -o BatchMode=yes "$n" \
                "pgrep -u \$USER -af 'apptainer|worker_|redis-server|squashfuse|fuse-overlayfs' 2>/dev/null | head -5" \
                2>/dev/null)
        fi
        if [[ -n "$leftover" ]]; then
            echo "  $n: STILL RUNNING:" | tee -a "$RUNNER_LOG" 2>/dev/null
            echo "$leftover" | tee -a "$RUNNER_LOG" 2>/dev/null
        else
            echo "  $n: clean" | tee -a "$RUNNER_LOG" 2>/dev/null
        fi
    done
    echo ">>> Cleanup complete" | tee -a "$RUNNER_LOG" 2>/dev/null

    # Final failsafe: kill any remaining ssh process owned by this user
    # whose ControlPath is our PBS-job mux dir. This catches daemonised
    # masters that ControlPersist hasn't yet timed out. Without this MOAB
    # can keep the job "Running" for several minutes after the runner is
    # actually done.
    pkill -KILL -u "$USER" -f "ssh-mux-${PBS_JOBID:-NOJOB}" 2>/dev/null || true
    echo ">>> Bash wrapper exiting." | tee -a "$RUNNER_LOG" 2>/dev/null
}
