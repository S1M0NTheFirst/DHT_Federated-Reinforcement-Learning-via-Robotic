#!/bin/bash
#MSUB -N NodeProbe
#MSUB -W group_list=hpc2-coe-users
#MSUB -l walltime=00:15:00
#MSUB -l nodes=n005.cluster.pssclabs.com:ppn=1+n016.cluster.pssclabs.com:ppn=1+n021.cluster.pssclabs.com:ppn=1+n023.cluster.pssclabs.com:ppn=1+n024.cluster.pssclabs.com:ppn=1+n027.cluster.pssclabs.com:ppn=1+n030.cluster.pssclabs.com:ppn=1+n031.cluster.pssclabs.com:ppn=1+n034.cluster.pssclabs.com:ppn=1+n036.cluster.pssclabs.com:ppn=1
#MSUB -j oe
#
# Node-health probe. Submit with:  msub cluster/tools/probe_nodes.sh
# Edit the `#MSUB -l nodes=` line above to whatever candidate nodes you want to
# test (use `pbsnodes -l free` first). ppn=1 keeps it lightweight + schedulable.
#
# For EACH allocated node it tests the three things the experiment actually
# relies on, from inside the allocation (where intra-job SSH is authorized):
#   1. PLAIN   — `ssh node echo ok`               (basic reachability)
#   2. LAUNCH  — `ssh -o ControlMaster=auto ... node cmd`  (exactly how
#                launch_robot starts a worker — THE decisive test; n020-type
#                broken nodes hang here and their robots never start)
#   3. MASTER  — `ssh -M -f -N node`              (persistent master; many
#                nodes hang here but still pass LAUNCH, so this is informational)
# Then a node-to-node MESH test (needed for cross-node checkpoint rsync).
#
# Output goes to the job's .o<id> file in the submit dir. Read it and pick
# nodes that are PLAIN-OK + LAUNCH-OK for clients; MASTER status doesn't matter.

set -uo pipefail
NODES=$(sort -u "$PBS_NODEFILE" | sed 's/\..*//')   # short names, unique
SELF=$(hostname -s)
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=8"
DOM=".cluster.pssclabs.com"

echo "============================================================"
echo "Node-health probe   (mother node: $SELF)"
echo "Candidates: $NODES"
echo "============================================================"
echo
printf "%-8s %-12s %-12s %-16s\n" "NODE" "PLAIN" "LAUNCH" "MASTER(info)"
printf "%-8s %-12s %-12s %-16s\n" "----" "-----" "------" "------------"

GOOD=""
for n in $NODES; do
    if [ "$n" = "$SELF" ]; then
        printf "%-8s %-12s %-12s %-16s\n" "$n" "SELF" "SELF" "-"
        GOOD="$GOOD $n"
        continue
    fi
    host="${n}${DOM}"

    # 1. plain reachability
    if timeout 12 $SSH "$host" "echo ok" >/dev/null 2>&1; then
        plain="OK"
    else
        plain="FAIL"
    fi

    # 2. LAUNCH path — same options launch_robot uses (ControlMaster=auto +
    #    a command). This is what determines whether robots can start.
    ctl="/tmp/probe-mux-$$-${n}"
    if timeout 15 $SSH -o ControlMaster=auto -o ControlPath="$ctl" \
            -o ControlPersist=10 -n "$host" "echo ok" >/dev/null 2>&1; then
        launch="OK"
    else
        launch="HANG/FAIL"
    fi

    # 3. MASTER mode (informational — many nodes hang here but still LAUNCH-OK)
    ctl2="/tmp/probe-mux2-$$-${n}"
    if timeout 12 $SSH -o ControlMaster=yes -o ControlPath="$ctl2" \
            -o ControlPersist=5 -M -f -N "$host" >/dev/null 2>&1; then
        master="OK"
        ssh -o ControlPath="$ctl2" -O exit "$host" 2>/dev/null || true
    else
        master="HANG/FAIL"
    fi

    printf "%-8s %-12s %-12s %-16s\n" "$n" "$plain" "$launch" "$master"
    if [ "$plain" = "OK" ] && [ "$launch" = "OK" ]; then
        GOOD="$GOOD $n"
    fi
done

echo
echo "Usable client nodes (PLAIN-OK + LAUNCH-OK):${GOOD}"
echo

# --- Node-to-node mesh (cross-node rsync path for migrations) ----------------
echo "============================================================"
echo "MESH  (src -> dst plain ssh; needed for checkpoint transfer)"
echo "Only testing among usable nodes."
echo "============================================================"
for src in $GOOD; do
    [ "$src" = "$SELF" ] && continue   # mother can't ssh to itself on this cluster
    line="$src ->"
    for dst in $GOOD; do
        [ "$src" = "$dst" ] && continue
        if timeout 15 $SSH "${src}${DOM}" \
                "$SSH ${dst}${DOM} 'echo ok'" >/dev/null 2>&1; then
            line="$line ${dst}:OK"
        else
            line="$line ${dst}:FAIL"
        fi
    done
    echo "$line"
done
echo
echo "Probe complete. Pick LAUNCH-OK nodes whose MESH is all-OK for clients."
