#!/bin/bash
# Submit a condition onto 3 currently-free, confirmed-working nodes.
#
# Why this exists: this MOAB install does not honor a generic
# `nodes=3:ppn=8` request as 3 DISTINCT hosts — it consolidates all 3 onto
# one host, and pick_alive_nodes then aborts ("need >=3 alive nodes"). So we
# must name 3 distinct nodes. This picks them at submit time from the set of
# nodes that are BOTH free (per the scheduler) AND known-good, then submits
# with `msub -l nodes=...`, which overrides the #MSUB directive in the script.
#
# Usage:
#   ./submit_free.sh ../condition_D_criu_warm/run_D.sh
#   WORKING_NODES="n001 n005 n016" ./submit_free.sh ../condition_E_cold_restart/run_E.sh
#   NEED=3 ./submit_free.sh ../condition_D_criu_warm/run_D.sh
#
# Run this on the cluster HEAD node (where msub/pbsnodes live).
set -uo pipefail

RUN_SCRIPT="${1:?usage: submit_free.sh <run_X.sh> [extra msub args...]}"
shift || true
DOMAIN=".cluster.pssclabs.com"
NEED="${NEED:-3}"

# Confirmed-good nodes, in PREFERENCE order (fast/healthy first). n033 is slow
# (BLAS thrash); n004/n020/n034 have failed in-job. Override with $WORKING_NODES.
WORKING_NODES="${WORKING_NODES:-n001 n005 n016 n021 n023 n024 n027 n035}"

# Free nodes per the scheduler. First whitespace token per line is the name;
# strip any domain so we compare on short names.
mapfile -t FREE_RAW < <(pbsnodes -l free 2>/dev/null | awk '{print $1}')
declare -A IS_FREE=()
for f in "${FREE_RAW[@]}"; do
    [[ -z "$f" ]] && continue
    IS_FREE["${f%%.*}"]=1
done

# Walk the working set in preference order, keep the ones the scheduler says
# are free, until we have NEED of them.
PICKED=()
for w in $WORKING_NODES; do
    short="${w%%.*}"
    if [[ -n "${IS_FREE[$short]:-}" ]]; then
        PICKED+=("${short}${DOMAIN}")
        [[ ${#PICKED[@]} -ge $NEED ]] && break
    fi
done

if [[ ${#PICKED[@]} -lt $NEED ]]; then
    echo "ERROR: only found ${#PICKED[@]} free working node(s); need $NEED." >&2
    echo "  working set : $WORKING_NODES" >&2
    echo "  free now    : ${FREE_RAW[*]:-<none>}" >&2
    echo "Wait for a node to free up, or widen WORKING_NODES=\"...\"." >&2
    exit 1
fi

# Build the +-joined spec: n1:ppn=8+n2:ppn=8+n3:ppn=8
SPEC=""
for p in "${PICKED[@]}"; do
    SPEC="${SPEC:+$SPEC+}${p}:ppn=8"
done

echo ">>> Submitting $RUN_SCRIPT"
echo ">>> nodes: $SPEC"
msub -l nodes="$SPEC" "$@" "$RUN_SCRIPT"
