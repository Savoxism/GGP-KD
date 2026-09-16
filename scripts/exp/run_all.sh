#!/usr/bin/env bash
# Runs every table of story.md §5, in the order the tables depend on each other.
#
# It deliberately STOPS after Table 5 when GRAPH_K is unset. Table 5 chooses the
# operating point and the rule for choosing it is a judgement call -- the
# smallest k at which Avg saturates and the reciprocal graph is about connected --
# so a script must not guess it. Read runs/table5_graph_k/results.csv, then
# re-invoke with GRAPH_K=<k>.
#
#   GPUS=0,1,2,3             bash scripts/exp/run_all.sh   # Table 5, then stop
#   GRAPH_K=100 GPUS=0,1,2,3 bash scripts/exp/run_all.sh   # Tables 3, 7, 4, 6, 1, 2, 8
#   DRY_RUN=1 GRAPH_K=100    bash scripts/exp/run_all.sh   # print the plan only
#
# Env: FROM / TO restrict the range (5, 3, 7, 4, 4h, 6, 1, 2, 8). Everything else
# is forwarded to the table scripts.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# 7 follows 3: it reads Table 3's `ours` as its reference row and the same graph
# artifact. 8 is last -- it is the only table whose runs cost several times an
# ordinary arm, and nothing reads it.
TABLES=(5 3 7 4 4h 6 1 2 8)
FROM="${FROM:-}"
TO="${TO:-}"

in_range() {
    local table="$1" seen_from=0 i
    [[ -z "$FROM" && -z "$TO" ]] && return 0
    for i in "${TABLES[@]}"; do
        [[ -n "$FROM" && "$i" == "$FROM" ]] && seen_from=1
        [[ -z "$FROM" ]] && seen_from=1
        if [[ "$i" == "$table" ]]; then
            (( seen_from )) || return 1
            return 0
        fi
        [[ -n "$TO" && "$i" == "$TO" ]] && return 1
    done
    return 1
}

script_for() {
    case "$1" in
        5)  echo "table5_graph_k.sh" ;;
        3)  echo "table3_exposure.sh" ;;
        4)  echo "table4_loss_terms.sh" ;;
        4h) echo "table4_heldout.sh" ;;
        6)  echo "table6_robustness.sh" ;;
        7)  echo "table7_controls.sh" ;;
        8)  echo "table8_budget.sh" ;;
        1)  echo "table1_main.sh" ;;
        2)  echo "table2_cost.sh" ;;
    esac
}

for value in "$FROM" "$TO"; do
    [[ -z "$value" ]] && continue
    if [[ -z "$(script_for "$value")" ]]; then
        echo "FROM/TO must be one of: ${TABLES[*]} (got: $value)" >&2
        exit 2
    fi
done

rule() { printf '%s\n' "--------------------------------------------------------------"; }

failed=()
for table in "${TABLES[@]}"; do
    in_range "$table" || continue
    if [[ "$table" != "5" && -z "${GRAPH_K:-}" ]]; then
        echo
        rule
        cat <<'EOF'
Table 5 is done. Stopping here on purpose.

Pick the operating point before anything else runs:

    column -s, -t runs/table5_graph_k/results.csv | less -S

Rule: the smallest k at which Avg saturates (within ~0.2 of the best) and the
reciprocal graph is about connected. Small beats tied-and-large -- it is the
whole cost argument. Then:

    GRAPH_K=<k> bash scripts/exp/run_all.sh
EOF
        rule
        exit 0
    fi
    script="$(script_for "$table")"
    echo
    rule
    echo "Table $table -- $script"
    rule
    # Second pass: the k sweep already ran. Table 5 still has to hold the run at
    # GRAPH_K, because it is Table 6's default row and Table 1's pair (c); run just
    # that one setting if it is missing, and skip the table otherwise.
    if [[ "$table" == "5" && -n "${GRAPH_K:-}" ]]; then
        csv="$REPO_ROOT/runs/table5_graph_k/results.csv"
        if [[ -f "$csv" ]] && grep -q ",graph_k_$GRAPH_K," "$csv"; then
            echo "GRAPH_K=$GRAPH_K is already chosen and Table 5 holds its run: skipping"
            continue
        fi
        echo "GRAPH_K=$GRAPH_K is already chosen: running Table 5 at that k only,"
        echo "to materialise the reference Tables 1 and 6 read."
        code=0
        GRAPH_KS="$GRAPH_K" bash "$SCRIPT_DIR/$script" || code=$?
        if (( code == 0 )); then
            continue
        fi
        failed+=("$table")
        echo "Table $table failed with exit $code" >&2
        break
    fi
    code=0
    bash "$SCRIPT_DIR/$script" || code=$?
    if (( code != 0 )); then
        failed+=("$table")
        echo "Table $table failed with exit $code" >&2
        # Later tables read earlier ones (4h scores 4's checkpoints, 6 and 1 read
        # 5's reference), so a failure is not something to run past.
        break
    fi
done

if (( ${#failed[@]} > 0 )); then
    echo
    echo "Failed tables: ${failed[*]}" >&2
    exit 1
fi
