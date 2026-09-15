#!/usr/bin/env bash
# Table 1 of story.md -- main results. 1 run per pair per seed.
#
# The method at the Table 5 operating point on the other two teacher->student
# pairs. Pair (c), qwen3_0_6b_to_minilmv2_h384, is Table 5's run at that k and is
# not repeated.
#
# Every pair exports into one runs/table1_main/results.csv. The export merges by
# (experiment, pair, arm, seed), so a second pair adds rows rather than replacing
# the first pair's -- the bug that left the old main table with one pair in it.
#
#   GRAPH_K=100 GPUS=0,1,2,3 SEEDS=42,43,44 bash scripts/exp/table1_main.sh
#
# Env: GRAPH_K (required), PAIRS (bge_m3_to_minilmv2_h768,qwen3_4b_to_bert_base),
#      plus GPUS, SEEDS, CORPUS, CACHE_ROOT, RESULT_BASE, CSV_OUT, DRY_RUN.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/stage_common.sh
source "$SCRIPT_DIR/lib/stage_common.sh"
stage_common_init
require_graph_k
# shellcheck source=lib/run_arms.sh
source "$SCRIPT_DIR/lib/run_arms.sh"

cd "$STAGE_REPO_ROOT"

EXPERIMENT="table1_main"
TABLE5_PAIR="qwen3_0_6b_to_minilmv2_h384"
PAIRS="${PAIRS:-bge_m3_to_minilmv2_h768,qwen3_4b_to_bert_base}"
CSV_OUT="${CSV_OUT:-$STAGE_REPO_ROOT/runs/$EXPERIMENT/results.csv}"
mkdir -p "$(dirname "$CSV_OUT")"

echo "Table 1: main results at graph_k=$GRAPH_K"
note "pair $TABLE5_PAIR comes from Table 5 (graph_k_$GRAPH_K) and is not repeated"
note "pairs run here: ${PAIRS:-none}"
echo

overall=0
IFS=',' read -r -a pair_list <<< "$PAIRS"
for pair in "${pair_list[@]}"; do
    [[ -z "$pair" ]] && continue
    if [[ "$pair" == "$TABLE5_PAIR" ]]; then
        warn "skipping $pair: Table 5 already measured it at graph_k=$GRAPH_K"
        continue
    fi
    echo "##############################################"
    echo "# Table 1 -- pair $pair"
    echo "##############################################"
    # One result base per pair: both pairs share a RUN_ID, and run_arms refuses
    # to write into an existing run root.
    PAIR="$pair" GRAPH_SPEC="main|$CORPUS|$(method_graph_flags)" \
        ARMS_SPEC="ours|main|ggpkd|
" CSV_OUT="$CSV_OUT" \
        RESULT_BASE="${RESULT_BASE:-$STAGE_REPO_ROOT/results/$EXPERIMENT}/$pair" \
        run_arms "$STAGE_REPO_ROOT" "$EXPERIMENT" || overall=$?
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    cat <<EOF

The method is reported at its defaults, not as a selected best.
Results (all pairs): $CSV_OUT
EOF
fi
exit $overall
