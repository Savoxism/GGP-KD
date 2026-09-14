#!/usr/bin/env bash
# Stage 3F of experiments.md -- the main-table deliverable. 6 runs.
#
# F: the main table, three teacher->student pairs at the Stage 0 operating point
#    with whatever Stage 1 left standing. Pair 1 is Stage 0's winner, so only the
#    other two are run here.
#
#   GRAPH_K=50 GPUS=0,1,2,3 bash scripts/exp/stage3_main_table.sh
#
# Run the three-lambda sensitivity separately with
# scripts/exp/stage3g_lambda_sensitivity.sh.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/stage_common.sh
source "$SCRIPT_DIR/lib/stage_common.sh"
stage_common_init
require_graph_k
# shellcheck source=lib/run_arms.sh
source "$SCRIPT_DIR/lib/run_arms.sh"

cd "$STAGE_REPO_ROOT"

# Pair 1 is Stage 0's winner; naming it here only to be explicit about the reuse.
STAGE0_PAIR="${STAGE0_PAIR:-qwen3_0_6b_to_minilmv2_h384}"
# Empty by default: the study is run on the Stage 0 pair only, whose main-table
# row is Stage 0's winner. Set PAIRS to add the other two back.
PAIRS="${PAIRS:-}"

echo "Stage 3F: main table at graph_k=$GRAPH_K"
note "pair $STAGE0_PAIR comes from Stage 0 and is not repeated"
note "pairs run here: ${PAIRS:-none}"
echo

overall=0
IFS=',' read -r -a pair_list <<< "$PAIRS"
for pair in "${pair_list[@]}"; do
    if [[ "$pair" == "$STAGE0_PAIR" ]]; then
        warn "skipping $pair: it is Stage 0's winner, already measured"
        continue
    fi
    echo "##############################################"
    echo "# main table -- pair $pair"
    echo "##############################################"
    csv="$STAGE_REPO_ROOT/runs/stage3_main_table/results.csv"
    mkdir -p "$(dirname "$csv")"
    PAIR="$pair" GRAPH_SPEC="main|$CORPUS|$(method_graph_flags)" \
        ARMS_SPEC="full|main|ggpkd|
" CSV_OUT="$csv" \
        RESULT_BASE="${RESULT_BASE:-$STAGE_REPO_ROOT/results/stage3_main_table/$pair}" \
        run_arms "$STAGE_REPO_ROOT" "stage3_main_table" || overall=$?
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo
    echo "Defaults are reported as defaults, not as a selected best: the point of"
    echo "the sensitivity study is that the surface is flat, so no line may read as a search."
fi
exit $overall
