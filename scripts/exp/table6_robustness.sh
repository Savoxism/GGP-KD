#!/usr/bin/env bash
# Table 6 of story.md -- robustness, one change at a time. 6 runs per seed.
#
# The default row is Table 5's run at the same k (graph_k_<GRAPH_K>): same graph
# flags, same objective, so it is not repeated here.
#
#   cal_0p25     --cal_weight 0.25        lambda_cal : lambda_row = 0.25 : 1
#   cal_1p0      --cal_weight 1.0         1 : 1
#   row_reweight --row_reweight           1/p_j row weights (GraphSAINT)
#   student_knn  --neighbor_source student  columns from the student's kNN
#   mutual_knn   --knn_mode mutual        mutual kNN filter
#   global_tau   --fixed_bandwidth        one median tau for every row
#
# The last three change the artifact, so each has its own graph key. These rows
# show the default is not a narrow peak; they are not a search, and no arm is
# selected from them -- except row_reweight, which becomes the default if it
# wins, because it adds no knob.
#
#   GRAPH_K=100 GPUS=0,1,2,3 bash scripts/exp/table6_robustness.sh
#
# Env: GRAPH_K (required), plus GPUS, SEEDS, PAIR, CORPUS, CACHE_ROOT,
#      RESULT_BASE, CSV_OUT, ARMS, DRY_RUN.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/stage_common.sh
source "$SCRIPT_DIR/lib/stage_common.sh"
stage_common_init
require_graph_k
# shellcheck source=lib/run_arms.sh
source "$SCRIPT_DIR/lib/run_arms.sh"

cd "$STAGE_REPO_ROOT"

EXPERIMENT="table6_robustness"

GRAPH_SPEC="main|$CORPUS|$(method_graph_flags)
student|$CORPUS|$(method_graph_flags) --neighbor_source student
mutual|$CORPUS|$(method_graph_flags) --knn_mode mutual
median_tau|$CORPUS|$(method_graph_flags) --fixed_bandwidth"

ARMS_SPEC="cal_0p25|main|ggpkd|--cal_weight 0.25
cal_1p0|main|ggpkd|--cal_weight 1.0
row_reweight|main|ggpkd|--row_reweight
student_knn|student|ggpkd|
mutual_knn|mutual|ggpkd|
global_tau|median_tau|ggpkd|
"

CSV_OUT="${CSV_OUT:-$STAGE_REPO_ROOT/runs/$EXPERIMENT/results.csv}"
mkdir -p "$(dirname "$CSV_OUT")"
export GRAPH_SPEC ARMS_SPEC CSV_OUT

echo "Table 6: robustness at graph_k=$GRAPH_K"
note "default row: Table 5's graph_k_$GRAPH_K run (lambda_cal:lambda_row = 0.5:1,"
note "teacher kNN, directed, per-row tau, no reweighting) -- not repeated here"
TABLE5_CSV="$STAGE_REPO_ROOT/runs/table5_graph_k/results.csv"
if [[ ! -f "$TABLE5_CSV" ]] || ! grep -q ",graph_k_$GRAPH_K," "$TABLE5_CSV"; then
    warn "no graph_k_$GRAPH_K row in $TABLE5_CSV; the default row is missing."
    warn "  GRAPH_KS=$GRAPH_K bash scripts/exp/table5_graph_k.sh"
fi
echo

status=0
run_arms "$STAGE_REPO_ROOT" "$EXPERIMENT" || status=$?

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    cat <<EOF

Read every arm as a delta against Table 5's graph_k_$GRAPH_K row, at the same
seeds. Gaps below ~0.2 are noise at one seed.
Results: $CSV_OUT
EOF
fi
exit $status
