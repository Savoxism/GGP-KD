#!/usr/bin/env bash
# Stage 2B of experiments.md -- the comparison ladder. 12 runs per pair.
#
# Every arm runs the complete objective. The anchor-row quota is matched, while
# the shared-pool calibration domain and the usable auxiliary rows are allowed to
# follow from each support, exactly as they do in the shipped system. Loss
# decomposition is isolated to Stages 1 and 2D.
#
#   in_batch        the rest of the batch                      (the baseline)
#   corpus_uniform  the same number, drawn from the corpus      kills "just leave the batch"
#   student_knn     the frozen base student's own top-k         method
#   teacher_knn     the teacher's own top-k                     source ablation
#
# `rewired` is deliberately absent: with a directed graph and no truncation every
# anchor has degree exactly graph_k, so it draws the same number from the same
# distribution as corpus_uniform. Verified identical for 400/400 anchors.
#
#   GRAPH_K=200 GPUS=0,1,2,3 bash scripts/exp/stage2b_ladder.sh
#   GRAPH_K=200 ARMS=student_knn bash scripts/exp/stage2b_ladder.sh   # re-run one arm
#
# Env: PAIRS (qwen3_0_6b_to_minilmv2_h384 only; the second pair is out of scope),
#      SUPPORT_SIZE (batch_size - 1), ARMS, plus the usual GPUS/SEEDS/DRY_RUN.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/stage_common.sh
source "$SCRIPT_DIR/lib/stage_common.sh"
stage_common_init
require_graph_k
# shellcheck source=lib/run_arms.sh
source "$SCRIPT_DIR/lib/run_arms.sh"

cd "$STAGE_REPO_ROOT"

EXPERIMENT="stage2b_ladder"
BATCH_SIZE="${BATCH_SIZE:-64}"
# The like-for-like count: every arm scores an anchor against this many texts,
# and the baseline has exactly batch_size - 1 available.
SUPPORT_SIZE="${SUPPORT_SIZE:-$((BATCH_SIZE - 1))}"
PAIRS="${PAIRS:-qwen3_0_6b_to_minilmv2_h384}"

FULL="--batch_size $BATCH_SIZE"
BUDGET="--diffusion_quota $SUPPORT_SIZE"

echo "Stage 2B: comparison ladder at graph_k=$GRAPH_K, $SUPPORT_SIZE comparisons per anchor"
note "objective: full (r=0 + r=1 + row); only loss-decomposition stages remove terms"
note "pairs: ${PAIRS//,/, }"
echo

overall=0
IFS=',' read -r -a pair_list <<< "$PAIRS"
for pair in "${pair_list[@]}"; do
    echo "##############################################"
    echo "# ladder -- pair $pair"
    echo "##############################################"

    # The method graph is student-selected with teacher-valued rows. The first
    # two controls reuse that graph for L_row; their anchor supports are batch or
    # corpus-uniform. The teacher arm changes the graph's column source explicitly.
    graph_spec="student_knn|$CORPUS|$(method_graph_flags)
teacher_knn|$CORPUS|$(method_graph_flags) --neighbor_source teacher"
    arms="in_batch|student_knn|ggpkd|--batch_local --relation_target direct $FULL
corpus_uniform|student_knn|ggpkd|--support_policy corpus_uniform --relation_target direct $FULL $BUDGET
student_knn|student_knn|ggpkd|$FULL $BUDGET
teacher_knn|teacher_knn|ggpkd|$FULL $BUDGET
"

    csv_out="${CSV_OUT:-$STAGE_REPO_ROOT/runs/$EXPERIMENT/results.csv}"
    mkdir -p "$(dirname "$csv_out")"
    PAIR="$pair" GRAPH_SPEC="$graph_spec" ARMS_SPEC="$arms" CSV_OUT="$csv_out" \
        RESULT_BASE="${RESULT_BASE:-$STAGE_REPO_ROOT/results/$EXPERIMENT/$pair}" \
        run_arms "$STAGE_REPO_ROOT" "$EXPERIMENT" || overall=$?
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    cat <<'EOF'

Read it as a complete-system support comparison. The r=1 anchor-row quota is
matched, but r=0 pool width and L_row exposure are outcomes and must be reported
beside Avg. Report train_encoded_texts_cum, pool columns and row_count so a gain
is not misattributed to an unreported amount of auxiliary supervision.
EOF
fi
exit $overall
