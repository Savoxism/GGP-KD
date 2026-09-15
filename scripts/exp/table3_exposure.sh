#!/usr/bin/env bash
# Table 3 of story.md -- exposure, not mechanism. 8 runs per seed.
#
# Encoder, optimizer, steps and seed are fixed; only which texts are co-encoded
# and which rows are matched on them vary. Read in pairs:
#
#   pointwise_random   -> pointwise_neighbor   neighbour batching alone
#   in_batch_random    -> in_batch_neighbor    batch-local KD, with and without
#                                              neighbours in the batch
#   anchor_rows        -> ours                 supervising every encoded row
#   ours               -> uniform_target       graded values on the same columns
#   ours               -> random_columns       graph columns vs random pool texts
#
# Pointwise neighbour batching reads the graph from --ggpkd_cache_path, which the
# runner builds before any arm starts, so every neighbour arm is grouped by the
# same neighbourhoods the subgraph arms distil from.
#
#   GRAPH_K=100 GPUS=0,1,2,3 bash scripts/exp/table3_exposure.sh
#
# Env: GRAPH_K (required), BATCH_SIZE (64, for the exposure count only), EPOCHS
#      (5, likewise), plus GPUS, SEEDS, PAIR, CORPUS, CACHE_ROOT, RESULT_BASE,
#      CSV_OUT, ARMS, DRY_RUN.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/stage_common.sh
source "$SCRIPT_DIR/lib/stage_common.sh"
stage_common_init
require_graph_k
# shellcheck source=lib/run_arms.sh
source "$SCRIPT_DIR/lib/run_arms.sh"

cd "$STAGE_REPO_ROOT"

EXPERIMENT="table3_exposure"
GRAPH_SPEC="main|$CORPUS|$(method_graph_flags)"

ARMS_SPEC="pointwise_random|main|pointwise|
pointwise_neighbor|main|pointwise|--batch_sampler neighbor
in_batch_random|main|ggpkd|--batch_local
in_batch_neighbor|main|ggpkd|--batch_local --batch_sampler neighbor
anchor_rows|main|ggpkd|--row_set anchors
ours|main|ggpkd|
uniform_target|main|ggpkd|--row_target uniform
random_columns|main|ggpkd|--row_columns random
"

CSV_OUT="${CSV_OUT:-$STAGE_REPO_ROOT/runs/$EXPERIMENT/results.csv}"
mkdir -p "$(dirname "$CSV_OUT")"
export GRAPH_SPEC ARMS_SPEC CSV_OUT

echo "Table 3: exposure arms at graph_k=$GRAPH_K"
note "ours is the method with no flags; every other arm changes one thing"
echo

status=0
run_arms "$STAGE_REPO_ROOT" "$EXPERIMENT" || status=$?

if [[ "${DRY_RUN:-0}" != "1" && "$status" == "0" ]]; then
    # The nbrs / row column is measured, not intended: the greedy partition runs
    # out of unused neighbours and tops batches up at random.
    artifact="$(graph_artifact_path main)"
    coverage_csv="$STAGE_REPO_ROOT/runs/$EXPERIMENT/coverage_$PAIR.csv"
    first_seed="${SEEDS%%,*}"
    echo
    echo "Measured in-batch neighbours per row (Table 3, nbrs / row):"
    "$PYTHON_BIN" "$SCRIPT_DIR/coverage.py" \
        --artifact "$artifact" \
        --pair "$PAIR" \
        --batch-size "${BATCH_SIZE:-64}" \
        --epochs "${EPOCHS:-5}" \
        --seed "$first_seed" \
        --out "$coverage_csv" || status=$?
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    cat <<EOF

The subgraph rows' nbrs / row is train_row_exposed_mass (share of each row's
teacher mass inside the pool), not the in-batch count above.
Results: $CSV_OUT
EOF
fi
exit $status
