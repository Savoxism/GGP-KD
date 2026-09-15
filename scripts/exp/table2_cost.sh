#!/usr/bin/env bash
# Table 2 of story.md -- cost per optimizer step. 3 runs per seed.
#
#   pointwise  --method pointwise   64 texts, no rows
#   in_batch   --batch_local        64 texts, 64 rows, L_cal only
#   ours       the method           the subgraph pool, every encoded row
#
# Step time and peak memory are only comparable when no two runs share a GPU, so
# JOBS_PER_GPU is forced to 1 whatever the environment says. Runs on different
# GPUs are still concurrent: use GPUs of one model, or GPUS=<one id>.
#
#   GRAPH_K=100 GPUS=0 bash scripts/exp/table2_cost.sh
#
# Env: GRAPH_K (required), plus GPUS, SEEDS (42), PAIR, CORPUS, CACHE_ROOT,
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

EXPERIMENT="table2_cost"
if [[ -n "${JOBS_PER_GPU:-}" && "$JOBS_PER_GPU" != "1" ]]; then
    warn "JOBS_PER_GPU=$JOBS_PER_GPU ignored: Table 2 measures unshared GPUs"
fi
JOBS_PER_GPU=1

GRAPH_SPEC="main|$CORPUS|$(method_graph_flags)"
ARMS_SPEC="pointwise|main|pointwise|
in_batch|main|ggpkd|--batch_local
ours|main|ggpkd|
"

CSV_OUT="${CSV_OUT:-$STAGE_REPO_ROOT/runs/$EXPERIMENT/results.csv}"
mkdir -p "$(dirname "$CSV_OUT")"
export GRAPH_SPEC ARMS_SPEC CSV_OUT JOBS_PER_GPU

echo "Table 2: cost per step at graph_k=$GRAPH_K (one job per GPU)"
note "read train_encoded_texts_cum, train_pool_size, train_row_count,"
note "train_mean_step_seconds, train_peak_memory_mb and cost_peak_gpu_mib"
echo

status=0
run_arms "$STAGE_REPO_ROOT" "$EXPERIMENT" || status=$?

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    cat <<EOF

Graph build time is in the graph.main line of the run tree's stats.tsv; it is
the offline cost and is reported separately from the per-step columns.
Results: $CSV_OUT
EOF
fi
exit $status
