#!/usr/bin/env bash
# Table 7 of story.md §6.2 -- the three controls that decide what the gain is
# attributable to. 4 runs per seed.
#
#   shuffled_target  --row_target shuffled   the teacher's own values, permuted
#                                            among the same columns: support,
#                                            entropy and the histogram survive,
#                                            only "which neighbour is worth what"
#                                            is destroyed. Uniform (Table 3 row 7)
#                                            is the loose bracket, this is the
#                                            tight one.
#   random_pool      --pool_source random    the same number of texts encoded per
#                                            step, drawn uniformly instead of from
#                                            the anchors' neighbourhoods. It holds
#                                            the compute of a 4.7k pool fixed and
#                                            removes only its structure, which is
#                                            the confound in comparing B=64 with a
#                                            subgraph pool.
#   dense_rows       --row_columns pool      dense relational KD on the same pool:
#                                            every encoded text is a row centre
#                                            against every other. The strong
#                                            baseline for sparse local rows -- read
#                                            train_row_eff_denom beside Avg, since
#                                            the claim is equal quality at fewer
#                                            pair evaluations.
#   dense_rows_only  + --cal_weight 0        the same, without L_cal. With L_cal on,
#                                            each anchor's pool row is supervised
#                                            twice at two temperatures; this arm is
#                                            the clean one-term dense baseline.
#
# The reference row is Table 3's `ours`: same graph key, same k, no holdout, so it
# is not repeated here.
#
#   GRAPH_K=100 GPUS=0,1,2,3 bash scripts/exp/table7_controls.sh
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

EXPERIMENT="table7_controls"

# One graph, the method's own: every arm here changes the objective or the pool,
# never the graph, so all four read the artifact Table 3 built.
GRAPH_SPEC="main|$CORPUS|$(method_graph_flags)"

ARMS_SPEC="shuffled_target|main|ggpkd|--row_target shuffled
random_pool|main|ggpkd|--pool_source random
dense_rows|main|ggpkd|--row_columns pool
dense_rows_only|main|ggpkd|--row_columns pool --cal_weight 0
"

CSV_OUT="${CSV_OUT:-$STAGE_REPO_ROOT/runs/$EXPERIMENT/results.csv}"
mkdir -p "$(dirname "$CSV_OUT")"
export GRAPH_SPEC ARMS_SPEC CSV_OUT

echo "Table 7: attribution controls at graph_k=$GRAPH_K"
note "reference row: Table 3's ours (same graph, same k, no holdout) -- not repeated"
note "read Avg beside train_row_count, train_row_eff_denom and train_row_exposed_mass"
note "random_pool: a row needs 2 graph columns inside the pool, and a random pool"
note "rarely supplies them -- if train_row_count collapses, the arm is L_cal on a"
note "random pool, which is the finding, not a failure. Report row_count with Avg."
TABLE3_CSV="$STAGE_REPO_ROOT/runs/table3_exposure/results.csv"
if [[ ! -f "$TABLE3_CSV" ]] || ! grep -q ",ours," "$TABLE3_CSV"; then
    warn "no ours row in $TABLE3_CSV; the reference row is missing."
    warn "  GRAPH_K=$GRAPH_K ARMS=ours,uniform_target bash scripts/exp/table3_exposure.sh"
fi
echo

status=0
run_arms "$STAGE_REPO_ROOT" "$EXPERIMENT" || status=$?

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    cat <<REPORT

How to read it (story.md §10):
  graded beats uniform but ties shuffled  -> the ordering of the values is not
                                             yet evidenced; entropy is the
                                             competing explanation
  random_pool ties ours                   -> do not put the neighbourhood sampler
                                             at the centre of a causal claim
  dense_rows ties ours                    -> position sparse rows on measured cost
                                             (pair evaluations), not on quality
Results: $CSV_OUT
REPORT
fi
exit $status
