#!/usr/bin/env bash
# Table 4 of story.md -- loss terms, trained under an edge holdout. 5 runs per seed.
#
#   ours              the method under the same holdout; the reference the other
#                     arms are read against (Table 5's run withheld nothing)
#   cal_off           --cal_weight 0      L_row alone
#   row_off           --row_weight 0      L_cal alone
#   anchors_only      --row_set anchors   the removed L_{r=1} as a row term
#   anchors_excluded  --row_set non_anchors
#   holdout_strict    the method under a holdout whose withheld scores also stay
#                     out of tau_j (--holdout_bandwidth surviving). The default
#                     reads tau_j = (s(1) - s(k)) / log k off the raw top-k, so a
#                     withheld pair's teacher score still shapes its row: that
#                     makes the default a *target-only* holdout (story.md §6.4).
#                     If this arm matches `ours` on the held-out columns, the
#                     probe is not measuring the leak; if it does not, the
#                     held-out numbers have to be reported under this arm.
#
# Every arm withholds the same 20% of graph edges from every training term, so
# scripts/exp/table4_heldout.sh can score the checkpoints on relations none of
# them saw. The prediction: cal_off matches ours on held-out pairs inside one
# reciprocal component and loses on pairs across components.
#
#   GRAPH_K=100 GPUS=0,1,2,3 bash scripts/exp/table4_loss_terms.sh
#   bash scripts/exp/table4_heldout.sh
#
# Env: GRAPH_K (required), HOLDOUT_FRAC (0.2), HOLDOUT_SEED (12345), plus GPUS,
#      SEEDS, PAIR, CORPUS, CACHE_ROOT, RESULT_BASE, CSV_OUT, ARMS, DRY_RUN.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/stage_common.sh
source "$SCRIPT_DIR/lib/stage_common.sh"
stage_common_init
require_graph_k
# shellcheck source=lib/run_arms.sh
source "$SCRIPT_DIR/lib/run_arms.sh"

cd "$STAGE_REPO_ROOT"

EXPERIMENT="table4_loss_terms"
HOLDOUT_FRAC="${HOLDOUT_FRAC:-0.2}"
HOLDOUT_SEED="${HOLDOUT_SEED:-12345}"
if ! awk -v value="$HOLDOUT_FRAC" 'BEGIN { exit !(value > 0 && value < 1) }'; then
    echo "HOLDOUT_FRAC must be in (0, 1): Table 4's held-out columns need a holdout" >&2
    exit 2
fi

# The holdout seed is separate from the training seed on purpose: the split must
# be identical across seeds and arms, or the held-out columns do not score the
# same withheld pairs. The key carries `_holdout` because the cache path is
# graph_<key>.pt, and table4_heldout.sh reads exactly that file.
GRAPH_SPEC="main_holdout|$CORPUS|$(method_graph_flags) --holdout_edge_frac $HOLDOUT_FRAC --holdout_seed $HOLDOUT_SEED
main_holdout_strict|$CORPUS|$(method_graph_flags) --holdout_edge_frac $HOLDOUT_FRAC --holdout_seed $HOLDOUT_SEED --holdout_bandwidth surviving"

ARMS_SPEC="ours|main_holdout|ggpkd|
cal_off|main_holdout|ggpkd|--cal_weight 0
row_off|main_holdout|ggpkd|--row_weight 0
anchors_only|main_holdout|ggpkd|--row_set anchors
anchors_excluded|main_holdout|ggpkd|--row_set non_anchors
holdout_strict|main_holdout_strict|ggpkd|
"

CSV_OUT="${CSV_OUT:-$STAGE_REPO_ROOT/runs/$EXPERIMENT/results.csv}"
mkdir -p "$(dirname "$CSV_OUT")"
export GRAPH_SPEC ARMS_SPEC CSV_OUT

echo "Table 4: loss terms at graph_k=$GRAPH_K, holdout $HOLDOUT_FRAC (seed $HOLDOUT_SEED)"
note "absolute numbers do not line up with Tables 1 and 5, which withhold"
note "nothing; the comparison that matters is between these arms"
note "holdout_strict trains on its own artifact (graph_main_holdout_strict.pt);"
note "it withholds the same pairs, so table4_heldout.sh scores it on the same set"
echo

status=0
run_arms "$STAGE_REPO_ROOT" "$EXPERIMENT" || status=$?

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    cat <<EOF

Compare against this table's own \`ours\` arm, which withheld the same edges.
Then score the held-out pairs:
    bash scripts/exp/table4_heldout.sh
Results: $CSV_OUT
EOF
fi
exit $status
