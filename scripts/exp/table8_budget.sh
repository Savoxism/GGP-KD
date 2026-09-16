#!/usr/bin/env bash
# Table 8 of story.md §6.1 (P2) -- quality against training budget, not steps.
#
# Every other table fixes the optimizer steps and lets the pool cost vary. That is
# the comparison story.md §2.6 calls confounded: a step of the method encodes a
# subgraph pool, a step of the baselines encodes 64 texts, so "same steps" is not
# "same budget". This table spends the baselines' saved budget and asks whether
# they catch up.
#
#   ours_e<base>        the method at the default budget, the reference point
#   pointwise_e<E>      pointwise KD at E epochs, for E in BUDGET_EPOCHS
#   in_batch_e<E>       in-batch KD (L_cal on the batch) at E epochs
#
# Read Avg against train_encoded_tokens_cum (the budget axis story.md asks for),
# train_encoded_texts_cum, and wall_seconds from the run tree's stats.tsv. Step
# time and peak memory belong to Table 2, which owns an unshared GPU; this table
# may share GPUs because its axis is tokens, not seconds.
#
# One seed by default: this is a curve, and its runs are up to
# max(BUDGET_EPOCHS)/base times the cost of an ordinary arm. Set BUDGET_SEEDS to
# put error bars on it.
#
#   GRAPH_K=100 GPUS=0,1,2,3 bash scripts/exp/table8_budget.sh
#   BUDGET_EPOCHS=5,10,20,40 GRAPH_K=100 GPUS=0,1 bash scripts/exp/table8_budget.sh
#
# Env: GRAPH_K (required), BUDGET_EPOCHS (5,10,20), BUDGET_SEEDS (the first of
#      SEEDS), EPOCHS (5, the method's budget), plus GPUS, PAIR, CORPUS,
#      CACHE_ROOT, RESULT_BASE, CSV_OUT, ARMS, DRY_RUN.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/stage_common.sh
source "$SCRIPT_DIR/lib/stage_common.sh"
stage_common_init
require_graph_k
# shellcheck source=lib/run_arms.sh
source "$SCRIPT_DIR/lib/run_arms.sh"

cd "$STAGE_REPO_ROOT"

EXPERIMENT="table8_budget"
BUDGET_EPOCHS="${BUDGET_EPOCHS:-5,10,20}"
BASE_EPOCHS="${EPOCHS:-5}"
# A curve wants points, not seeds; SEEDS is taken as the paper's list everywhere
# else, so narrow it here rather than make every caller remember to.
SEEDS="${BUDGET_SEEDS:-${SEEDS%%,*}}"
export SEEDS

if [[ ! "$BASE_EPOCHS" =~ ^[0-9]+$ ]]; then
    echo "EPOCHS must be an integer, got: $BASE_EPOCHS" >&2
    exit 2
fi

GRAPH_SPEC="main|$CORPUS|$(method_graph_flags)"
ARMS_SPEC="ours_e${BASE_EPOCHS}|main|ggpkd|--epochs $BASE_EPOCHS
"
IFS=',' read -r -a budget_list <<< "$BUDGET_EPOCHS"
for epochs in "${budget_list[@]}"; do
    [[ "$epochs" =~ ^[0-9]+$ ]] || { echo "BUDGET_EPOCHS must be integers, got: $epochs" >&2; exit 2; }
    # --epochs is appended after the launcher's own, and the last one parsed wins,
    # so an arm's budget is a plain flag like any other switch.
    ARMS_SPEC+="pointwise_e$epochs|main|pointwise|--epochs $epochs
in_batch_e$epochs|main|ggpkd|--batch_local --epochs $epochs
"
done

CSV_OUT="${CSV_OUT:-$STAGE_REPO_ROOT/runs/$EXPERIMENT/results.csv}"
mkdir -p "$(dirname "$CSV_OUT")"
export GRAPH_SPEC ARMS_SPEC CSV_OUT

echo "Table 8: quality against budget at graph_k=$GRAPH_K"
note "method at $BASE_EPOCHS epochs; baselines at {${BUDGET_EPOCHS//,/, }} epochs"
note "seeds: $SEEDS (set BUDGET_SEEDS for more)"
note "budget axis: train_encoded_tokens_cum, not optimizer steps"
echo

status=0
run_arms "$STAGE_REPO_ROOT" "$EXPERIMENT" || status=$?

if [[ "${DRY_RUN:-0}" != "1" && -f "$CSV_OUT" ]]; then
    echo
    "$PYTHON_BIN" - "$CSV_OUT" "$PAIR" <<'PY' || true
import csv, sys
from collections import defaultdict

path, pair = sys.argv[1], sys.argv[2]
columns = (
    ("Avg", "score_Avg All"),
    ("epochs", "cfg_epochs"),
    ("tokens", "train_encoded_tokens_cum"),
    ("wall_s", "cost_wall_seconds"),
    ("texts", "train_encoded_texts_cum"),
)
groups = defaultdict(list)
with open(path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if row.get("pair") == pair and row.get("status") == "ok":
            groups[row["arm"]].append(row)

def mean(rows, key):
    values = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    if not values:
        return "-"
    value = sum(values) / len(values)
    return f"{value:,.0f}" if value >= 1000 else f"{value:.2f}"

print(f"{'arm':>18}  " + "  ".join(f"{name:>14}" for name, _ in columns) + "  seeds")
for arm in sorted(groups):
    rows = groups[arm]
    print(f"{arm:>18}  " + "  ".join(f"{mean(rows, key):>14}" for _, key in columns)
          + f"  {len(rows)}")
PY
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    cat <<REPORT

Rule (story.md §10): a win on steps that disappears on tokens or GPU-hours is a
trade-off to report, not training efficiency to claim.
Results: $CSV_OUT
REPORT
fi
exit $status
