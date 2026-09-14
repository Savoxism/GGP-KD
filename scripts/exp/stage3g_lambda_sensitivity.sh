#!/usr/bin/env bash
# Stage 3G of experiments.md -- independent sensitivity for all three loss
# coefficients. The Stage 0 winner is the default reference
# (lambda_0, lambda_1, lambda_row) = (0.5, 0.5, 1.0), so it is not repeated.
#
# Each arm changes exactly one coefficient and explicitly pins the other two.
# Every coefficient must remain strictly positive: zero-weight arms belong only
# to the loss-decomposition study in Stages 1/2D.
#
# Default matrix (6 arms, 18 runs at three seeds):
#   lambda_0:   0.25, 1.0
#   lambda_1:   0.25, 1.0
#   lambda_row: 0.5,  2.0
#
#   GRAPH_K=50 GPUS=0,1,2,3 SEEDS=42,43,44 \
#       bash scripts/exp/stage3g_lambda_sensitivity.sh
#   GRAPH_K=50 GPUS=0 DRY_RUN=1 \
#       bash scripts/exp/stage3g_lambda_sensitivity.sh
#
# Env: GRAPH_K (required), LAMBDA0_VALUES, LAMBDA1_VALUES,
#      LAMBDA_ROW_VALUES, plus the common runner variables GPUS, SEEDS, PAIR,
#      CORPUS, CACHE_ROOT, RESULT_BASE, CSV_OUT, ARMS and DRY_RUN.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/stage_common.sh
source "$SCRIPT_DIR/lib/stage_common.sh"
stage_common_init
require_graph_k
# shellcheck source=lib/run_arms.sh
source "$SCRIPT_DIR/lib/run_arms.sh"

cd "$STAGE_REPO_ROOT"

EXPERIMENT="stage3g_lambda_sensitivity"
LAMBDA0_VALUES="${LAMBDA0_VALUES:-0.25,0.5,0.75,1.0}"
LAMBDA1_VALUES="${LAMBDA1_VALUES:-0.25,0.5,0.75,1.0}"
LAMBDA_ROW_VALUES="${LAMBDA_ROW_VALUES:-0.1,0.25,0.5,1.0}"

BASE_LAMBDA0="0.5"
BASE_LAMBDA1="0.5"
BASE_LAMBDA_ROW="1.0"

GRAPH_SPEC="main|$CORPUS|$(method_graph_flags)"
ARMS_SPEC=""

validate_positive_weight() {
    local axis="$1" value="$2"
    if [[ ! "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
        echo "$axis values must be positive finite numbers, got: $value" >&2
        exit 2
    fi
    if ! awk -v value="$value" 'BEGIN { exit !(value > 0) }'; then
        echo "$axis values must stay > 0; zero belongs to loss decomposition" >&2
        exit 2
    fi
}

weight_label() {
    local value="$1"
    value="${value//./p}"
    value="${value//+/plus}"
    value="${value//-/minus}"
    printf '%s' "$value"
}

append_axis() {
    local axis="$1" values="$2" value label
    local -a parsed=()
    IFS=',' read -r -a parsed <<< "$values"
    if (( ${#parsed[@]} == 0 )); then
        echo "$axis needs at least one sensitivity value" >&2
        exit 2
    fi
    for value in "${parsed[@]}"; do
        validate_positive_weight "$axis" "$value"
        label="$(weight_label "$value")"
        case "$axis" in
            lambda0)
                ARMS_SPEC+="lambda0_$label|main|ggpkd|--r0_weight $value --r1_weight $BASE_LAMBDA1 --row_weight $BASE_LAMBDA_ROW
"
                ;;
            lambda1)
                ARMS_SPEC+="lambda1_$label|main|ggpkd|--r0_weight $BASE_LAMBDA0 --r1_weight $value --row_weight $BASE_LAMBDA_ROW
"
                ;;
            lambda_row)
                ARMS_SPEC+="lambda_row_$label|main|ggpkd|--r0_weight $BASE_LAMBDA0 --r1_weight $BASE_LAMBDA1 --row_weight $value
"
                ;;
            *)
                echo "Unknown lambda axis: $axis" >&2
                exit 2
                ;;
        esac
    done
}

append_axis "lambda0" "$LAMBDA0_VALUES"
append_axis "lambda1" "$LAMBDA1_VALUES"
append_axis "lambda_row" "$LAMBDA_ROW_VALUES"

CSV_OUT="${CSV_OUT:-$STAGE_REPO_ROOT/runs/$EXPERIMENT/results.csv}"
mkdir -p "$(dirname "$CSV_OUT")"
export GRAPH_SPEC ARMS_SPEC CSV_OUT

echo "Stage 3G: independent lambda sensitivity at graph_k=$GRAPH_K"
note "reference: Stage 0 winner at (lambda_0, lambda_1, lambda_row) = (0.5, 0.5, 1.0)"
note "one coefficient changes per arm; all three loss groups remain active"
echo

status=0
run_arms "$STAGE_REPO_ROOT" "$EXPERIMENT" || status=$?

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    cat <<EOF

Read every arm against Stage 0's graph_k=$GRAPH_K reference. This is a local
sensitivity diagnostic, not a hyperparameter search; do not select the best arm.
Results: $CSV_OUT
EOF
fi
exit $status
