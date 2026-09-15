#!/usr/bin/env bash
# Shared preamble for the per-table scripts in scripts/exp (story.md §5).
#
# Sourced, never executed. It adds two things on top of `run_arms.sh`:
#
#   require_graph_k   the operating point is read off Table 5 by a human and every
#                     other table inherits it, so no table may quietly default it.
#   note / warn       one prefix for everything a table script says about itself.

stage_common_init() {
    STAGE_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[1]}")/../.." && pwd)"
    PYTHON_BIN="${PYTHON_BIN:-$STAGE_REPO_ROOT/.venv/bin/python}"
    PAIR="${PAIR:-qwen3_0_6b_to_minilmv2_h384}"
    CORPUS="${CORPUS:-data/train_set/merged_3_data_5k_each.csv}"
    CORPUS_KEY="$(basename "${CORPUS%.*}")"
    # One seed while the tables are still being shaped; SEEDS=42,43,44 for the
    # numbers that go into the paper.
    SEEDS="${SEEDS:-42}"
    CACHE_ROOT="${CACHE_ROOT:-$STAGE_REPO_ROOT/cache/exp}"
    export PAIR SEEDS CACHE_ROOT PYTHON_BIN
}

note() { printf '  %s\n' "$*"; }
warn() { printf '  WARNING: %s\n' "$*" >&2; }

# Table 5 picks the operating point; nothing after it may guess one.
require_graph_k() {
    if [[ -z "${GRAPH_K:-}" ]]; then
        cat >&2 <<'EOF'
GRAPH_K is not set.

Every table except Table 5 runs at the operating point Table 5 selects, and
picking it is a judgement call (the smallest k at which Avg saturates and the
reciprocal graph is about connected), so it is not defaulted here. Run
scripts/exp/table5_graph_k.sh, read runs/table5_graph_k/results.csv, then:

    GRAPH_K=<k> bash scripts/exp/<this script>
EOF
        exit 2
    fi
    if [[ ! "$GRAPH_K" =~ ^[0-9]+$ ]] || (( GRAPH_K < 2 )); then
        echo "GRAPH_K must be an integer >= 2, got: $GRAPH_K" >&2
        exit 2
    fi
}

# The method at the operating point. Graph-defining flags only; the objective is
# whatever config/ggpkd_config.py ships.
method_graph_flags() {
    printf -- '--graph_k %s' "$GRAPH_K"
}

# Where run_arms puts a graph artifact, so post-hoc tools read the same file the
# runs trained against.
graph_artifact_path() {
    local key="$1" pair="${2:-$PAIR}"
    printf '%s' "$CACHE_ROOT/$pair/$CORPUS_KEY/graph_$key.pt"
}
