#!/usr/bin/env bash
# Table 5 of story.md -- k and reciprocal connectivity. 4 runs per seed.
#
# graph_k is one of the method's two knobs and sets three things at once: the
# row width, the per-row temperature tau_j = (s(1) - s(k)) / log k, and through
# both the pool size a step encodes. §3 predicts where Avg stops improving: once
# the reciprocal kNN graph is about connected, L_row pins the teacher geometry up
# to a handful of offsets and a wider row buys encode cost, not signal.
#
# It runs first because it chooses the operating point every other table uses,
# and its run at that k is the default row of Table 6 (not repeated there).
#
#   GPUS=0,1,2,3 bash scripts/exp/table5_graph_k.sh
#   DRY_RUN=1    bash scripts/exp/table5_graph_k.sh
#
# Env: GPUS, SEEDS (42), PAIR, CORPUS, GRAPH_KS (25,50,100,200),
#      CACHE_ROOT, RESULT_BASE, CSV_OUT, ARMS, DRY_RUN.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/stage_common.sh
source "$SCRIPT_DIR/lib/stage_common.sh"
stage_common_init
# shellcheck source=lib/run_arms.sh
source "$SCRIPT_DIR/lib/run_arms.sh"

cd "$STAGE_REPO_ROOT"

EXPERIMENT="table5_graph_k"
GRAPH_KS="${GRAPH_KS:-25,50,100,200}"

GRAPH_SPEC=""
ARMS_SPEC=""
IFS=',' read -r -a ks <<< "$GRAPH_KS"
for k in "${ks[@]}"; do
    [[ "$k" =~ ^[0-9]+$ ]] || { echo "graph_k must be an integer, got: $k" >&2; exit 2; }
    # One graph per k: graph_k changes the artifact, and the runner forwards a
    # graph's flags to every training run that uses it.
    GRAPH_SPEC+="k$k|$CORPUS|--graph_k $k
"
    ARMS_SPEC+="graph_k_$k|k$k|ggpkd|
"
done

CSV_OUT="${CSV_OUT:-$STAGE_REPO_ROOT/runs/$EXPERIMENT/results.csv}"
mkdir -p "$(dirname "$CSV_OUT")"
export GRAPH_SPEC ARMS_SPEC CSV_OUT

echo "Table 5: graph_k in {${GRAPH_KS//,/, }} on the method's defaults"
note "the arms carry no objective flags: this is the method as it ships"
note "read Avg beside graph_reciprocal_components, graph_reciprocal_isolated,"
note "train_row_exposed_mass, train_pool_size and train_peak_memory_mb"
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
    ("components", "graph_reciprocal_components"),
    ("isolated", "graph_reciprocal_isolated"),
    ("exposed", "train_row_exposed_mass"),
    ("pool", "train_pool_size"),
    ("peak MiB", "train_peak_memory_mb"),
)
groups = defaultdict(list)
with open(path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if row.get("pair") == pair and row.get("status") == "ok":
            groups[row["arm"]].append(row)

def mean(rows, key):
    values = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return f"{sum(values) / len(values):.2f}" if values else "-"

def k_of(arm):
    return int(arm.rsplit("_", 1)[-1]) if arm.rsplit("_", 1)[-1].isdigit() else 0

print(f"{'k':>5}  " + "  ".join(f"{name:>10}" for name, _ in columns) + "  seeds")
for arm in sorted(groups, key=k_of):
    rows = groups[arm]
    print(f"{k_of(arm):>5}  " + "  ".join(f"{mean(rows, key):>10}" for _, key in columns)
          + f"  {len(rows)}")
PY
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    cat <<EOF

Rule: take the smallest k at which Avg saturates -- within noise (~0.2, the sd
of a one-seed difference) of the best k -- and at which the reciprocal graph is
about connected (a handful of components, the isolated count near zero). Both
should agree; if they do not, report both and take the larger k.
Small beats tied-and-large: it is the whole cost argument.
Then run every other table with GRAPH_K=<k>.
Results: $CSV_OUT
EOF
fi
exit $status
