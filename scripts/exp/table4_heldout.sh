#!/usr/bin/env bash
# Table 4 of story.md -- held-out relation columns. No training.
#
# Scores the checkpoints table4_loss_terms.sh produced on the graph edges every
# arm withheld, split by whether both endpoints lie in the same component of the
# reciprocal graph built on the training edges. scripts/exp/heldout_geometry.py
# builds the sets and documents the metrics; the two Table 4 columns are
# `spearman` on relation_set heldout_same_component / heldout_cross_component.
#
# By default it scores the most recent Table 4 run against graph_main_holdout.pt,
# the graph those runs trained on. Point RUN_ROOT / SOURCE_EXPERIMENT /
# ARTIFACT_KEY elsewhere to score another run tree with the same instrument.
#
#   bash scripts/exp/table4_heldout.sh
#   RUN_ROOT=results/table4_loss_terms/<run_id> bash scripts/exp/table4_heldout.sh
#
# Env: RUN_ROOT, SOURCE_EXPERIMENT (table4_loss_terms), ARTIFACT_KEY
#      (main_holdout), ANCHORS (512), KNN_K (10), OUT_DIR, CACHE_ROOT, CORPUS,
#      DRY_RUN.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/stage_common.sh
source "$SCRIPT_DIR/lib/stage_common.sh"
stage_common_init

cd "$STAGE_REPO_ROOT"

EXPERIMENT="table4_heldout"
SOURCE_EXPERIMENT="${SOURCE_EXPERIMENT:-table4_loss_terms}"
ARTIFACT_KEY="${ARTIFACT_KEY:-main_holdout}"
ANCHORS="${ANCHORS:-512}"
KNN_K="${KNN_K:-10}"
OUT_DIR="${OUT_DIR:-$STAGE_REPO_ROOT/runs/$EXPERIMENT}"

# Which run tree to score. run_arms writes the pointer on every successful
# sweep, so the common case needs no argument.
POINTER="$STAGE_REPO_ROOT/runs/$SOURCE_EXPERIMENT/last_run_root.txt"
if [[ -z "${RUN_ROOT:-}" && -f "$POINTER" ]]; then
    RUN_ROOT="$(cat "$POINTER")"
fi

echo "Table 4 held-out geometry"
note "run tree: ${RUN_ROOT:-<latest $SOURCE_EXPERIMENT run, via $POINTER>}"
note "graph:    graph_$ARTIFACT_KEY.pt"
note "sets:     heldout_edges, heldout_same_component, heldout_cross_component,"
note "          nonlocal, unsupervised"
note "output:   $OUT_DIR/heldout_geometry.csv"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo
    echo "DRY_RUN: nothing scored"
    exit 0
fi

if [[ -z "${RUN_ROOT:-}" ]]; then
    echo "No RUN_ROOT given and no $POINTER; run $SOURCE_EXPERIMENT first" >&2
    exit 2
fi
mkdir -p "$OUT_DIR"

# A tree that nests one level per pair (Table 1's layout) is scored pair by pair.
mapfile -t RUN_DIRS < <(
    if [[ -d "$RUN_ROOT/runs" ]]; then
        printf '%s\n' "$RUN_ROOT"
    else
        find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -exec test -d '{}/runs' ';' -print
    fi
)
if (( ${#RUN_DIRS[@]} == 0 )); then
    echo "No run trees with a runs/ directory under $RUN_ROOT" >&2
    exit 2
fi

parts=()
for run_dir in "${RUN_DIRS[@]}"; do
    # The pair is recorded by the runner rather than parsed out of the path.
    pair="$(awk -F'\t' '$1=="pair"{print $2}' "$run_dir/run_config.tsv" 2>/dev/null || true)"
    pair="${pair:-$PAIR}"
    # Named, never guessed: the artifact decides both the supervised set and the
    # held-out edges, so it has to be the one these runs trained against.
    artifact="$(graph_artifact_path "$ARTIFACT_KEY" "$pair")"
    if [[ ! -f "$artifact" ]]; then
        echo "No graph artifact $artifact; set ARTIFACT_KEY to the graph this run tree trained on" >&2
        exit 2
    fi
    out="$OUT_DIR/heldout_geometry_$pair.csv"
    echo "  $pair: $run_dir"
    "$PYTHON_BIN" "$SCRIPT_DIR/heldout_geometry.py" \
        --runs "$run_dir/runs" \
        --cache "$CACHE_ROOT/$pair/$CORPUS_KEY/teacher_train.pt" \
        --artifact "$artifact" \
        --train-data "$CORPUS" \
        --pair "$pair" \
        --anchors "$ANCHORS" \
        --knn-k "$KNN_K" \
        --out "$out"
    parts+=("$out")
done

"$PYTHON_BIN" - "$OUT_DIR/heldout_geometry.csv" "${parts[@]}" <<'PY'
import csv, sys
from pathlib import Path

out, parts = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
rows, fields = [], []
for part in parts:
    with part.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for name in reader.fieldnames or []:
            if name not in fields:
                fields.append(name)
        rows.extend(reader)
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(f"merged {len(parts)} CSVs -> {out} ({len(rows)} rows)")
PY

echo
echo "Join $OUT_DIR/heldout_geometry.csv with runs/$SOURCE_EXPERIMENT/results.csv"
echo "on (pair, arm, seed)."
