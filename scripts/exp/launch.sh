#!/usr/bin/env bash
# One command for the whole plan of story.md §5: preflight, Table 5, the graph_k
# choice, then Tables 3, 4, 4h, 6, 1, 2 at that k. Runs detached, so an ssh drop
# does not kill it.
#
#   GPUS=4,5 JOBS_PER_GPU=2 bash scripts/exp/launch.sh start               # pilot, 1 seed
#   PROFILE=paper GPUS=4,5 JOBS_PER_GPU=2 bash scripts/exp/launch.sh start # 3 seeds
#   bash scripts/exp/launch.sh status [RUN_ID]
#   bash scripts/exp/launch.sh stop   [RUN_ID]
#
# Env:
#   PROFILE        pilot (SEEDS=42) or paper (SEEDS=42,43,44)
#   SEEDS          overrides the profile
#   GRAPH_K        skip the automatic choice (Table 5 then runs at that k only
#                  if its results do not already hold it)
#   FROM / TO      restrict the tables after the choice (3, 4, 4h, 6, 1, 2)
#   TOLERANCE, MIN_LARGEST_FRAC   the selection rule (select_graph_k.py)
#   GPUS, JOBS_PER_GPU, PAIRS, CACHE_ROOT, PYTHON_BIN   forwarded to the tables
#   FOREGROUND=1   run the controller in this shell instead of detaching
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

LAUNCH_ROOT="$REPO_ROOT/runs/launch"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
TABLE5_PAIR="qwen3_0_6b_to_minilmv2_h384"
RUNS_PER_SEED=28 # Table 5: 4, 3: 8, 4: 5, 6: 6, 1: 2, 2: 3

latest_run() {
    [[ -f "$LAUNCH_ROOT/latest" ]] || { echo "no launch has been started" >&2; exit 1; }
    cat "$LAUNCH_ROOT/latest"
}

run_dir_of() {
    local run_id="${1:-$(latest_run)}"
    local dir="$LAUNCH_ROOT/$run_id"
    [[ -d "$dir" ]] || { echo "no launch named $run_id under $LAUNCH_ROOT" >&2; exit 1; }
    printf '%s' "$dir"
}

alive() {
    local pid_file="$1/controller.pid"
    [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

phase() {
    printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$RUN_DIR/phases.log"
    echo "== $*"
}

preflight() {
    local problems=0
    [[ -x "$PYTHON_BIN" ]] || { echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2; problems=1; }
    [[ -n "${GPUS:-}" ]] || { echo "set GPUS (e.g. GPUS=4,5)" >&2; problems=1; }
    [[ -f data/train_set/merged_3_data_5k_each.csv ]] || {
        echo "missing training corpus data/train_set/merged_3_data_5k_each.csv" >&2
        problems=1
    }
    case "${PROFILE:-pilot}" in pilot | paper) ;; *)
        echo "PROFILE must be pilot or paper, got: $PROFILE" >&2
        problems=1
        ;;
    esac
    if command -v nvidia-smi >/dev/null 2>&1 && [[ -n "${GPUS:-}" ]]; then
        local available gpu
        available="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')"
        IFS=',' read -r -a requested <<< "$GPUS"
        for gpu in "${requested[@]}"; do
            grep -qx "$gpu" <<< "$available" || {
                echo "GPU $gpu is not present (nvidia-smi lists: ${available//$'\n'/,})" >&2
                problems=1
            }
        done
    fi
    for script in "$SCRIPT_DIR"/table*.sh "$SCRIPT_DIR/run_all.sh" "$SCRIPT_DIR/lib/"*.sh; do
        bash -n "$script" || problems=1
    done
    if (( problems == 0 )); then
        "$PYTHON_BIN" -c "import main" >/dev/null || {
            echo "main.py does not import with $PYTHON_BIN" >&2
            problems=1
        }
    fi
    if [[ -f "$LAUNCH_ROOT/latest" ]]; then
        local previous="$LAUNCH_ROOT/$(cat "$LAUNCH_ROOT/latest")"
        if [[ -d "$previous" ]] && alive "$previous"; then
            echo "launch $(basename "$previous") is still running; stop it first" >&2
            problems=1
        fi
    fi
    return $problems
}

controller() {
    RUN_DIR="$1"
    export RUN_ID
    RUN_ID="$(basename "$RUN_DIR")"
    # shellcheck source=/dev/null
    source "$RUN_DIR/env.sh"
    trap 'code=$?; printf "%s\n" "$code" > "$RUN_DIR/exit"; phase "controller exit $code"' EXIT

    phase "start run_id=$RUN_ID seeds=$SEEDS gpus=$GPUS jobs_per_gpu=${JOBS_PER_GPU:-1}"
    if [[ -z "${GRAPH_K:-}" ]]; then
        phase "table 5: graph_k sweep"
        bash "$SCRIPT_DIR/table5_graph_k.sh"
        phase "select graph_k"
        "$PYTHON_BIN" "$SCRIPT_DIR/select_graph_k.py" --pair "$TABLE5_PAIR" \
            --tolerance "${TOLERANCE:-0.2}" --min-largest-frac "${MIN_LARGEST_FRAC:-0.995}" \
            --out "$RUN_DIR/graph_k"
        GRAPH_K="$(cat "$RUN_DIR/graph_k")"
        export GRAPH_K
        FROM="${FROM:-3}"
    else
        printf '%s\n' "$GRAPH_K" > "$RUN_DIR/graph_k"
        export GRAPH_K
    fi
    phase "tables at graph_k=$GRAPH_K (FROM=${FROM:-5} TO=${TO:-2})"
    FROM="${FROM:-5}" TO="${TO:-2}" bash "$SCRIPT_DIR/run_all.sh"
    phase "all tables done"
}

cmd_start() {
    preflight
    local profile="${PROFILE:-pilot}"
    local seeds="${SEEDS:-}"
    if [[ -z "$seeds" ]]; then
        [[ "$profile" == "paper" ]] && seeds="42,43,44" || seeds="42"
    fi
    local run_id
    run_id="launch-$(date -u +%Y%m%dT%H%M%SZ)"
    RUN_DIR="$LAUNCH_ROOT/$run_id"
    mkdir -p "$RUN_DIR"

    local commit dirty=""
    commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    [[ -n "$(git status --porcelain 2>/dev/null)" ]] && dirty="+dirty"
    local n_seeds
    IFS=',' read -r -a seed_list <<< "$seeds"
    n_seeds=${#seed_list[@]}

    # Everything the controller needs, pinned at start so a later edit of the
    # caller's shell cannot change a run half-way through.
    {
        printf 'export PROFILE=%q SEEDS=%q GPUS=%q JOBS_PER_GPU=%q\n' \
            "$profile" "$seeds" "$GPUS" "${JOBS_PER_GPU:-1}"
        printf 'export PYTHON_BIN=%q SOURCE_COMMIT=%q\n' "$PYTHON_BIN" "$commit$dirty"
        for name in GRAPH_K FROM TO TOLERANCE MIN_LARGEST_FRAC PAIRS CACHE_ROOT; do
            [[ -n "${!name:-}" ]] && printf 'export %s=%q\n' "$name" "${!name}"
        done
    } > "$RUN_DIR/env.sh"
    printf '%s\n' "$run_id" > "$LAUNCH_ROOT/latest"

    echo "run_id:   $run_id"
    echo "commit:   $commit$dirty"
    echo "profile:  $profile (seeds $seeds)"
    echo "gpus:     $GPUS x ${JOBS_PER_GPU:-1} job(s)"
    echo "runs:     ~$((RUNS_PER_SEED * n_seeds)) training runs (28 per seed)"
    echo "log:      $RUN_DIR/controller.log"
    [[ -n "$dirty" ]] && echo "WARNING: the working tree has uncommitted changes"

    if [[ "${FOREGROUND:-0}" == "1" ]]; then
        controller "$RUN_DIR" 2>&1 | tee -a "$RUN_DIR/controller.log"
        return "${PIPESTATUS[0]}"
    fi
    setsid nohup bash "$SCRIPT_DIR/launch.sh" _controller "$RUN_DIR" \
        > "$RUN_DIR/controller.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$RUN_DIR/controller.pid"
    echo "pid:      $! (detached)"
    echo
    echo "  bash scripts/exp/launch.sh status"
    echo "  tail -f $RUN_DIR/controller.log"
}

cmd_status() {
    RUN_DIR="$(run_dir_of "${1:-}")"
    local state="finished"
    if alive "$RUN_DIR"; then
        state="running (pid $(cat "$RUN_DIR/controller.pid"))"
    elif [[ -f "$RUN_DIR/exit" ]]; then
        state="exited $(cat "$RUN_DIR/exit")"
    fi
    echo "run:      $(basename "$RUN_DIR")"
    echo "state:    $state"
    [[ -f "$RUN_DIR/graph_k" ]] && echo "graph_k:  $(cat "$RUN_DIR/graph_k")"
    local log="$RUN_DIR/controller.log"
    if [[ -f "$log" ]]; then
        echo "runs:     $(grep -c '^Completed:' "$log" || true) completed," \
            "$(grep -c '^FAILED:' "$log" || true) failed"
    fi
    echo
    [[ -f "$RUN_DIR/phases.log" ]] && cat "$RUN_DIR/phases.log"
    echo
    [[ -f "$log" ]] && tail -n 15 "$log"
}

cmd_stop() {
    RUN_DIR="$(run_dir_of "${1:-}")"
    if ! alive "$RUN_DIR"; then
        echo "$(basename "$RUN_DIR") is not running"
        return 0
    fi
    local pid
    pid="$(cat "$RUN_DIR/controller.pid")"
    # setsid made the controller a process-group leader; run_arms traps TERM and
    # stops its training children before it exits.
    kill -TERM -- "-$pid"
    echo "sent TERM to process group $pid"
}

case "${1:-}" in
    start) cmd_start ;;
    status) cmd_status "${2:-}" ;;
    stop) cmd_stop "${2:-}" ;;
    _controller) controller "$2" ;;
    *)
        sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
