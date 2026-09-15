#!/usr/bin/env python3
"""Pick the operating graph_k from Table 5 (story.md §5, Table 5).

Two readings of "enough neighbours", and the larger k of the two wins:

    saturation    the smallest k whose mean Avg is within `--tolerance` of the best
                  k. 0.2 is the story's noise rule for a one-seed difference.
    connectivity  the smallest k whose reciprocal kNN graph is about connected
                  (giant-component share >= `--min-largest-frac`). Proposition 2:
                  past that point L_row pins the teacher geometry up to a handful
                  of offsets, and a wider row buys encode cost, not signal.

Small beats tied-and-large: it is the whole cost argument. The choice is printed
with every k's numbers, so a human can overrule it with GRAPH_K=<k>.

    python scripts/exp/select_graph_k.py --pair qwen3_0_6b_to_minilmv2_h384 \
        --out runs/launch/<run_id>/graph_k
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

SCORE = "score_Avg All"
LARGEST = "graph_reciprocal_largest_frac"
ISOLATED = "graph_reciprocal_isolated"


def _float(value: str | None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load(path: Path, pair: str) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            arm = row.get("arm", "")
            if row.get("pair") != pair or row.get("status") != "ok":
                continue
            if not arm.startswith("graph_k_") or _float(row.get(SCORE)) is None:
                continue
            groups[int(arm.removeprefix("graph_k_"))].append(row)
    return dict(groups)


def _mean(rows: list[dict], key: str) -> float | None:
    values = [v for v in (_float(r.get(key)) for r in rows) if v is not None]
    return sum(values) / len(values) if values else None


def select(
    groups: dict[int, list[dict]], tolerance: float, min_largest_frac: float
) -> tuple[int, dict]:
    if len(groups) < 2:
        raise ValueError(f"need at least two graph_k values, found {sorted(groups)}")
    avg = {k: _mean(rows, SCORE) for k, rows in groups.items()}
    largest = {k: _mean(rows, LARGEST) for k, rows in groups.items()}
    best = max(avg.values())
    saturated = min(k for k, value in avg.items() if value >= best - tolerance)
    connected_ks = [
        k for k, value in largest.items() if value is not None and value >= min_largest_frac
    ]
    if any(value is None for value in largest.values()):
        # An older export without graph columns: fall back to saturation alone.
        connected = saturated
    elif connected_ks:
        connected = min(connected_ks)
    else:
        connected = max(groups)
    chosen = max(saturated, connected)
    return chosen, {
        "best_avg": best,
        "saturated": saturated,
        "connected": connected,
        "avg": avg,
        "largest": largest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default="runs/table5_graph_k/results.csv", type=Path)
    parser.add_argument("--pair", default="qwen3_0_6b_to_minilmv2_h384")
    parser.add_argument("--tolerance", type=float, default=0.2)
    parser.add_argument("--min-largest-frac", type=float, default=0.995)
    parser.add_argument("--out", type=Path, help="write the chosen k to this file")
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"no Table 5 results at {args.csv}", file=sys.stderr)
        return 1
    groups = load(args.csv, args.pair)
    try:
        chosen, detail = select(groups, args.tolerance, args.min_largest_frac)
    except ValueError as error:
        print(f"cannot select graph_k: {error}", file=sys.stderr)
        return 1

    print(f"{'k':>5} {'Avg':>8} {'seeds':>5} {'largest':>8} {'isolated':>8}")
    for k in sorted(groups):
        rows = groups[k]
        largest = detail["largest"][k]
        isolated = _mean(rows, ISOLATED)
        print(
            f"{k:>5} {detail['avg'][k]:>8.2f} {len(rows):>5} "
            f"{'-' if largest is None else f'{largest:.4f}':>8} "
            f"{'-' if isolated is None else f'{isolated:.0f}':>8}"
        )
    print(
        f"saturation k={detail['saturated']} (within {args.tolerance} of best "
        f"{detail['best_avg']:.2f}); connectivity k={detail['connected']} "
        f"(largest component >= {args.min_largest_frac}); chosen graph_k={chosen}"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(f"{chosen}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
