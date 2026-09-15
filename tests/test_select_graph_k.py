"""The Table 5 rule that picks the operating graph_k for every other table."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.exp.select_graph_k import load, select

ROOT = Path(__file__).resolve().parents[1]
PAIR = "qwen3_0_6b_to_minilmv2_h384"


def _write(path, rows):
    fields = ["experiment", "pair", "arm", "seed", "status", "score_Avg All",
              "graph_reciprocal_largest_frac", "graph_reciprocal_isolated"]  # fmt: skip
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"experiment": "table5_graph_k", "pair": PAIR, "status": "ok", **row})


def _row(k, avg, largest, seed=42, **extra):
    return {"arm": f"graph_k_{k}", "seed": seed, "score_Avg All": avg,
            "graph_reciprocal_largest_frac": largest, "graph_reciprocal_isolated": 0, **extra}  # fmt: skip


def test_connectivity_outvotes_an_early_score_plateau(tmp_path):
    # The 2026-09-15 student-graph numbers: Avg ties from k=50, the graph connects at 100.
    path = tmp_path / "t5.csv"
    _write(path, [_row(25, 75.00, 0.981), _row(50, 75.40, 0.994), _row(100, 75.57, 0.999),
                  _row(200, 75.52, 1.0)])  # fmt: skip
    chosen, detail = select(load(path, PAIR), tolerance=0.2, min_largest_frac=0.995)
    assert (detail["saturated"], detail["connected"], chosen) == (50, 100, 100)


def test_the_smallest_tied_k_wins_when_both_rules_agree(tmp_path):
    path = tmp_path / "t5.csv"
    _write(path, [_row(25, 74.0, 0.97), _row(50, 75.5, 0.999), _row(100, 75.6, 1.0)])
    chosen, _ = select(load(path, PAIR), tolerance=0.2, min_largest_frac=0.995)
    assert chosen == 50


def test_seeds_are_averaged_and_other_pairs_and_failures_ignored(tmp_path):
    path = tmp_path / "t5.csv"
    _write(path, [
        _row(50, 75.0, 0.999, seed=42), _row(50, 75.4, 0.999, seed=43),
        _row(100, 75.5, 0.999, seed=42), _row(100, 75.5, 0.999, seed=43),
        _row(50, 99.0, 0.999, seed=44, status="failed"),
        _row(50, 99.0, 0.999, seed=45, pair="other_pair"),
    ])  # fmt: skip
    groups = load(path, PAIR)
    assert {k: len(rows) for k, rows in groups.items()} == {50: 2, 100: 2}
    chosen, detail = select(groups, tolerance=0.2, min_largest_frac=0.995)
    assert detail["avg"][50] == pytest.approx(75.2)
    assert chosen == 100


def test_one_k_is_not_a_sweep(tmp_path):
    path = tmp_path / "t5.csv"
    _write(path, [_row(100, 75.5, 1.0)])
    with pytest.raises(ValueError):
        select(load(path, PAIR), tolerance=0.2, min_largest_frac=0.995)


def test_cli_writes_the_choice(tmp_path):
    path, out = tmp_path / "t5.csv", tmp_path / "graph_k"
    _write(path, [_row(50, 75.5, 0.999), _row(100, 75.6, 1.0)])
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/exp/select_graph_k.py"),
         "--csv", str(path), "--pair", PAIR, "--out", str(out)],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert out.read_text().strip() == "50"
    assert "chosen graph_k=50" in result.stdout
