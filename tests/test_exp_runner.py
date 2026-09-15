import os
import subprocess
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"

TABLE_SCRIPTS = (
    "table1_main.sh",
    "table2_cost.sh",
    "table3_exposure.sh",
    "table4_heldout.sh",
    "table4_loss_terms.sh",
    "table5_graph_k.sh",
    "table6_robustness.sh",
    "run_all.sh",
)


def _dry_run(script, **extra_env):
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "GPUS": "0",
        "PYTHON_BIN": str(PYTHON_BIN),
        "RUN_ID": "runner-test",
        "SEEDS": "42",
        **extra_env,
    }
    return subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "exp" / script)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("script", TABLE_SCRIPTS)
def test_every_script_parses(script):
    result = subprocess.run(
        ["bash", "-n", str(REPO_ROOT / "scripts" / "exp" / script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "script",
    [
        "table1_main.sh",
        "table2_cost.sh",
        "table3_exposure.sh",
        "table4_loss_terms.sh",
        "table6_robustness.sh",
    ],
)
def test_tables_after_table5_refuse_to_guess_graph_k(script):
    env = {k: v for k, v in os.environ.items() if k != "GRAPH_K"}
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "exp" / script)],
        cwd=REPO_ROOT,
        env={**env, "DRY_RUN": "1", "GPUS": "0", "PYTHON_BIN": str(PYTHON_BIN)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "GRAPH_K is not set" in result.stderr


def test_table5_needs_no_graph_k_and_plans_one_graph_per_k():
    env = {k: v for k, v in os.environ.items() if k != "GRAPH_K"}
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "exp" / "table5_graph_k.sh")],
        cwd=REPO_ROOT,
        env={**env, "DRY_RUN": "1", "GPUS": "0", "PYTHON_BIN": str(PYTHON_BIN)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "runs:   4" in result.stdout
    assert "graphs: 4" in result.stdout
    for k in (25, 50, 100, 200):
        assert f"graph_k_{k}" in result.stdout
        assert f"graph=k{k}" in result.stdout


def test_run_all_stops_after_table5_without_graph_k():
    env = {k: v for k, v in os.environ.items() if k != "GRAPH_K"}
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "exp" / "run_all.sh")],
        cwd=REPO_ROOT,
        env={**env, "DRY_RUN": "1", "GPUS": "0", "PYTHON_BIN": str(PYTHON_BIN)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "table5_graph_k sweep" in result.stdout
    assert "Stopping here on purpose" in result.stdout
    assert "table3_exposure sweep" not in result.stdout


def test_run_all_honours_from_and_to():
    result = _dry_run("run_all.sh", GRAPH_K="100", FROM="4", TO="6")
    assert result.returncode == 0, result.stderr
    assert "Table 4 -- table4_loss_terms.sh" in result.stdout
    assert "Table 4h -- table4_heldout.sh" in result.stdout
    assert "Table 6 -- table6_robustness.sh" in result.stdout
    for absent in ("table5_graph_k.sh", "table3_exposure.sh", "table1_main.sh"):
        assert absent not in result.stdout


def test_runner_rejects_an_unknown_graph_key():
    command = f"""
source {REPO_ROOT / 'scripts/exp/lib/run_arms.sh'}
GRAPH_SPEC='main|data/train_set/merged_3_data_5k_each.csv|'
ARMS_SPEC='bad|missing|ggpkd|'
PAIR=qwen3_0_6b_to_minilmv2_h384
SEEDS=42
GPUS=0
DRY_RUN=1
PYTHON_BIN={PYTHON_BIN}
run_arms {REPO_ROOT} runner_validation
"""
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "references unknown graph_key" in result.stderr


# --------------------------------------------------------------------------- #
# coverage.py -- Table 3's measured neighbours per row
# --------------------------------------------------------------------------- #


def _block_graph(n_items=640, k=8, block=64, seed=0):
    """Rows whose neighbours all sit in the same block of `block` texts."""
    rng = np.random.default_rng(seed)
    neighbors = np.empty((n_items, k), dtype=np.int64)
    for i in range(n_items):
        start = (i // block) * block
        choices = [j for j in range(start, start + block) if j != i]
        neighbors[i] = rng.choice(choices, size=k, replace=False)
    return neighbors


def test_random_batching_matches_the_analytic_count():
    from scripts.exp.coverage import measure

    neighbors = _block_graph()
    rows = {row["batching"]: row for row in measure(neighbors, 32, 20, seed=0)}
    random_row = rows["random"]
    analytic = random_row["analytic_random_neighbors_per_row"]
    assert analytic == pytest.approx(31 * 8 / 639, rel=1e-5)
    assert random_row["in_batch_neighbors_per_row"] == pytest.approx(analytic, rel=0.1)


def test_neighbor_batching_puts_more_neighbours_in_the_batch():
    from scripts.exp.coverage import measure

    # The greedy partition fills a batch from its seed's row only, so the effect
    # needs rows at least as wide as the batch.
    neighbors = _block_graph(k=40)
    rows = {row["batching"]: row for row in measure(neighbors, 32, 3, seed=1)}
    assert (
        rows["neighbor"]["in_batch_neighbors_per_row"]
        > 3 * rows["random"]["in_batch_neighbors_per_row"]
    )
    assert rows["neighbor"]["n_batches"] == rows["random"]["n_batches"]
