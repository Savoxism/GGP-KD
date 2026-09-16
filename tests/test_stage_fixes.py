"""The fixes behind the 2026-09-12 sweep's broken tables, kept on the table scripts.

* pair_order read exactly 1.0 or 0.0 on a rectangular, sparse held-out mask;
* the runner overwrote multi-pair CSVs and could not re-run a single arm;
* every table script must plan exactly the arms story.md §5 names, with no flag
  the rewritten CLI no longer accepts.
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.distill.geometry import pair_order_accuracy

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"


# --------------------------------------------------------------------------- #
# pair_order on [anchors, corpus]
# --------------------------------------------------------------------------- #


def _rectangular(seed):
    torch.manual_seed(seed)
    corpus = F.normalize(torch.randn(2000, 16), dim=-1)
    anchors = torch.arange(0, 2000, 40)
    mask = torch.zeros(len(anchors), 2000, dtype=torch.bool)
    for row in range(len(anchors)):
        # Almost every admissible column lies past the first 50, which is all the
        # old sampler could reach.
        mask[row, torch.randperm(2000)[:12]] = True
    return corpus, anchors, mask


def test_pair_order_reads_a_sparse_mask_on_a_rectangular_matrix():
    corpus, anchors, mask = _rectangular(0)
    teacher_rows = corpus[anchors] @ corpus.t()
    assert (
        pair_order_accuracy(teacher_rows, teacher_rows, restrict=mask, n_triplets=20000)
        == 1.0
    )
    other = F.normalize(torch.randn(2000, 16), dim=-1)
    student_rows = other[anchors] @ other.t()
    accuracy = pair_order_accuracy(
        teacher_rows, student_rows, restrict=mask, n_triplets=20000
    )
    assert 0.4 < accuracy < 0.6


def test_pair_order_scores_only_the_masked_columns_of_each_row():
    corpus, anchors, mask = _rectangular(1)
    teacher_rows = corpus[anchors] @ corpus.t()
    corrupted = teacher_rows + torch.randn_like(teacher_rows) * 5.0
    student_rows = torch.where(mask, teacher_rows, corrupted)
    assert (
        pair_order_accuracy(teacher_rows, student_rows, restrict=mask, n_triplets=20000)
        == 1.0
    )


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def _export(root, out, pair, experiment="table1_main"):
    return subprocess.run(
        [
            sys.executable,
            "scripts/exp/export_runs.py",
            str(root),
            "--experiment",
            experiment,
            "--pair",
            pair,
            "--out",
            str(out),
            "--merge",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_export_merges_rows_by_pair_arm_and_seed(tmp_path):
    """Table 1 exports two pairs into one CSV; the second must not erase the first."""
    out = tmp_path / "results.csv"
    for pair in ("bge_m3_to_minilmv2_h768", "qwen3_4b_to_bert_base"):
        root = tmp_path / pair / "run"
        (root / "runs" / "ours" / "seed_42").mkdir(parents=True)
        (root / "run_config.tsv").write_text("commit\tabc123\n", encoding="utf-8")
        _export(root, out, pair)
    # A re-run of one pair replaces its row and keeps the other pair's.
    _export(tmp_path / "qwen3_4b_to_bert_base" / "run", out, "qwen3_4b_to_bert_base")

    with out.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert sorted(row["pair"] for row in rows) == [
        "bge_m3_to_minilmv2_h768",
        "qwen3_4b_to_bert_base",
    ]
    assert all(row["arm"] == "ours" for row in rows)
    assert all(row["status"] == "no_manifest" for row in rows)
    assert all(row["git_commit"] == "abc123" for row in rows)


def test_export_reads_the_new_config_metric_and_graph_columns(tmp_path):
    root = tmp_path / "run"
    seed_dir = root / "runs" / "ours" / "seed_42"
    seed_dir.mkdir(parents=True)
    manifest = {
        "run_id": "r1",
        "git": {"sha": "deadbeef"},
        "config": {
            "distill_method": "ggpkd",
            "graph_k": 100,
            "cal_weight": 0.5,
            "row_weight": 1.0,
            "row_set": "all",
            "row_target": "teacher",
            "row_columns": "graph",
            "row_reweight": False,
            "batch_local": False,
            "batch_sampler": "random",
            "fixed_bandwidth": False,
        },
        "artifact": {
            "metadata": {
                "graph_k": 100,
                "bandwidth": "median",
                "knn_mode": "directed",
                "neighbor_source": "teacher",
                "holdout_edge_frac": 0.2,
                "holdout_seed": 12345,
            },
            "graph_stats": {
                "reciprocal_components": 15,
                "reciprocal_isolated": 13,
                "reciprocal_largest_frac": 0.99,
                "reciprocity": 0.41,
                "target_kl_uniform": 0.3,
            },
        },
    }
    (seed_dir / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    epoch = {
        "epoch": 5,
        "train": {
            "loss_total": 1.5,
            "loss_cal": 0.7,
            "loss_row": 1.1,
            "pool_size": 4700.0,
            "row_exposed_mass": 0.61,
            "peak_memory_mb": 14438.0,
            "encoded_texts_cum": 123456,
        },
    }
    (seed_dir / "epochs.jsonl").write_text(json.dumps(epoch) + "\n", encoding="utf-8")
    out = tmp_path / "results.csv"
    _export(root, out, "p", experiment="table3_exposure")

    with out.open(newline="", encoding="utf-8") as handle:
        (row,) = list(csv.DictReader(handle))
    assert row["cfg_graph_k"] == "100"
    assert row["cfg_cal_weight"] == "0.5"
    assert row["cfg_row_set"] == "all"
    assert row["cfg_batch_local"] == "False"
    assert row["cfg_holdout_seed"] == "12345"
    # The artifact wins over the config for how the graph was built.
    assert row["cfg_fixed_bandwidth"] == "True"
    assert row["train_loss_cal"] == "0.7"
    assert row["train_row_exposed_mass"] == "0.61"
    assert row["train_encoded_texts_cum"] == "123456"
    assert row["graph_reciprocal_components"] == "15"
    assert row["graph_target_kl_uniform"] == "0.3"
    assert row["git_commit"] == "deadbeef"
    for gone in ("cfg_support_policy", "train_loss_rel", "train_candidates_per_anchor"):
        assert gone not in row


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def _run_arms(arms):
    corpus = "data/train_set/merged_3_data_5k_each.csv"
    command = f"""
source {REPO_ROOT / "scripts/exp/lib/run_arms.sh"}
GRAPH_SPEC='main|{corpus}|
other|{corpus}|'
ARMS_SPEC='a|main|ggpkd|
b|other|ggpkd|'
PAIR=qwen3_0_6b_to_minilmv2_h384
SEEDS=42,43
GPUS=0
DRY_RUN=1
ARMS={arms}
PYTHON_BIN={PYTHON_BIN}
run_arms {REPO_ROOT} runner_validation
"""
    return subprocess.run(
        ["bash", "-c", command],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_runner_runs_only_the_named_arms_and_their_graphs():
    result = _run_arms("b")
    assert result.returncode == 0, result.stderr
    assert "runs:   2" in result.stdout
    assert "graphs: 1" in result.stdout
    assert "graph=other" in result.stdout and "graph=main" not in result.stdout


def test_runner_rejects_an_arm_the_sweep_does_not_define():
    result = _run_arms("zzz")
    assert result.returncode == 2
    assert "does not define" in result.stderr


# --------------------------------------------------------------------------- #
# Table dry runs
# --------------------------------------------------------------------------- #


def _planned(output):
    """{arm: (graph, method, flags)} from run_arms' DRY_RUN listing."""
    arms = {}
    for line in output.splitlines():
        if " graph=" not in line or " method=" not in line:
            continue
        label, rest = line.strip().split(None, 1)
        graph = rest.split("graph=", 1)[1].split()[0]
        method = rest.split("method=", 1)[1].split()[0]
        flags = rest.split("flags:", 1)[1].strip()
        arms[label] = (graph, method, flags)
    return arms


def _table(script, **extra_env):
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "GPUS": "0",
        "GRAPH_K": "100",
        "SEEDS": "42",
        "PYTHON_BIN": str(PYTHON_BIN),
        "RUN_ID": "stage-fix-test",
        **extra_env,
    }
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "exp" / script)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    return result.stdout, output


TABLE_MATRIX = {
    "table3_exposure.sh": {
        "pointwise_random": ("main", "pointwise", ""),
        "pointwise_neighbor": ("main", "pointwise", "--batch_sampler neighbor"),
        "in_batch_random": ("main", "ggpkd", "--batch_local"),
        "in_batch_neighbor": ("main", "ggpkd", "--batch_local --batch_sampler neighbor"),
        "anchor_rows": ("main", "ggpkd", "--row_set anchors"),
        "ours": ("main", "ggpkd", ""),
        "uniform_target": ("main", "ggpkd", "--row_target uniform"),
        "random_columns": ("main", "ggpkd", "--row_columns random"),
    },
    "table4_loss_terms.sh": {
        "ours": ("main_holdout", "ggpkd", ""),
        "cal_off": ("main_holdout", "ggpkd", "--cal_weight 0"),
        "row_off": ("main_holdout", "ggpkd", "--row_weight 0"),
        "anchors_only": ("main_holdout", "ggpkd", "--row_set anchors"),
        "anchors_excluded": ("main_holdout", "ggpkd", "--row_set non_anchors"),
        "holdout_strict": ("main_holdout_strict", "ggpkd", ""),
    },
    "table7_controls.sh": {
        "shuffled_target": ("main", "ggpkd", "--row_target shuffled"),
        "random_pool": ("main", "ggpkd", "--pool_source random"),
        "dense_rows": ("main", "ggpkd", "--row_columns pool"),
        "dense_rows_only": ("main", "ggpkd", "--row_columns pool --cal_weight 0"),
    },
    "table6_robustness.sh": {
        "cal_0p25": ("main", "ggpkd", "--cal_weight 0.25"),
        "cal_1p0": ("main", "ggpkd", "--cal_weight 1.0"),
        "row_reweight": ("main", "ggpkd", "--row_reweight"),
        "student_knn": ("student", "ggpkd", ""),
        "mutual_knn": ("mutual", "ggpkd", ""),
        "global_tau": ("median_tau", "ggpkd", ""),
    },
    "table2_cost.sh": {
        "pointwise": ("main", "pointwise", ""),
        "in_batch": ("main", "ggpkd", "--batch_local"),
        "ours": ("main", "ggpkd", ""),
    },
}


@pytest.mark.parametrize("script", sorted(TABLE_MATRIX))
def test_table_dry_runs_plan_the_fixed_matrix(script):
    stdout, _ = _table(script)
    expected = TABLE_MATRIX[script]
    assert _planned(stdout) == expected
    assert f"runs:   {len(expected)}" in stdout


def test_seeds_multiply_the_run_count():
    stdout, _ = _table("table3_exposure.sh", SEEDS="42,43,44")
    assert "runs:   24" in stdout


def test_table4_trains_every_arm_on_the_same_withheld_pairs():
    """Two artifacts, one split.

    `holdout_strict` needs its own artifact because it recomputes tau_j on the
    surviving edges, but the withheld pairs are a pure function of (frac, seed),
    which both graphs take from the same two variables -- so the held-out columns
    of table4_heldout.sh score every arm on the same relations.
    """
    stdout, _ = _table("table4_loss_terms.sh")
    assert "holdout 0.2 (seed 12345)" in stdout
    keys = {graph for graph, _, _ in _planned(stdout).values()}
    assert keys == {"main_holdout", "main_holdout_strict"}
    source = (REPO_ROOT / "scripts" / "exp" / "table4_loss_terms.sh").read_text()
    assert source.count("--holdout_edge_frac $HOLDOUT_FRAC --holdout_seed $HOLDOUT_SEED") == 2


def test_table4_refuses_to_run_without_a_holdout():
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "GPUS": "0",
        "GRAPH_K": "100",
        "PYTHON_BIN": str(PYTHON_BIN),
        "HOLDOUT_FRAC": "0",
    }
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "exp" / "table4_loss_terms.sh")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "need a holdout" in result.stderr


def test_table6_varies_one_thing_per_arm_and_names_its_reference():
    stdout, output = _table("table6_robustness.sh")
    assert "graphs: 4" in stdout
    assert "Table 5's graph_k_100 run" in output
    planned = _planned(stdout)
    # Graph variants carry no objective flags; objective variants use the main graph.
    for arm, (graph, _, flags) in planned.items():
        assert (graph == "main") == bool(flags), arm


def test_table2_forces_one_job_per_gpu():
    stdout, output = _table("table2_cost.sh", JOBS_PER_GPU="3")
    assert "1 job(s) per GPU" in stdout
    assert "JOBS_PER_GPU=3 ignored" in output


def test_table1_runs_each_pair_into_one_merged_csv():
    stdout, _ = _table("table1_main.sh")
    assert "Table 1 -- pair bge_m3_to_minilmv2_h768" in stdout
    assert "Table 1 -- pair qwen3_4b_to_bert_base" in stdout
    assert stdout.count("runs:   1") == 2
    assert "results/table1_main/bge_m3_to_minilmv2_h768/stage-fix-test" in stdout
    assert "results/table1_main/qwen3_4b_to_bert_base/stage-fix-test" in stdout


def test_table1_skips_the_pair_table5_already_measured():
    stdout, output = _table(
        "table1_main.sh", PAIRS="qwen3_0_6b_to_minilmv2_h384,bge_m3_to_minilmv2_h768"
    )
    assert "skipping qwen3_0_6b_to_minilmv2_h384" in output
    assert stdout.count("runs:   1") == 1


def test_table4_heldout_dry_run_scores_nothing():
    stdout, _ = _table("table4_heldout.sh")
    assert "graph_main_holdout.pt" in stdout
    assert "heldout_same_component" in stdout
    assert "DRY_RUN: nothing scored" in stdout
