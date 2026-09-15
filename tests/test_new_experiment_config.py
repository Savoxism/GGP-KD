"""CLI, evaluation-table and bandwidth tests that do not depend on the GGPKD objective.

The GGPKD objective, graph, collate and train step are covered in test_ggpkd_core.py.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import main
from distiller import KnowledgeDistiller, add_domain_averages
from src.distill.checkpointing import save_student_weights
from src.ggpkd.graph_builder import _knn_bandwidths


def test_eval_every_cli_is_explicit_and_rejects_negative_values(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--method", "ggpkd", "--eval_every", "0"],
    )
    args = main.parse_args()
    assert main.get_config(args.method, args).eval_every == 0

    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--method", "ggpkd", "--eval_every", "-1"],
    )
    args = main.parse_args()
    with pytest.raises(ValueError, match="non-negative"):
        main.get_config(args.method, args)


def test_evaluation_table_renders_every_family(capsys):
    """Nothing called this before, so a broken signature stayed invisible."""
    from src.distill.benchmarks import print_evaluation_table

    results = {
        "classification": {
            "data/test_set/emotion_test.csv": {"accuracy": 0.7, "f1": 0.62}
        },
        "pair": {
            "data/test_set/wic_test.csv": {"accuracy": 0.62, "average_precision": 0.65}
        },
        "sts": {"data/test_set/stsb_test.csv": 0.766},
    }
    print_evaluation_table(current_epoch=3, split="test", results=results)

    out = capsys.readouterr().out
    assert "TEST - EPOCH 4" in out
    for token in ("emotion", "wic", "stsb", "F1", "AP", "Spearman"):
        assert token in out, f"{token} missing from the table"
    # The in-domain group is complete here, so its average prints; the
    # out-of-domain group is not, so neither it nor the overall average does.
    assert "AVG IN-DOMAIN" in out
    assert "AVG OUT-OF-DOMAIN" not in out
    assert "MEAN" not in out

    print_evaluation_table(current_epoch=0, split="test", results=results, final=True)
    assert "FINAL TEST" in capsys.readouterr().out


def test_domain_averages_follow_the_paper_metric_protocol():
    results = {
        "classification": {
            "data/test_set/banking77_test.csv": {"accuracy": 0.01, "f1": 0.9},
            "data/test_set/emotion_test.csv": {"accuracy": 0.02, "f1": 0.6},
            "data/test_set/tweet_test.csv": {"accuracy": 0.03, "f1": 0.7},
        },
        "pair": {
            "data/test_set/mrpc_test.csv": {
                "accuracy": 0.04,
                "average_precision": 0.8,
            },
            "data/test_set/scitail_test.csv": {
                "accuracy": 0.05,
                "average_precision": 0.75,
            },
            "data/test_set/wic_test.csv": {
                "accuracy": 0.06,
                "average_precision": 0.65,
            },
        },
        "sts": {
            "data/test_set/sick_test.csv": 0.72,
            "data/test_set/sts12_test.csv": 0.68,
            "data/test_set/stsb_test.csv": 0.74,
        },
    }

    enriched = add_domain_averages(results)

    assert enriched["avg_in"] == 66.33
    assert enriched["avg_out"] == 75.83
    assert enriched["avg"] == 72.67
    assert enriched["classification"] is results["classification"]


def test_rkd_cli_uses_paper_defaults_and_accepts_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--method",
            "rkd",
            "--rkd_distance_weight",
            "0.5",
            "--rkd_angle_weight",
            "3.0",
            "--cache_path",
            "cache/rkd/teacher.pt",
            "--pooling_method",
            "mean",
        ],
    )

    args = main.parse_args()
    config = main.get_config(args.method, args)

    assert config.distill_method == "rkd"
    assert config.rkd_distance_weight == 0.5
    assert config.rkd_angle_weight == 3.0
    assert config.w_task == 0.0
    assert config.batch_size == 128
    assert config.epochs == 80
    assert config.learning_rate == 1e-4
    assert config.weight_decay == 1e-5
    assert config.rkd_lr_decay_epochs == (40, 60)
    assert config.cache_path == "cache/rkd/teacher.pt"
    assert config.pooling_method == "mean"


def test_knn_bandwidth_makes_the_kth_neighbour_k_times_less_likely():
    # The rule with no constant in it: tau_i = (s(1) - s(k)) / log(k), so the k-th
    # retrieved neighbour sits log(k) nats below the nearest and is therefore
    # exactly k times less likely. This is what replaced the target perplexity.
    rng = np.random.default_rng(0)
    for k in (8, 40, 200):
        scores = np.sort(rng.normal(size=(4, k)) * 0.1 + 0.7, axis=1)[:, ::-1].copy()
        taus = _knn_bandwidths(scores, k)
        ratio = np.exp((scores[:, 0] - scores[:, k - 1]) / taus)
        np.testing.assert_allclose(ratio, k, rtol=1e-9)


def test_knn_bandwidth_is_invariant_to_affine_rescaling():
    # The property that ruled out a single fixed temperature, and the reason this
    # rule is admissible as a replacement: a teacher whose cosines are spread
    # differently must produce the *same* transition row.
    rng = np.random.default_rng(2)
    scores = np.sort(rng.normal(size=(1, 48)) * 0.1 + 0.7, axis=1)[:, ::-1].copy()

    def row(raw):
        tau = float(_knn_bandwidths(raw, raw.shape[1])[0])
        logits = (raw[0] - raw[0].max()) / tau
        weights = np.exp(logits)
        return weights / weights.sum(), tau

    base_probs, base_tau = row(scores)
    for a, b in ((3.0, 0.0), (0.25, 0.0), (2.0, -1.5), (0.5, 4.0)):
        probs, tau = row(a * scores + b)
        np.testing.assert_allclose(probs, base_probs, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(tau, a * base_tau, rtol=1e-9)

    # The fixed-temperature row, by contrast, sharpens when the scale is stretched.
    fixed = np.exp((scores[0] - scores[0].max()) / 0.05)
    fixed /= fixed.sum()
    stretched = np.exp((3.0 * scores[0] - (3.0 * scores[0]).max()) / 0.05)
    stretched /= stretched.sum()
    assert not np.allclose(fixed, stretched, atol=1e-6)


def test_knn_bandwidth_has_a_fixed_sample_size_and_cannot_clamp():
    # The failure the perplexity solve had: it ran on the mutual-filtered list,
    # where degree varies, so a row with degree at or below the requested
    # perplexity could not reach the target entropy and was solved against its own
    # ceiling instead. Here every row reads exactly k raw scores, so there is no
    # degree to fall short and no target to miss.
    rng = np.random.default_rng(3)
    scores = np.sort(rng.normal(size=(500, 32)) * 0.1 + 0.7, axis=1)[:, ::-1].copy()
    taus = _knn_bandwidths(scores, 32)
    assert taus.shape == (500,)
    assert np.all(np.isfinite(taus)) and np.all(taus > 0.0)
    # Well conditioned: no heavy tail of near-uniform rows at enormous tau.
    assert taus.max() / taus.min() < 10.0


def test_knn_bandwidth_floors_a_row_with_no_scale_of_its_own():
    # All k neighbours tied in cosine: the row is uniform at every temperature, so
    # the floor only keeps the division finite.
    tied = np.full((1, 16), 0.8, dtype=np.float64)
    tau = float(_knn_bandwidths(tied, 16)[0])
    assert tau > 0.0 and np.isfinite(tau)

    with pytest.raises(ValueError, match="at least 2 neighbours"):
        _knn_bandwidths(np.zeros((1, 1)), 1)


def test_final_student_weights_are_idempotent(tmp_path):
    distiller = KnowledgeDistiller.__new__(KnowledgeDistiller)
    distiller.config = SimpleNamespace(
        weights_dir=str(tmp_path / "weights"),
        save_dir=str(tmp_path / "run"),
        student_model_name="student",
        teacher_model_name="teacher",
    )
    distiller.model_student = torch.nn.Linear(2, 2)
    distiller._saved_student_weight_epochs = set()

    save_student_weights(distiller, 4)
    save_student_weights(distiller, 4)

    files = list((tmp_path / "weights").glob("*.pt"))
    assert [path.name for path in files] == ["student_epoch_5.pt"]
    payload = torch.load(files[0], map_location="cpu", weights_only=False)
    assert payload["epoch"] == 5

