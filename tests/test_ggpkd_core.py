"""GGPKD as defined in story.md §2: config, graph, pool collate, loss, sampler, step.

Each test pins a property that would still produce a clean-looking number if it
were false: a row set that silently differs from its label, a holdout that leaks
through the calibration term, an inclusion weight that is not the real inclusion
probability, a pool that encodes a text twice.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import main
from config import GGPKDConfig
from distiller import KnowledgeDistiller
from src.criterions.ggpkd_distillation import GGPKDDistillation
from src.data_utils.batch_samplers import NeighborBatchSampler, batch_relevance_stats
from src.data_utils.ggpkd_dataset import GGPKDCollate
from src.ggpkd.graph_builder import (
    _build_transition,
    _compute_topk_cosine,
    _knn_bandwidths,
    _pad_rows,
    build_or_load_ggpkd_artifact,
    heldout_edge_mask,
    pool_inclusion_probability,
    reciprocal_component_labels,
    reciprocal_component_stats,
)
from src.methods import get_method

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _graph(n_items=80, dim=12, k=8, seed=0, holdout=0.0):
    torch.manual_seed(seed)
    teacher = torch.randn(n_items, dim)
    top_indices, top_scores = _compute_topk_cosine(teacher, k=k, device=torch.device("cpu"))
    temps = _knn_bandwidths(top_scores, k)
    rows, probs, _, _, _ = _build_transition(
        top_indices, top_scores, k, temps, holdout_edge_frac=holdout, holdout_seed=7
    )
    neighbors, padded_probs = _pad_rows(rows, probs)
    return {
        "teacher": teacher,
        "neighbors": neighbors,
        "probs": padded_probs,
        "temps": temps,
    }


def _criterion(graph, **kwargs):
    return GGPKDDistillation(
        teacher_embeddings=graph["teacher"],
        transition_neighbors=torch.from_numpy(graph["neighbors"]),
        transition_probs=torch.from_numpy(graph["probs"]),
        row_temps=torch.from_numpy(graph["temps"]).float(),
        cal_temp=float(np.median(graph["temps"])),
        **kwargs,
    )


class _Tok:
    """Token id = corpus index + 1, repeated to give the texts different lengths."""

    pad_token_id = 0

    def __call__(self, texts, max_length, truncation):
        ids = []
        for text in texts:
            index = int(text.split()[1])
            ids.append([index + 1] * (1 + index % 3))
        return {"input_ids": ids}


def _texts(n_items):
    return [f"t {i}" for i in range(n_items)]


def _batch(graph, anchors, holdout=0.0, chunk=16):
    collate = GGPKDCollate(
        _Tok(),
        16,
        _texts(graph["neighbors"].shape[0]),
        graph["neighbors"],
        holdout_edge_frac=holdout,
        holdout_seed=7,
        encode_chunk_size=chunk,
    )
    return collate([{"idx": int(i)} for i in anchors])


def _args(monkeypatch, *flags):
    monkeypatch.setattr(sys, "argv", ["main.py", "--method", "ggpkd", *flags])
    args = main.parse_args()
    return main.get_config(args.method, args)


# --------------------------------------------------------------------------- #
# Config and CLI
# --------------------------------------------------------------------------- #


def test_config_defaults_are_the_method():
    config = GGPKDConfig()
    assert config.graph_k == 100
    assert config.neighbor_source == "teacher"
    assert config.knn_mode == "directed"
    assert config.fixed_bandwidth is False
    assert config.holdout_edge_frac == 0.0
    assert (config.cal_weight, config.row_weight) == (0.5, 1.0)
    assert (config.row_set, config.row_target, config.row_columns) == (
        "all",
        "teacher",
        "graph",
    )
    assert config.row_reweight is False
    assert config.batch_local is False
    assert config.batch_sampler == "random"


def test_cli_sets_every_switch(monkeypatch):
    config = _args(
        monkeypatch,
        "--graph_k", "50",
        "--neighbor_source", "student",
        "--knn_mode", "mutual",
        "--fixed_bandwidth",
        "--holdout_edge_frac", "0.2",
        "--holdout_seed", "3",
        "--cal_weight", "0.25",
        "--row_weight", "2",
        "--row_set", "non_anchors",
        "--row_target", "uniform",
        "--row_reweight",
        "--ggpkd_cache_path", "cache/graph.pt",
    )  # fmt: skip
    assert config.graph_k == 50
    assert config.neighbor_source == "student"
    assert config.knn_mode == "mutual"
    assert config.fixed_bandwidth is True
    assert (config.holdout_edge_frac, config.holdout_seed) == (0.2, 3)
    assert (config.cal_weight, config.row_weight) == (0.25, 2.0)
    assert (config.row_set, config.row_target) == ("non_anchors", "uniform")
    assert config.row_reweight is True
    assert config.ggpkd_cache_path == "cache/graph.pt"


def test_batch_local_turns_the_row_term_off(monkeypatch):
    config = _args(monkeypatch, "--batch_local", "--batch_sampler", "neighbor")
    assert config.batch_local is True
    assert config.row_weight == 0.0
    assert config.batch_sampler == "neighbor"


@pytest.mark.parametrize(
    "flags",
    [
        ("--cal_weight", "0", "--row_weight", "0"),
        ("--batch_local", "--row_set", "anchors"),
        ("--batch_local", "--cal_weight", "0"),
        ("--row_columns", "random", "--holdout_edge_frac", "0.2"),
        ("--row_reweight", "--row_set", "anchors"),
        ("--row_reweight", "--batch_sampler", "neighbor"),
        ("--row_target", "uniform", "--row_weight", "0"),
    ],
)
def test_contradictory_arms_are_refused(monkeypatch, flags):
    with pytest.raises(ValueError):
        _args(monkeypatch, *flags)


@pytest.mark.parametrize(
    "removed_flag",
    (
        "--r0_weight",
        "--r1_weight",
        "--support_policy",
        "--diffusion_quota",
        "--hard_neg_k",
        "--random_neg_k",
        "--relation_target",
        "--calibration_mode",
        "--row_centers",
        "--truncation_tolerance",
        "--graph_temp",
        "--num_walks",
    ),
)
def test_removed_flags_are_rejected(monkeypatch, removed_flag):
    monkeypatch.setattr(sys, "argv", ["main.py", "--method", "ggpkd", removed_flag, "1"])
    with pytest.raises(SystemExit):
        main.parse_args()


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #


def test_directed_rows_keep_every_retrieved_column_and_mutual_is_a_subset():
    graph = _graph()
    teacher = graph["teacher"]
    top_indices, top_scores = _compute_topk_cosine(teacher, k=8, device=torch.device("cpu"))
    temps = _knn_bandwidths(top_scores, 8)
    edges = {}
    for mode in ("directed", "mutual"):
        rows, _, _, _, _ = _build_transition(top_indices, top_scores, 8, temps, knn_mode=mode)
        edges[mode] = {(i, int(j)) for i, row in enumerate(rows) for j in row}
        if mode == "directed":
            assert all(len(row) == 8 for row in rows)
    assert edges["mutual"] < edges["directed"]


def test_holdout_removes_the_withheld_edges_from_every_row():
    graph = _graph(holdout=0.3)
    rows = np.repeat(np.arange(graph["neighbors"].shape[0]), graph["neighbors"].shape[1])
    cols = graph["neighbors"].reshape(-1)
    kept = cols >= 0
    withheld = heldout_edge_mask(rows[kept], cols[kept], seed=7, frac=0.3)
    # A row whose every edge was withheld keeps one; nothing else may leak.
    assert withheld.sum() <= (graph["neighbors"] >= 0).sum(axis=1).tolist().count(1)
    np.testing.assert_allclose(graph["probs"].sum(axis=1), 1.0, rtol=1e-5)


def test_artifact_reloads_only_when_its_settings_match(tmp_path, capsys):
    teacher = torch.randn(60, 8)
    path = str(tmp_path / "graph.pt")
    common = dict(cache_path=path, log_dir=str(tmp_path), graph_k=6)
    first = build_or_load_ggpkd_artifact(teacher, **common)
    capsys.readouterr()
    build_or_load_ggpkd_artifact(teacher, **common)
    assert "Loaded GGPKD artifact" in capsys.readouterr().out
    rebuilt = build_or_load_ggpkd_artifact(teacher, knn_mode="mutual", **common)
    assert "config mismatch" in capsys.readouterr().out
    assert rebuilt["metadata"]["knn_mode"] == "mutual"
    assert set(first) == {
        "transition_neighbors",
        "transition_probs",
        "row_temps",
        "graph_log_path",
        "graph_stats",
        "metadata",
    }


def test_fixed_bandwidth_is_one_median_temperature(tmp_path):
    teacher = torch.randn(60, 8)
    common = dict(log_dir=str(tmp_path), graph_k=6)
    knn = build_or_load_ggpkd_artifact(teacher, cache_path=str(tmp_path / "a.pt"), **common)
    fixed = build_or_load_ggpkd_artifact(
        teacher, cache_path=str(tmp_path / "b.pt"), fixed_bandwidth=True, **common
    )
    assert fixed["row_temps"].unique().numel() == 1
    assert float(fixed["row_temps"][0]) == pytest.approx(float(np.median(knn["row_temps"].numpy())), rel=1e-4)


def test_student_graph_takes_columns_from_the_student_and_values_from_the_teacher(tmp_path):
    torch.manual_seed(1)
    teacher, student = torch.randn(50, 8), torch.randn(50, 8)
    artifact = build_or_load_ggpkd_artifact(
        teacher,
        cache_path=str(tmp_path / "student.pt"),
        log_dir=str(tmp_path),
        graph_k=5,
        neighbor_source="student:test",
        neighbor_embeddings=lambda: student,
    )
    student_top, _ = _compute_topk_cosine(student, k=5, device=torch.device("cpu"))
    _, teacher_scores = _compute_topk_cosine(teacher, k=5, device=torch.device("cpu"))
    np.testing.assert_array_equal(artifact["transition_neighbors"].numpy(), student_top)
    np.testing.assert_allclose(
        artifact["row_temps"].numpy(), _knn_bandwidths(teacher_scores, 5), rtol=1e-5
    )
    normalized = F.normalize(teacher, dim=-1)
    row = torch.from_numpy(student_top[0])
    expected = torch.softmax((normalized[0] @ normalized[row].T) / artifact["row_temps"][0], -1)
    np.testing.assert_allclose(artifact["transition_probs"][0].numpy(), expected.numpy(), atol=1e-5)
    # A teacher graph at the same path never loads under the student's name.
    teacher_graph = build_or_load_ggpkd_artifact(
        teacher, cache_path=str(tmp_path / "student.pt"), log_dir=str(tmp_path), graph_k=5
    )
    assert teacher_graph["metadata"]["neighbor_source"] == "teacher"


def test_reciprocal_components_follow_the_reciprocal_edges():
    # Two blocks linked only by one-way edges, plus a node nobody retrieves back.
    neighbors = np.array(
        [[1, 2], [0, 2], [0, 1], [4, 0], [3, 0], [0, 1]], dtype=np.int64
    )
    labels, _ = reciprocal_component_labels(neighbors)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] != labels[0]
    stats = reciprocal_component_stats(neighbors)
    assert stats["reciprocal_components"] == 3
    assert stats["reciprocal_isolated"] == 1


def test_pool_inclusion_probability_is_exact():
    graph = _graph(n_items=60, k=5)
    neighbors, batch = graph["neighbors"], 6
    p = pool_inclusion_probability(neighbors, batch)
    rng = np.random.default_rng(0)
    trials = 20000
    hits = np.zeros(neighbors.shape[0])
    for _ in range(trials):
        anchors = rng.choice(neighbors.shape[0], batch, replace=False)
        rows = neighbors[anchors]
        hits[np.unique(np.concatenate([anchors, rows[rows >= 0]]))] += 1
    sigma = np.sqrt(p * (1 - p) / trials)
    assert np.all(np.abs(hits / trials - p) < 5 * sigma + 1e-3)
    assert np.all(p >= batch / neighbors.shape[0] - 1e-12)


def test_holdout_is_symmetric_reproducible_and_seeded():
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 5000, size=20000)
    cols = rng.integers(0, 5000, size=20000)
    forward = heldout_edge_mask(rows, cols, seed=99, frac=0.2)
    assert np.array_equal(forward, heldout_edge_mask(cols, rows, seed=99, frac=0.2))
    assert np.array_equal(forward, heldout_edge_mask(rows, cols, seed=99, frac=0.2))
    assert 0.18 < forward.mean() < 0.22
    other = heldout_edge_mask(rows, cols, seed=100, frac=0.2)
    assert 0.6 < float((forward == other).mean()) < 0.75
    assert not heldout_edge_mask(rows, cols, seed=99, frac=0.0).any()


# --------------------------------------------------------------------------- #
# Pool collate
# --------------------------------------------------------------------------- #


def test_pool_is_the_anchors_and_their_rows_encoded_once():
    graph = _graph()
    anchors = [3, 17, 40, 41]
    batch = _batch(graph, anchors, chunk=7)
    rows = graph["neighbors"][anchors]
    expected = np.unique(np.concatenate([anchors, rows[rows >= 0]]))
    np.testing.assert_array_equal(batch["pool_idx"].numpy(), expected)
    assert batch["pool_idx"][batch["anchor_pos"]].tolist() == anchors
    first_token = torch.cat([chunk["input_ids"][:, 0] for chunk in batch["pool_chunks"]])
    assert first_token.numel() == expected.size
    assert first_token.index_select(0, batch["pool_inverse"]).tolist() == (expected + 1).tolist()
    assert "cal_exclude" not in batch


def test_in_batch_pool_is_the_batch():
    collate = GGPKDCollate(_Tok(), 16, _texts(20), None)
    batch = collate([{"idx": i} for i in (5, 2, 9)])
    assert batch["pool_idx"].tolist() == [2, 5, 9]
    assert batch["anchor_pos"].tolist() == [1, 0, 2]
    with pytest.raises(ValueError):
        collate([{"idx": 1}])


def test_holdout_also_masks_the_calibration_pairs():
    graph = _graph(holdout=0.3)
    anchors = [1, 2, 3]
    batch = _batch(graph, anchors, holdout=0.3)
    rows = np.broadcast_to(np.asarray(anchors)[:, None], batch["cal_exclude"].shape)
    cols = np.broadcast_to(batch["pool_idx"].numpy()[None, :], batch["cal_exclude"].shape)
    np.testing.assert_array_equal(
        batch["cal_exclude"].numpy(), heldout_edge_mask(rows, cols, seed=7, frac=0.3)
    )


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


def _forward(criterion, embeddings, batch):
    return criterion(
        embeddings[batch["pool_idx"]],
        batch["pool_idx"],
        batch["anchor_pos"],
        batch.get("cal_exclude"),
    )


def _expected_row_loss(graph, embeddings, batch, row_set="all", weights=None):
    pool = batch["pool_idx"].tolist()
    position = {node: p for p, node in enumerate(pool)}
    anchors = set(batch["anchor_pos"].tolist())
    normalized = F.normalize(embeddings[batch["pool_idx"]].float(), dim=-1)
    kls, row_weights = [], []
    for p, node in enumerate(pool):
        if (row_set == "anchors" and p not in anchors) or (
            row_set == "non_anchors" and p in anchors
        ):
            continue
        cols, mass = [], []
        for u, w in zip(graph["neighbors"][node], graph["probs"][node]):
            if u >= 0 and w > 0 and int(u) in position and position[int(u)] != p:
                cols.append(position[int(u)])
                mass.append(float(w))
        if len(cols) < 2:
            continue
        target = torch.tensor(mass) / sum(mass)
        logits = normalized[p] @ normalized[cols].T / float(graph["temps"][node])
        kls.append((target * (target.log() - torch.log_softmax(logits, -1))).sum())
        row_weights.append(1.0 if weights is None else float(weights[node]))
    kls, row_weights = torch.stack(kls), torch.tensor(row_weights)
    return float((kls * row_weights).sum() / row_weights.sum()), len(kls)


def test_row_loss_is_the_mean_kl_over_every_encoded_row():
    graph = _graph()
    embeddings = torch.randn(80, 12)
    batch = _batch(graph, [0, 9, 33, 70])
    _, metrics = _forward(_criterion(graph, cal_weight=0.0), embeddings, batch)
    expected, count = _expected_row_loss(graph, embeddings, batch)
    assert metrics["loss_row"] == pytest.approx(expected, rel=1e-4)
    assert metrics["row_count"] == count


@pytest.mark.parametrize("row_set", ["anchors", "non_anchors"])
def test_row_sets_select_exactly_their_rows(row_set):
    graph = _graph()
    embeddings = torch.randn(80, 12)
    batch = _batch(graph, [0, 9, 33, 70])
    _, metrics = _forward(_criterion(graph, cal_weight=0.0, row_set=row_set), embeddings, batch)
    expected, count = _expected_row_loss(graph, embeddings, batch, row_set=row_set)
    assert metrics["loss_row"] == pytest.approx(expected, rel=1e-4)
    assert metrics["row_count"] == count
    if row_set == "anchors":
        # An anchor's whole row is in the pool by construction.
        assert metrics["row_exposed_mass"] == pytest.approx(1.0, abs=1e-5)


def test_inclusion_weights_give_a_weighted_row_mean():
    graph = _graph()
    embeddings = torch.randn(80, 12)
    batch = _batch(graph, [0, 9, 33, 70])
    p = pool_inclusion_probability(graph["neighbors"], 4)
    criterion = _criterion(graph, cal_weight=0.0, row_inclusion=torch.from_numpy(p))
    _, metrics = _forward(criterion, embeddings, batch)
    expected, _ = _expected_row_loss(graph, embeddings, batch, weights=1.0 / p)
    assert metrics["loss_row"] == pytest.approx(expected, rel=1e-4)
    assert 0 < metrics["row_ess_ratio"] <= 1


def test_the_teacher_itself_has_zero_loss():
    graph = _graph()
    batch = _batch(graph, [4, 8, 15, 16])
    for kwargs in ({}, {"row_columns": "random"}):
        loss, metrics = _forward(_criterion(graph, **kwargs), graph["teacher"], batch)
        assert metrics["loss_row"] == pytest.approx(0.0, abs=1e-5)
        assert metrics["loss_cal"] == pytest.approx(0.0, abs=1e-5)


def test_uniform_target_changes_the_values_not_the_columns():
    graph = _graph()
    embeddings = torch.randn(80, 12)
    batch = _batch(graph, [1, 2, 3, 4])
    _, teacher = _forward(_criterion(graph), embeddings, batch)
    _, uniform = _forward(_criterion(graph, row_target="uniform"), embeddings, batch)
    assert uniform["row_count"] == teacher["row_count"]
    assert uniform["row_eff_denom"] == teacher["row_eff_denom"]
    assert uniform["loss_row"] != pytest.approx(teacher["loss_row"])
    assert uniform["row_teacher_entropy"] > teacher["row_teacher_entropy"]


def test_random_columns_keep_each_rows_width():
    graph = _graph()
    embeddings = torch.randn(80, 12)
    batch = _batch(graph, [1, 2, 3, 4])
    _, graph_metrics = _forward(_criterion(graph), embeddings, batch)
    torch.manual_seed(0)
    _, random_metrics = _forward(_criterion(graph, row_columns="random"), embeddings, batch)
    assert random_metrics["row_count"] == graph_metrics["row_count"]
    assert random_metrics["row_eff_denom"] == pytest.approx(graph_metrics["row_eff_denom"])
    assert random_metrics["loss_row"] != pytest.approx(graph_metrics["loss_row"])


def test_calibration_exclusion_removes_the_withheld_pairs():
    graph = _graph(holdout=0.3)
    embeddings = torch.randn(80, 12)
    batch = _batch(graph, [1, 2, 3, 4], holdout=0.3)
    criterion = _criterion(graph, row_weight=0.0)
    _, masked = _forward(criterion, embeddings, batch)
    unmasked_batch = {key: value for key, value in batch.items() if key != "cal_exclude"}
    _, unmasked = _forward(criterion, embeddings, unmasked_batch)
    removed = batch["cal_exclude"].float().sum(-1).mean().item()
    assert removed > 0
    assert unmasked["cal_columns"] - masked["cal_columns"] == pytest.approx(removed, abs=1e-4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"row_set": "anchors"},
        {"row_set": "non_anchors"},
        {"row_target": "uniform"},
        {"row_columns": "random"},
        {"cal_weight": 0.0},
        {"row_weight": 0.0},
    ],
)
def test_every_arm_backpropagates(kwargs):
    graph = _graph()
    embeddings = torch.randn(80, 12, requires_grad=True)
    batch = _batch(graph, [0, 1, 2, 3])
    loss, metrics = _forward(_criterion(graph, **kwargs), embeddings, batch)
    loss.backward()
    assert torch.isfinite(loss) and loss.item() > 0
    assert torch.isfinite(embeddings.grad).all() and embeddings.grad.abs().sum() > 0
    assert metrics["loss_total"] == pytest.approx(
        metrics["loss_cal_weighted"] + metrics["loss_row_weighted"], rel=1e-5
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cal_weight": 0.0, "row_weight": 0.0},
        {"cal_weight": -1.0},
        {"row_set": "some"},
        {"row_target": "graded"},
        {"row_columns": "neighbors"},
        {"row_inclusion": torch.zeros(80)},
    ],
)
def test_criterion_refuses_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        _criterion(_graph(), **kwargs)


def test_a_non_finite_encoder_output_is_named():
    graph = _graph()
    embeddings = torch.randn(80, 12)
    embeddings[3] = float("nan")
    batch = _batch(graph, [3, 4, 5, 6])
    with pytest.raises(RuntimeError, match="pool_embeddings"):
        _forward(_criterion(graph), embeddings, batch)


# --------------------------------------------------------------------------- #
# Neighbour batches
# --------------------------------------------------------------------------- #


def _blocks(n_items=240, k=8, block=24):
    neighbors = np.zeros((n_items, k), dtype=np.int64)
    for i in range(n_items):
        start = (i // block) * block
        neighbors[i] = [j for j in range(start, start + block) if j != i][:k]
    return neighbors


def test_neighbor_sampler_partitions_the_corpus_and_is_reseeded():
    neighbors = _blocks()
    sampler = NeighborBatchSampler(neighbors, batch_size=8, seed=3, drop_last=False)
    first = [list(batch) for batch in sampler]
    assert sorted(i for batch in first for i in batch) == list(range(240))
    sampler.set_epoch(1)
    assert [list(batch) for batch in sampler] != first
    assert [list(b) for b in NeighborBatchSampler(neighbors, 8, seed=3, drop_last=False)] == first


def test_neighbor_batches_hold_more_neighbours_than_random_ones():
    neighbors = _blocks()
    neighbor = batch_relevance_stats(list(NeighborBatchSampler(neighbors, 8, seed=1)), neighbors)
    order = np.random.default_rng(1).permutation(240)
    random = batch_relevance_stats([order[i : i + 8] for i in range(0, 240, 8)], neighbors)
    assert neighbor["in_batch_neighbors_per_row"] > 3 * random["in_batch_neighbors_per_row"]


# --------------------------------------------------------------------------- #
# Train step
# --------------------------------------------------------------------------- #


class _TinyStudent(torch.nn.Module):
    def __init__(self, vocab, dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, dim)

    def forward(self, input_ids, attention_mask, return_dict=True, output_hidden_states=False):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_train_step_encodes_the_pool_once_and_updates_the_student(monkeypatch):
    from torch.amp import GradScaler

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    graph = _graph()
    distiller = KnowledgeDistiller.__new__(KnowledgeDistiller)
    distiller.config = SimpleNamespace(distill_method="ggpkd", w_task=0.0)
    distiller.method = get_method("ggpkd")
    distiller.device_s = torch.device("cpu")
    distiller.model_student = _TinyStudent(vocab=81, dim=12)
    distiller.criterion = _criterion(graph)
    distiller.optimizer = torch.optim.Adam(distiller.model_student.parameters(), lr=1e-2)
    distiller.scheduler = torch.optim.lr_scheduler.LambdaLR(distiller.optimizer, lambda _: 1.0)
    distiller.scaler = GradScaler("cuda", enabled=False)
    distiller.current_epoch = 0
    distiller.current_step = 0

    batch = _batch(graph, [0, 10, 20, 30], chunk=9)
    before = distiller.model_student.embedding.weight.detach().clone()
    loss, metrics = distiller.train_step(batch)

    assert torch.isfinite(loss)
    assert metrics["loss_row"] > 0 and metrics["loss_cal"] > 0 and metrics["row_count"] > 0
    assert distiller.encoded_texts_total == batch["pool_idx"].numel()
    assert not torch.equal(before, distiller.model_student.embedding.weight.detach())


# --------------------------------------------------------------------------- #
# Corpus alignment for neighbour batching
# --------------------------------------------------------------------------- #


def test_pointwise_trains_on_the_corpus_the_graph_is_built_over():
    import pandas as pd

    df = pd.DataFrame({"text": ["a", "b", "a", "c", "b", "d"], "source": list("xxyyzz")})
    ctx = SimpleNamespace(
        config=SimpleNamespace(task_type="single_cls", ggpkd_anchor_column=None)
    )
    pointwise_df, pointwise_keep = get_method("pointwise").prepare_frame(ctx, df)
    ggpkd_df, ggpkd_keep = get_method("ggpkd").prepare_frame(ctx, df)
    assert pointwise_df["text"].tolist() == ggpkd_df["text"].tolist() == ["a", "b", "c", "d"]
    np.testing.assert_array_equal(pointwise_keep, ggpkd_keep)


def test_neighbor_batches_refuse_a_graph_over_another_corpus():
    distiller = KnowledgeDistiller.__new__(KnowledgeDistiller)
    distiller.config = SimpleNamespace(
        batch_sampler="neighbor", batch_size=4, seed=0, distill_method="pointwise"
    )
    distiller.method = get_method("pointwise")
    distiller.ggpkd_artifact = {"transition_neighbors": torch.zeros(10, 3, dtype=torch.long)}
    distiller.train_ds = list(range(12))
    with pytest.raises(ValueError, match="10 nodes"):
        distiller.build_batch_sampler()
