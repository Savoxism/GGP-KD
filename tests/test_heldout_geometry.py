"""The held-out geometry probe behind Table 4's held-out columns.

Each test pins a way the probe could still produce a clean-looking number while
measuring the wrong thing:

* the metrics must read the mask they are given, or every arm is scored on its
  own training edges;
* the relation sets must exclude every pair L_row could have supervised -- the
  anchor's row and its in-edges -- or an arm appears to generalize by fitting;
* the component split must partition the held-out edges by the training graph's
  reciprocal components, or the C3 prediction is read on the wrong pairs.
"""

import numpy as np
import torch

from src.distill.geometry import knn_recall, pair_order_accuracy
from src.ggpkd.graph_builder import reciprocal_component_labels


def _cosines(embeddings):
    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    return normalized @ normalized.t()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_a_perfect_student_scores_one_on_both_metrics():
    teacher = torch.randn(60, 16)
    cosines = _cosines(teacher)
    off_diagonal = ~torch.eye(60, dtype=torch.bool)
    assert knn_recall(cosines, cosines, k=5, restrict=off_diagonal) == 1.0
    assert (
        pair_order_accuracy(cosines, cosines, restrict=off_diagonal, n_triplets=5000)
        == 1.0
    )


def test_an_unrelated_student_scores_near_chance_on_ordering():
    torch.manual_seed(0)
    teacher = _cosines(torch.randn(80, 16))
    student = _cosines(torch.randn(80, 16))
    off_diagonal = ~torch.eye(80, dtype=torch.bool)
    accuracy = pair_order_accuracy(
        teacher, student, restrict=off_diagonal, n_triplets=20000
    )
    assert 0.4 < accuracy < 0.6


def test_the_restrict_mask_actually_removes_pairs():
    """A mask that were ignored would score every arm on its own training edges."""
    torch.manual_seed(1)
    teacher = _cosines(torch.randn(50, 16))
    # Corrupt the student everywhere *except* one held-out block, then score only
    # that block: a metric that reads the mask sees a perfect student.
    held_out = torch.zeros(50, 50, dtype=torch.bool)
    held_out[:10, 10:20] = True
    noise = torch.randn(50, 50) * 5.0
    student = torch.where(held_out, teacher, teacher + noise)
    assert (
        pair_order_accuracy(teacher, student, restrict=held_out, n_triplets=20000)
        == 1.0
    )


# --------------------------------------------------------------------------- #
# Relation sets
# --------------------------------------------------------------------------- #


def _random_rows(n_items, width, seed):
    rng = np.random.default_rng(seed)
    rows = np.full((n_items, width), -1, dtype=np.int64)
    for i in range(n_items):
        degree = int(rng.integers(width // 2, width + 1))
        rows[i, :degree] = rng.choice(
            [j for j in range(n_items) if j != i], size=degree, replace=False
        )
    return rows


def test_relation_sets_exclude_everything_any_arm_was_trained_on():
    """The masks are the experiment's claim; a leak here fakes the whole result.

    L_row supervises every encoded row, so the pair (anchor, u) is a training
    target when u is in the anchor's row *and* when the anchor is in u's row.
    """
    from scripts.exp.heldout_geometry import build_masks

    n_items, width, graph_k = 400, 10, 12
    transition = _random_rows(n_items, width, seed=3)
    artifact = {
        "transition_neighbors": torch.from_numpy(transition),
        "metadata": {"graph_k": graph_k},
    }
    teacher = torch.nn.functional.normalize(torch.randn(n_items, 24), p=2, dim=-1)
    anchors = np.arange(0, n_items, 7)

    masks = build_masks(
        artifact,
        anchors,
        teacher,
        graph_k=graph_k,
        nonlocal_mult=5,
        holdout_frac=0.2,
        holdout_seed=12345,
        device="cpu",
    )
    assert set(masks) == {
        "unsupervised",
        "nonlocal",
        "heldout_edges",
        "heldout_same_component",
        "heldout_cross_component",
    }
    for position, anchor in enumerate(anchors):
        anchor = int(anchor)
        out_edges = {int(j) for j in transition[anchor] if j >= 0}
        in_edges = {int(u) for u in np.flatnonzero((transition == anchor).any(axis=1))}
        trained = out_edges | in_edges | {anchor}
        assert in_edges, "the fixture must exercise in-edges"
        for name in masks:
            scored = set(torch.nonzero(masks[name][position]).flatten().tolist())
            assert not (scored & trained), f"{name} leaked a supervised pair"
    # The holdout set has to be non-empty, or Table 4's held-out columns silently
    # report on nothing.
    assert int(masks["heldout_edges"].sum()) > 0


def test_no_holdout_artifact_yields_no_holdout_set():
    from scripts.exp.heldout_geometry import build_masks

    n_items = 200
    rows = np.array(
        [[(i + offset) % n_items for offset in (1, 2, 3, 4, 5)] for i in range(n_items)],
        dtype=np.int64,
    )
    artifact = {
        "transition_neighbors": torch.from_numpy(rows),
        "metadata": {"graph_k": 5},
    }
    teacher = torch.nn.functional.normalize(torch.randn(n_items, 8), p=2, dim=-1)
    masks = build_masks(
        artifact,
        np.arange(0, n_items, 11),
        teacher,
        graph_k=5,
        nonlocal_mult=4,
        holdout_frac=0.0,
        holdout_seed=0,
        device="cpu",
    )
    assert "heldout_edges" not in masks
    assert "heldout_same_component" not in masks
    assert "heldout_cross_component" not in masks


def test_the_component_split_partitions_heldout_edges_by_training_components():
    """Two reciprocal rings make two components; the teacher ignores them.

    The teacher's top-k crosses the halves freely, so held-out edges land both
    inside a component and across the two, and each must go to the right set.
    """
    from scripts.exp.heldout_geometry import build_masks

    n_items, half, graph_k = 320, 160, 12
    offsets = (-3, -2, -1, 1, 2, 3)
    rows = np.empty((n_items, len(offsets)), dtype=np.int64)
    for i in range(n_items):
        base = 0 if i < half else half
        rows[i] = [base + (i - base + o) % half for o in offsets]
    labels, reciprocity = reciprocal_component_labels(rows)
    assert reciprocity == 1.0
    assert len(set(labels.tolist())) == 2

    torch.manual_seed(4)
    teacher = torch.nn.functional.normalize(torch.randn(n_items, 6), p=2, dim=-1)
    anchors = np.arange(0, n_items, 3)
    masks = build_masks(
        {"transition_neighbors": torch.from_numpy(rows), "metadata": {}},
        anchors,
        teacher,
        graph_k=graph_k,
        nonlocal_mult=3,
        holdout_frac=0.3,
        holdout_seed=7,
        device="cpu",
    )
    heldout = masks["heldout_edges"]
    same = masks["heldout_same_component"]
    cross = masks["heldout_cross_component"]
    assert not bool((same & cross).any())
    assert torch.equal(same | cross, heldout)
    assert int(same.sum()) > 0 and int(cross.sum()) > 0

    anchor_labels = torch.from_numpy(labels[anchors])[:, None]
    column_labels = torch.from_numpy(labels)[None, :]
    assert bool((anchor_labels == column_labels)[same].all())
    assert not bool((anchor_labels == column_labels)[cross].any())
