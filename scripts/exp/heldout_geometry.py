#!/usr/bin/env python3
"""Held-out teacher geometry, measured post-hoc on saved student weights.

The held-out columns of Table 4 (story.md §5, C3). A downstream score says a
student is better; a training loss says it fitted what it was shown. Neither says
whether the student recovered teacher relations it was *never* supervised on, and
§3's argument makes a sharper prediction than "it generalizes": L_row pins the
teacher's geometry up to one additive offset per connected component of the
reciprocal kNN graph, and L_cal is what ties the components together. So
held-out pairs inside one component should be recovered with or without L_cal,
and pairs across components only with it.

The relation sets below are computed from the cached teacher and the graph
artifact, so no teacher forward pass is needed and this runs on any checkpoint
after the fact. None contains a pair any arm was trained on as a row entry:

    heldout_edges  the edges withheld from the graph by `--holdout_edge_frac`.
                   Reconstructed with the same symmetric hash the build used, so
                   the split cannot drift between training and evaluation. Only
                   present for a run trained with a holdout.
    heldout_same_component, heldout_cross_component
                   heldout_edges split by whether both endpoints lie in the same
                   component of the reciprocal graph built on the artifact's
                   (training) rows. The split is the Table 4 prediction.
    nonlocal       teacher ranks graph_k+1 .. `--nonlocal-mult` * graph_k. These
                   are teacher-relevant but outside every arm's retrieval width,
                   so this set exists even for a family trained without a holdout
                   -- it is the "did the student learn structure beyond its own
                   neighbourhood" question.
    unsupervised   every pair except the anchor's own graph row and in-edges.
                   The broadest set, and the loosest: most of it is pairs the teacher
                   considers unrelated, so it moves with anisotropy. Reported for
                   completeness, not as the headline.

Three metrics per set, and no more. Each answers something a reader can name:

    spearman       rank correlation of teacher and student cosines over the set,
                   pooled across anchors -- so it also reads the global cosine
                   scale. `spearman_anchor` is the mean per-anchor correlation,
                   which reads only each anchor's own ordering.
    knn_recall     of the teacher's top-k inside the set, how many the student
                   also ranks top-k *across every unsupervised column*. The wide
                   ranking pool is the point: scored only among themselves, five
                   held-out positives are trivially recovered and every arm reads
                   ~0.87. For `heldout_edges` this is the "did it recover a hidden
                   neighbour" number.
    pair_order     P[ the student orders two candidates as the teacher does ].
                   The one metric a collapsed space cannot win.

Everything is read off the *whole* corpus: the student encodes all N texts once
(cheap -- 13.5k short texts) and the teacher matrix comes from the cache, so each
sampled anchor's row is exact rather than restricted to a probe subset.

Usage:
    python scripts/exp/heldout_geometry.py \
        --runs results/table4_loss_terms/<run_id>/runs \
        --cache cache/exp/<pair>/<corpus>/teacher_train.pt \
        --artifact cache/exp/<pair>/<corpus>/graph_main_holdout.pt \
        --train-data data/train_set/merged_3_data_5k_each.csv \
        --out runs/table4_heldout/heldout_geometry.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exp.coverage import load_teacher
from src.distill.geometry import _spearman, pair_order_accuracy
from src.ggpkd.graph_builder import heldout_edge_mask, reciprocal_component_labels

RELATION_SETS = (
    "heldout_edges",
    "heldout_same_component",
    "heldout_cross_component",
    "nonlocal",
    "unsupervised",
)


def resolve_anchor_column(frame: pd.DataFrame) -> str:
    for column in ("text", "premise", "sentence1"):
        if column in frame.columns:
            return column
    raise ValueError(f"no anchor column among {list(frame.columns)}")


def corpus_texts(train_data: Path) -> list[str]:
    """The deduplicated anchor corpus, in the order the graph was built on.

    Must match `src.methods.ggpkd.prepare_frame` exactly: the teacher cache and the
    artifact are indexed by these row positions, so a different dedup rule here
    would silently pair every anchor with another anchor's teacher vector.
    """
    frame = pd.read_csv(train_data)
    column = resolve_anchor_column(frame)
    texts = frame[column].astype(str)
    keep = ~texts.duplicated(keep="first").to_numpy()
    return [str(t) for t in texts[keep].tolist()]


@torch.no_grad()
def encode_corpus(
    weights_path: Path,
    texts: list[str],
    student_model_name: str | None,
    max_length: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """CLS-pooled student embeddings for the whole corpus.

    Same pooling as the benchmark path and the training-time probe, so the
    geometry described here is the geometry the reported scores were read from.
    """
    from transformers import AutoModel, AutoTokenizer

    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    name = student_model_name or payload.get("student_model_name")
    if not name:
        raise ValueError(
            f"{weights_path} carries no student_model_name; pass --student-model"
        )
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    missing, unexpected = model.load_state_dict(
        payload["model_state_dict"], strict=False
    )
    if missing:
        # Loading a checkpoint that does not fit the architecture would otherwise
        # be reported as a mediocre student rather than as a loading bug.
        raise ValueError(
            f"{weights_path} is missing parameters for {name}: {missing[:5]}"
        )
    if unexpected:
        print(
            f"  note: ignoring {len(unexpected)} unexpected keys (e.g. {unexpected[:2]})"
        )
    model.eval().to(device)

    outputs = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        result = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        )
        outputs.append(result.last_hidden_state[:, 0, :].float().cpu())
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return torch.cat(outputs, dim=0)


def build_masks(
    artifact: dict,
    anchors: np.ndarray,
    teacher_norm: torch.Tensor,
    graph_k: int,
    nonlocal_mult: int,
    holdout_frac: float,
    holdout_seed: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """One boolean [P, N] mask per relation set. True means "score this pair".

    The supervised set every mask excludes is the anchor's own graph row *and* its
    in-edges -- the texts u with anchor in N(u). L_row supervises every encoded
    row, so a pair (anchor, u) is a training target from u's side whenever the
    anchor is in u's row, even though u is not in the anchor's.
    """
    n_items = int(teacher_norm.size(0))
    transition = artifact["transition_neighbors"].numpy()
    labels, _ = reciprocal_component_labels(transition)

    supervised = torch.zeros(len(anchors), n_items, dtype=torch.bool)
    position_of = np.full(n_items, -1, dtype=np.int64)
    position_of[np.asarray(anchors, dtype=np.int64)] = np.arange(len(anchors))
    for position, anchor in enumerate(anchors):
        anchor = int(anchor)
        columns = transition[anchor]
        columns = columns[columns >= 0]
        supervised[position, torch.from_numpy(columns).long()] = True
        supervised[position, anchor] = True
    sources, slots = np.nonzero(transition >= 0)
    targets = transition[sources, slots].astype(np.int64)
    hit = position_of[targets] >= 0
    supervised[
        torch.from_numpy(position_of[targets[hit]]),
        torch.from_numpy(sources[hit].astype(np.int64)),
    ] = True

    masks: dict[str, torch.Tensor] = {}
    masks["unsupervised"] = ~supervised

    # Teacher ranks, needed for both the nonlocal band and the holdout set.
    anchor_rows = teacher_norm[torch.from_numpy(anchors).long()].to(device)
    scores = anchor_rows @ teacher_norm.to(device).t()
    scores[
        torch.arange(len(anchors), device=scores.device),
        torch.from_numpy(anchors).to(scores.device),
    ] = float("-inf")
    rank_width = min(int(nonlocal_mult) * int(graph_k), n_items - 1)
    ranked = scores.topk(rank_width, dim=-1).indices.cpu()

    nonlocal_mask = torch.zeros(len(anchors), n_items, dtype=torch.bool)
    if rank_width > graph_k:
        band = ranked[:, graph_k:rank_width]
        nonlocal_mask.scatter_(1, band, True)
    # A pair in the band that is nonetheless a graph edge is dropped: the band is
    # defined by the anchor's rank, and an in-edge can sit at any rank.
    masks["nonlocal"] = nonlocal_mask & ~supervised

    if holdout_frac > 0:
        rows = np.repeat(anchors[:, None], ranked.shape[1], axis=1)
        withheld = heldout_edge_mask(rows, ranked.numpy(), holdout_seed, holdout_frac)
        # Restricted to within graph_k: an edge past the retrieval width was never
        # a candidate for the graph, so the holdout did not withhold it -- it is
        # simply nonlocal, and counting it here would inflate the set with
        # relations the mask had no say over.
        within = np.zeros_like(withheld)
        within[:, : min(graph_k, within.shape[1])] = True
        held = torch.from_numpy(withheld & within)
        holdout_mask = torch.zeros(len(anchors), n_items, dtype=torch.bool)
        holdout_mask.scatter_(1, ranked, held)
        # A row whose every edge was withheld keeps its nearest neighbour
        # (holdout_starved_rows), so that one pair was trained on after all.
        holdout_mask &= ~supervised
        masks["heldout_edges"] = holdout_mask
        # Components of the *training* graph: the artifact's rows already lack
        # the withheld edges, which is the graph L_row actually saw.
        label_tensor = torch.from_numpy(labels)
        same = label_tensor[torch.from_numpy(np.asarray(anchors, dtype=np.int64))][
            :, None
        ] == label_tensor[None, :]
        masks["heldout_same_component"] = holdout_mask & same
        masks["heldout_cross_component"] = holdout_mask & ~same
    return masks


def score(
    teacher_rows: torch.Tensor,
    student_rows: torch.Tensor,
    mask: torch.Tensor,
    pool: torch.Tensor,
    knn_k: int,
    seed: int,
) -> dict[str, float]:
    """The three metrics on one relation set.

    `mask` is the relation set -- the pairs being scored. `pool` is the set the
    student's ranking competes over, and it must be much wider than `mask` for
    the recall to mean anything (see `_positive_recall`).
    """
    admissible = int(mask.sum())
    if admissible < 100:
        return {"n_pairs": float(admissible)}
    flat_teacher = teacher_rows[mask]
    flat_student = student_rows[mask]
    return {
        "n_pairs": float(admissible),
        "pairs_per_anchor": float(mask.sum(dim=-1).double().mean()),
        "pool_per_anchor": float(pool.sum(dim=-1).double().mean()),
        # Agreement on the set itself: both of these are about whether the student
        # orders these particular pairs the way the teacher does, so they read the
        # set directly.
        "spearman": _spearman(flat_teacher, flat_student),
        # The pooled number above mixes every anchor's pairs, so an arm that
        # sharpens each neighbourhood but compresses the global scale reads worse
        # than a flat one. This one asks only about each anchor's own ordering.
        "spearman_anchor": _per_anchor_spearman(teacher_rows, student_rows, mask),
        "pair_order": pair_order_accuracy(
            teacher_rows, student_rows, restrict=mask, n_triplets=200000, seed=seed
        ),
        # Retrieval against a wide pool: did the student *find* the hidden
        # neighbours among everything it could have ranked instead.
        "knn_recall": _positive_recall(teacher_rows, student_rows, mask, pool, knn_k),
    }


def _per_anchor_spearman(
    teacher_rows: torch.Tensor,
    student_rows: torch.Tensor,
    mask: torch.Tensor,
    min_pairs: int = 3,
) -> float:
    """Mean over anchors of the Spearman between that anchor's own pairs."""
    values = []
    for row in range(mask.size(0)):
        columns = mask[row]
        if int(columns.sum()) < min_pairs:
            continue
        values.append(_spearman(teacher_rows[row][columns], student_rows[row][columns]))
    return float(np.mean(values)) if values else float("nan")


def _positive_recall(
    teacher_rows: torch.Tensor,
    student_rows: torch.Tensor,
    positives: torch.Tensor,
    pool: torch.Tensor,
    k: int,
) -> float:
    """Of the teacher's top-k inside `positives`, how many the student ranks
    top-k across the whole `pool`.

    The pool is what makes this a measurement rather than an arithmetic identity.
    Ranking the held-out positives only among themselves scores ~0.87 on a corpus
    where each anchor has about five of them -- the student is choosing five out
    of five, and every arm gets the same near-perfect number. Competing them
    against every unsupervised column instead asks the question the experiment
    means to ask: among all the relations it could have ranked highly, did the
    student put the hidden teacher neighbours near the top.

    Anchors with fewer than k positives are excluded rather than scored against a
    denominator they cannot reach.
    """
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    teacher_masked = teacher_rows.masked_fill(~positives, float("-inf"))
    student_masked = student_rows.masked_fill(~pool, float("-inf"))
    usable = positives.sum(dim=-1) >= k
    if not bool(usable.any()):
        return float("nan")
    teacher_top = teacher_masked.topk(k, dim=-1).indices
    student_top = student_masked.topk(k, dim=-1).indices
    overlaps = [
        float(torch.isin(teacher_top[row], student_top[row]).sum()) / k
        for row in torch.nonzero(usable, as_tuple=False).flatten().tolist()
    ]
    return float(np.mean(overlaps))


def find_runs(runs_root: Path) -> list[tuple[str, str, Path]]:
    """(arm, seed, weights file) for every finished run under `runs_root`.

    Layout is the one both runners write: <runs>/<arm>/seed_<n>/weights/*.pt.
    """
    found = []
    for weights in sorted(runs_root.glob("*/seed_*/weights/student_epoch_*.pt")):
        seed_dir = weights.parent.parent
        found.append(
            (seed_dir.parent.name, seed_dir.name.replace("seed_", ""), weights)
        )
    # Keep only the last epoch per run: the study compares final students.
    latest: dict[tuple[str, str], Path] = {}
    for arm, seed, weights in found:
        epoch = int(weights.stem.split("_")[-1])
        key = (arm, seed)
        if key not in latest or epoch > int(latest[key].stem.split("_")[-1]):
            latest[key] = weights
    return [(arm, seed, path) for (arm, seed), path in sorted(latest.items())]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="<run_root>/runs directory")
    parser.add_argument("--cache", required=True, help="teacher embedding cache (.pt)")
    parser.add_argument("--artifact", required=True, help="graph artifact (.pt)")
    parser.add_argument(
        "--train-data", default="data/train_set/merged_3_data_5k_each.csv"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--pair", default="")
    parser.add_argument("--student-model", default=None)
    parser.add_argument("--anchors", type=int, default=512)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--nonlocal-mult", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    texts = corpus_texts(Path(args.train_data))
    teacher = load_teacher(Path(args.cache))
    if teacher.size(0) != len(texts):
        raise ValueError(
            f"teacher cache has {teacher.size(0)} rows but the deduplicated corpus "
            f"has {len(texts)}; the cache was built on a different corpus"
        )
    teacher_norm = F.normalize(teacher, p=2, dim=-1)

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    metadata = artifact.get("metadata", {})
    graph_k = int(metadata.get("graph_k", 100))
    holdout_frac = float(metadata.get("holdout_edge_frac", 0.0))
    holdout_seed = int(metadata.get("holdout_seed", 0))
    print(
        f"artifact: graph_k={graph_k} holdout_frac={holdout_frac} "
        f"holdout_seed={holdout_seed}"
    )
    if holdout_frac == 0.0:
        print(
            "  NOTE: this artifact withheld no edges, so the `heldout_edges` set "
            "is empty and only `nonlocal` / `unsupervised` are reported"
        )
    labels, _ = reciprocal_component_labels(artifact["transition_neighbors"].numpy())
    n_components = int(labels.max()) + 1 if labels.size else 0
    print(f"  training graph: {n_components} reciprocal components")

    rng = np.random.default_rng(args.seed)
    anchors = rng.choice(len(texts), size=min(args.anchors, len(texts)), replace=False)
    anchors.sort()
    masks = build_masks(
        artifact,
        anchors,
        teacher_norm,
        graph_k,
        args.nonlocal_mult,
        holdout_frac,
        holdout_seed,
        args.device,
    )
    for name, mask in masks.items():
        per_anchor = float(mask.sum(dim=-1).double().mean())
        print(f"  {name}: {per_anchor:.1f} pairs/anchor")
        if name != "unsupervised" and per_anchor < 2 * args.knn_k:
            # With barely more positives than k, most anchors are dropped from the
            # recall (they cannot reach the denominator) and the ones that survive
            # are choosing almost every positive. Both make the column stop
            # separating the arms, which is the only thing it is there to do.
            print(
                f"    WARNING: only {per_anchor:.1f} pairs per anchor against "
                f"--knn-k {args.knn_k}. Lower k, or raise --holdout-frac / "
                "--nonlocal-mult, or read this set on spearman and pair_order "
                "instead of on recall"
            )

    # Every metric's ranking pool: the widest set of columns no arm was trained
    # on. `unsupervised` is a superset of every other set by construction, so
    # each of them is scored as a retrieval problem against the same wide field.
    ranking_pool = masks["unsupervised"]

    anchor_index = torch.from_numpy(anchors).long()
    teacher_rows = teacher_norm[anchor_index] @ teacher_norm.t()

    runs = find_runs(Path(args.runs))
    if not runs:
        raise ValueError(f"no finished runs with saved weights under {args.runs}")
    print(f"{len(runs)} runs to score")

    fieldnames = [
        "pair",
        "arm",
        "seed",
        "relation_set",
        "n_pairs",
        "pairs_per_anchor",
        "pool_per_anchor",
        "spearman",
        "spearman_anchor",
        "knn_recall",
        "pair_order",
        "knn_k",
        "n_anchors",
        "graph_k",
        "holdout_frac",
        "reciprocal_components",
        "weights",
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for arm, seed, weights in runs:
            print(f"scoring {arm} seed {seed}")
            student = encode_corpus(
                weights,
                texts,
                args.student_model,
                args.max_length,
                args.batch_size,
                args.device,
            )
            student_norm = F.normalize(student, p=2, dim=-1)
            student_rows = student_norm[anchor_index] @ student_norm.t()
            for name in RELATION_SETS:
                if name not in masks:
                    continue
                metrics = score(
                    teacher_rows,
                    student_rows,
                    masks[name],
                    ranking_pool,
                    args.knn_k,
                    args.seed,
                )
                writer.writerow(
                    {
                        "pair": args.pair,
                        "arm": arm,
                        "seed": seed,
                        "relation_set": name,
                        "knn_k": args.knn_k,
                        "n_anchors": len(anchors),
                        "graph_k": graph_k,
                        "holdout_frac": holdout_frac,
                        "reciprocal_components": n_components,
                        "weights": str(weights),
                        **{
                            key: (
                                round(value, 6) if isinstance(value, float) else value
                            )
                            for key, value in metrics.items()
                        },
                    }
                )
                handle.flush()
                print(
                    f"    {name:<24} spearman={metrics.get('spearman', float('nan')):.4f} "
                    f"recall@{args.knn_k}={metrics.get('knn_recall', float('nan')):.4f} "
                    f"order={metrics.get('pair_order', float('nan')):.4f}"
                )
    print(f"wrote {out_path}")
    print(
        json.dumps(
            {
                "runs": len(runs),
                "relation_sets": [n for n in RELATION_SETS if n in masks],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
