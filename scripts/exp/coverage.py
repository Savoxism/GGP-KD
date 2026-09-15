#!/usr/bin/env python3
"""Measured in-batch neighbours per row, random vs neighbour batching (Table 3).

Table 3's "nbrs / row" column counts, for a batch-local arm, how many of a row
centre's graph neighbours N(i) sit in its own batch. The neighbour arms are read
against the random arms on that count, so it has to be the count the sampler
achieved rather than the one it was meant to achieve: a greedy partition runs out
of unused neighbours, and the batches it tops up at random pull the number down.

No training and no teacher pass. Rows come from the graph artifact's
`transition_neighbors`, and the batches from the same code the DataLoader uses:
`NeighborBatchSampler` for neighbour batching, a seeded permutation for random.
Both drop the ragged last batch, as a batch-relational run does.

Random batching has a closed form to check the measurement against. A given
neighbour lands in the anchor's batch with probability (B-1)/(N-1), so the
expected count is (B-1) * k / (N-1), with k the mean row width actually stored
(held-out edges and the mutual filter both make it smaller than graph_k).

Usage:
    python scripts/exp/coverage.py \\
        --artifact cache/exp/<pair>/<corpus>/graph_main.pt \\
        --batch-size 64 --epochs 5 --out runs/table3_exposure/coverage.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_utils.batch_samplers import NeighborBatchSampler, batch_relevance_stats


def load_teacher(cache_path: Path) -> torch.Tensor:
    """The cached teacher matrix, however the cache chose to wrap it."""
    blob = torch.load(cache_path, map_location="cpu", weights_only=False)
    if isinstance(blob, torch.Tensor):
        embeddings = blob
    elif isinstance(blob, dict):
        for key in ("teacher_cls", "embeddings", "teacher", "cls"):
            if key in blob and torch.is_tensor(blob[key]):
                embeddings = blob[key]
                break
        else:
            tensors = [v for v in blob.values() if torch.is_tensor(v) and v.dim() == 2]
            if len(tensors) != 1:
                raise ValueError(
                    f"cannot identify the teacher matrix in {cache_path}: "
                    f"keys={list(blob)}"
                )
            embeddings = tensors[0]
    else:
        raise ValueError(f"unsupported teacher cache object: {type(blob)}")
    return embeddings.float()


def random_batches(
    n_items: int, batch_size: int, epochs: int, seed: int
) -> list[np.ndarray]:
    batches = []
    for epoch in range(epochs):
        order = np.random.default_rng([seed, epoch]).permutation(n_items)
        for start in range(0, n_items - batch_size + 1, batch_size):
            batches.append(order[start : start + batch_size])
    return batches


def neighbor_batches(
    neighbors: np.ndarray, batch_size: int, epochs: int, seed: int
) -> list[list[int]]:
    sampler = NeighborBatchSampler(neighbors, batch_size, seed, drop_last=True)
    batches = []
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        batches.extend(list(sampler))
    return batches


def analytic_neighbors_per_row(
    neighbors: np.ndarray, batch_size: int
) -> float:
    n_items = int(neighbors.shape[0])
    mean_degree = float((neighbors >= 0).sum(axis=1).mean())
    return (batch_size - 1) * mean_degree / max(1, n_items - 1)


def measure(
    neighbors: np.ndarray, batch_size: int, epochs: int, seed: int
) -> list[dict[str, object]]:
    """One row per batching rule."""
    n_items = int(neighbors.shape[0])
    mean_degree = float((neighbors >= 0).sum(axis=1).mean())
    analytic = analytic_neighbors_per_row(neighbors, batch_size)
    rules = {
        "random": random_batches(n_items, batch_size, epochs, seed),
        "neighbor": neighbor_batches(neighbors, batch_size, epochs, seed),
    }
    rows = []
    for name, batches in rules.items():
        stats = batch_relevance_stats(batches, neighbors)
        rows.append(
            {
                "batching": name,
                "batch_size": batch_size,
                "epochs": epochs,
                "n_items": n_items,
                "mean_degree": round(mean_degree, 4),
                "in_batch_neighbors_per_row": round(
                    stats.get("in_batch_neighbors_per_row", float("nan")), 6
                ),
                "in_batch_precision": round(
                    stats.get("in_batch_precision", float("nan")), 6
                ),
                "n_batches": int(stats.get("n_batches", 0)),
                # The closed form holds for random batching only; beside the
                # neighbour row it is the baseline that row is read against.
                "analytic_random_neighbors_per_row": round(analytic, 6),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="GGPKD graph artifact (.pt)")
    parser.add_argument("--out", required=True, help="CSV to write")
    parser.add_argument("--pair", default="", help="label for the CSV's pair column")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="training seed; NeighborBatchSampler regroups from (seed, epoch), so "
        "this reproduces the batches that seed's neighbour run saw",
    )
    args = parser.parse_args()

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    neighbors = artifact["transition_neighbors"].numpy()
    metadata = artifact.get("metadata", {})
    rows = measure(neighbors, args.batch_size, args.epochs, args.seed)

    fieldnames = ["pair", "graph_k", "holdout_edge_frac", *rows[0].keys()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "pair": args.pair,
                    "graph_k": metadata.get("graph_k", ""),
                    "holdout_edge_frac": metadata.get("holdout_edge_frac", ""),
                    **row,
                }
            )
            print(
                f"  {row['batching']:<9} B={row['batch_size']} "
                f"nbrs/row={row['in_batch_neighbors_per_row']:.3f} "
                f"precision={row['in_batch_precision']:.4f} "
                f"(analytic random: {row['analytic_random_neighbors_per_row']:.3f})"
            )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
