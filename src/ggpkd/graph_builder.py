"""The teacher-valued kNN graph GGPKD distils from.

Built once per (teacher, neighbour source, corpus, graph settings) and cached. For
every text j it stores the neighbour list N(j), the teacher transition row

    P^T_j(u) = softmax_{u in N(j)}( s^T_ju / tau_j ),

and tau_j. Nothing else: a step's pool, its supervised rows and their weights are
all derived from these arrays at training time.
"""

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import gammaln
from tqdm import tqdm

# 12: candidate pools, hard negatives, source ids and the symmetrized mode removed;
#     `fixed_bandwidth` now means one median bandwidth for every row; the build
#     reports reciprocal-graph connectivity.
ARTIFACT_VERSION = 12

KNN_MODES = ("directed", "mutual")
HOLDOUT_BANDWIDTHS = ("full", "surviving")
NEIGHBOR_SOURCES = ("teacher", "student")

# A row whose k neighbours all tie in cosine is uniform at every temperature; the
# floor only keeps the division finite.
MIN_BANDWIDTH = 1e-6


def _knn_bandwidths(top_scores: np.ndarray, graph_k: int) -> np.ndarray:
    """tau_i = (s_i(1) - s_i(k)) / log k, read off the raw top-k.

    The k-th neighbour sits log k nats below the nearest, i.e. is k times less
    likely. Under s -> a*s + b the bandwidth scales by a and the softmax is
    shift-invariant, so every row is invariant to the teacher's cosine scale.
    """
    k_eff = min(int(graph_k), top_scores.shape[1])
    if k_eff < 2:
        raise ValueError(f"graph_k must retrieve at least 2 neighbours, got {k_eff}")
    span = top_scores[:, 0].astype(np.float64) - top_scores[:, k_eff - 1].astype(
        np.float64
    )
    return np.maximum(span / np.log(k_eff), MIN_BANDWIDTH)


def heldout_edge_mask(
    rows: np.ndarray, cols: np.ndarray, seed: int, frac: float
) -> np.ndarray:
    """True where the pair (row, col) is withheld from training.

    A pure, symmetric function of the unordered pair and `seed` (splitmix64's
    finalizer), so the split is identical across arms and seeds, (i, j) and (j, i)
    are withheld together, and the evaluation recomputes it without reading the
    artifact.
    """
    if not 0.0 <= frac < 1.0:
        raise ValueError(f"holdout fraction must be in [0, 1), got {frac}")
    if frac == 0.0:
        return np.zeros(np.shape(rows), dtype=bool)
    low = np.minimum(rows, cols).astype(np.uint64)
    high = np.maximum(rows, cols).astype(np.uint64)
    key = low * np.uint64(0x9E3779B97F4A7C15) + high
    # Folded in Python so the wraparound is explicit and numpy raises no overflow
    # warning on the correct path.
    seed_term = np.uint64((int(seed) * 0xBF58476D1CE4E5B9) % (1 << 64))
    key = key ^ seed_term
    key = (key ^ (key >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    key = (key ^ (key >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    key = key ^ (key >> np.uint64(31))
    uniform = (key >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)
    return uniform < float(frac)


def pool_inclusion_probability(neighbors: np.ndarray, batch_size: int) -> np.ndarray:
    """P(text j is encoded in a step's pool), exactly.

    Anchors are drawn uniformly without replacement, and j enters the pool iff some
    anchor lies in S_j = {j} U {i : j in N(i)}. Hence

        P(j not in pool) = C(N - |S_j|, B) / C(N, B),   |S_j| = indegree(j) + 1.

    Without correction the row loss is weighted by this probability; weighting rows
    by its inverse is GraphSAINT's normalization (story.md, Table 6).
    """
    n_items = int(neighbors.shape[0])
    batch = int(batch_size)
    if not 1 <= batch <= n_items:
        raise ValueError(f"batch_size must be in [1, {n_items}], got {batch}")
    valid = neighbors >= 0
    indegree = np.bincount(neighbors[valid].astype(np.int64), minlength=n_items)
    outside = (n_items - (indegree + 1)).astype(np.float64)
    log_miss = np.full(n_items, -np.inf)
    reachable = outside >= batch
    o = outside[reachable]
    log_miss[reachable] = (
        gammaln(o + 1)
        + gammaln(n_items - batch + 1)
        - gammaln(o - batch + 1)
        - gammaln(n_items + 1)
    )
    return 1.0 - np.exp(log_miss)


def reciprocal_component_labels(neighbors: np.ndarray) -> tuple[np.ndarray, float]:
    """Component label of every text in the reciprocal kNN graph, and its reciprocity.

    The reciprocal graph keeps u-j iff u in N(j) and j in N(u). Cosine is symmetric,
    so a reciprocal edge ties the offsets of its two rows, and L_row pins the
    teacher's geometry up to one additive offset per component (story.md §3).
    """
    n_items, width = neighbors.shape
    rows = np.repeat(np.arange(n_items, dtype=np.int64), width)
    cols = neighbors.reshape(-1).astype(np.int64)
    keep = cols >= 0
    adjacency = csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.float32), (rows[keep], cols[keep])),
        shape=(n_items, n_items),
    )
    reciprocal = adjacency.multiply(adjacency.T).tocsr()
    _, labels = connected_components(reciprocal, directed=False)
    return labels.astype(np.int64), float(reciprocal.nnz / max(1, adjacency.nnz))


def reciprocal_component_stats(neighbors: np.ndarray) -> dict[str, float]:
    """Component count, isolated texts and giant-component share (Table 5)."""
    labels, reciprocity = reciprocal_component_labels(neighbors)
    sizes = np.bincount(labels)
    return {
        "reciprocity": reciprocity,
        "reciprocal_components": int(sizes.size),
        "reciprocal_isolated": int((sizes == 1).sum()),
        "reciprocal_largest_frac": float(sizes.max() / max(1, labels.size)),
    }


def _fingerprint(embeddings: torch.Tensor) -> str:
    """Content hash, so a changed teacher, pooling or corpus never reuses a graph."""
    array = embeddings.detach().to(torch.float32).cpu().numpy()
    digest = hashlib.sha1(np.ascontiguousarray(array).tobytes())
    digest.update(str(array.shape).encode("utf-8"))
    return digest.hexdigest()


def _default_device(embeddings: torch.Tensor) -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else embeddings.device


def _without_tf32(
    run: Callable[[torch.device], object], device: torch.device, what: str
):
    """Run on `device` with TF32 off, falling back to CPU on OOM.

    TF32 would drop the matmul to ~10 mantissa bits, and these cosines decide graph
    membership and bandwidths.
    """
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return run(device)
    except torch.cuda.OutOfMemoryError:
        print(f"GGPKD {what}: out of memory on {device}, falling back to CPU")
        torch.cuda.empty_cache()
        return run(torch.device("cpu"))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _compute_topk_cosine(
    embeddings: torch.Tensor,
    k: int,
    chunk_size: int = 1024,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k cosine neighbours of every row, self excluded."""
    device = device or _default_device(embeddings)

    def _run(target: torch.device) -> tuple[np.ndarray, np.ndarray]:
        normalized = F.normalize(embeddings.float(), p=2, dim=-1).to(target)
        n_items = normalized.size(0)
        k_eff = min(k + 1, n_items)
        all_indices, all_scores = [], []
        for start in tqdm(range(0, n_items, chunk_size), desc="GGPKD top-k cosine"):
            end = min(start + chunk_size, n_items)
            sims = normalized[start:end] @ normalized.T
            sims[
                torch.arange(end - start, device=target),
                torch.arange(start, end, device=target),
            ] = -float("inf")
            scores, indices = torch.topk(sims, k=k_eff, dim=-1)
            all_indices.append(indices[:, :k].cpu().numpy().astype(np.int64))
            all_scores.append(scores[:, :k].cpu().numpy().astype(np.float32))
        return np.concatenate(all_indices), np.concatenate(all_scores)

    return _without_tf32(_run, device, "top-k cosine")


def _cosine_scores_at_indices(
    embeddings: torch.Tensor,
    indices: np.ndarray,
    chunk_size: int = 128,
    candidate_chunk_size: int = 32,
    device: torch.device | None = None,
) -> np.ndarray:
    """Teacher cosines for a preselected [N, K] neighbour table (student-kNN arm)."""
    if indices.ndim != 2:
        raise ValueError(f"indices must be [N, K], got shape={indices.shape}")
    if int(embeddings.size(0)) != int(indices.shape[0]):
        raise ValueError(
            f"indices have {indices.shape[0]} rows but embeddings have "
            f"{int(embeddings.size(0))}"
        )
    if indices.size and (indices.min() < 0 or indices.max() >= embeddings.size(0)):
        raise ValueError("indices contain an out-of-range corpus node")
    device = device or _default_device(embeddings)

    def _run(target: torch.device) -> np.ndarray:
        normalized = F.normalize(embeddings.float(), p=2, dim=-1).to(target)
        index_tensor = torch.from_numpy(indices.astype(np.int64, copy=False)).to(target)
        output = np.empty(indices.shape, dtype=np.float32)
        for start in tqdm(
            range(0, indices.shape[0], chunk_size),
            desc="GGPKD teacher scores on selected kNN",
        ):
            end = min(start + chunk_size, indices.shape[0])
            anchors = normalized[start:end]
            block = index_tensor[start:end]
            scores = torch.empty(
                (end - start, indices.shape[1]), dtype=normalized.dtype, device=target
            )
            for col in range(0, indices.shape[1], candidate_chunk_size):
                col_end = min(col + candidate_chunk_size, indices.shape[1])
                scores[:, col:col_end] = torch.einsum(
                    "bd,bkd->bk", anchors, normalized[block[:, col:col_end]]
                )
            output[start:end] = scores.cpu().numpy()
        return output

    return _without_tf32(_run, device, "selected-pair cosine")


def _build_transition(
    top_indices: np.ndarray,
    top_scores: np.ndarray,
    graph_k: int,
    row_temps: np.ndarray,
    knn_mode: str = "directed",
    holdout_edge_frac: float = 0.0,
    holdout_seed: int = 0,
    holdout_bandwidth: str = "full",
) -> tuple[
    list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, dict
]:
    """Neighbour lists, transition rows, and the bandwidth each row was built at.

    * ``directed`` (method): keep all of topk(i), so every row has k columns.
    * ``mutual``: keep j iff i is also in topk(j); a row left empty falls back to
      its raw top-k.

    `holdout_edge_frac` withholds a symmetric subset of the surviving edges before
    the rows are normalized; the rows renormalize over what is left.

    `holdout_bandwidth` decides whether the withheld edges still reach training
    through tau_j. ``full`` (the default) keeps tau_j = (s(1) - s(k)) / log k off
    the raw top-k, so a held-out pair's teacher score shapes every target in its
    row: that is a *target-only* holdout and story.md §6.4 requires it to be called
    one. ``surviving`` recomputes tau_j on what the holdout left, which is what an
    information holdout needs.
    """
    if knn_mode not in KNN_MODES:
        raise ValueError(f"knn_mode must be one of {KNN_MODES}, got {knn_mode!r}")
    if holdout_bandwidth not in HOLDOUT_BANDWIDTHS:
        raise ValueError(
            f"holdout_bandwidth must be one of {HOLDOUT_BANDWIDTHS}, "
            f"got {holdout_bandwidth!r}"
        )
    n_items = top_indices.shape[0]
    top_sets = (
        [set(top_indices[i, :graph_k].tolist()) for i in range(n_items)]
        if knn_mode == "mutual"
        else None
    )
    row_neighbors, row_probs, row_scores = [], [], []
    fallback_flags = np.zeros(n_items, dtype=bool)
    row_temps = np.asarray(row_temps, dtype=np.float64).copy()
    held_out_edges = 0
    holdout_starved = 0
    rebandwidthed = 0

    for i in tqdm(range(n_items), desc=f"GGPKD {knn_mode} kNN graph"):
        neighbors = top_indices[i, :graph_k].astype(np.int64)
        scores = top_scores[i, :graph_k].astype(np.float64)
        if top_sets is not None:
            keep = np.fromiter(
                (i in top_sets[int(j)] for j in neighbors),
                dtype=bool,
                count=neighbors.size,
            )
            if keep.any():
                neighbors, scores = neighbors[keep], scores[keep]
            else:
                fallback_flags[i] = True

        if holdout_edge_frac > 0.0:
            withheld = heldout_edge_mask(
                np.full(neighbors.shape, i, dtype=np.int64),
                neighbors,
                holdout_seed,
                holdout_edge_frac,
            )
            held_out_edges += int(withheld.sum())
            if withheld.all():
                # A row with no columns has no target; it keeps its nearest
                # neighbour and is counted, so the exception stays visible.
                holdout_starved += 1
                withheld[0] = False
            neighbors, scores = neighbors[~withheld], scores[~withheld]
            if holdout_bandwidth == "surviving" and scores.size >= 2:
                # The same rule, read off the row that training actually sees:
                # the m-th surviving neighbour sits log m nats below the nearest.
                span = float(scores[0] - scores[-1])
                row_temps[i] = max(span / np.log(scores.size), MIN_BANDWIDTH)
                rebandwidthed += 1

        centered = scores - scores.max()
        weights = np.exp(centered / float(row_temps[i]))
        weights = weights / max(float(weights.sum()), 1e-12)
        row_neighbors.append(neighbors)
        row_probs.append(weights.astype(np.float32))
        row_scores.append(scores.astype(np.float32))

    stats = {
        # After the loop, so a surviving-bandwidth build reports the temperatures
        # it actually wrote into the rows.
        "row_temp_mean": float(row_temps.mean()),
        "row_temp_min": float(row_temps.min()),
        "row_temp_max": float(row_temps.max()),
        "row_temp_p50": float(np.median(row_temps)),
        "degenerate_bandwidth_rows": int((row_temps <= MIN_BANDWIDTH).sum()),
        "held_out_edges": int(held_out_edges),
        "holdout_starved_rows": int(holdout_starved),
        "rebandwidthed_rows": int(rebandwidthed),
    }
    return row_neighbors, row_probs, row_scores, fallback_flags, row_temps, stats


def _pad_rows(
    row_neighbors: list[np.ndarray], row_probs: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    width = max(1, max(len(neighbors) for neighbors in row_neighbors))
    neighbors = np.full((len(row_neighbors), width), -1, dtype=np.int64)
    probs = np.zeros((len(row_neighbors), width), dtype=np.float32)
    for row, (nbrs, prob) in enumerate(zip(row_neighbors, row_probs)):
        neighbors[row, : len(nbrs)] = nbrs
        probs[row, : len(prob)] = prob
    return neighbors, probs


def _gini(values: np.ndarray) -> float:
    total = values.sum()
    if values.size == 0 or total <= 0:
        return 0.0
    ordered = np.sort(values)
    n = ordered.size
    rank = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (rank * ordered).sum()) / (n * total) - (n + 1.0) / n)


def _degree_stats(neighbors: np.ndarray) -> dict[str, float]:
    n_items = neighbors.shape[0]
    valid = neighbors >= 0
    degrees = valid.sum(axis=1)
    indegree = np.bincount(neighbors[valid].astype(np.int64), minlength=n_items)
    ordered = np.sort(indegree)[::-1]
    top = max(1, int(round(0.01 * n_items)))
    return {
        "avg_degree": float(degrees.mean()),
        "min_degree": float(degrees.min()),
        "max_degree": float(degrees.max()),
        "indegree_p50": float(np.percentile(indegree, 50)),
        "indegree_p99": float(np.percentile(indegree, 99)),
        "indegree_max": float(indegree.max()),
        "indegree_gini": _gini(indegree.astype(np.float64)),
        "hub_edge_share_top1pct": float(ordered[:top].sum() / max(1, indegree.sum())),
    }


def _target_sharpness_stats(
    neighbors: np.ndarray, probs: np.ndarray
) -> dict[str, float]:
    """KL(row || uniform on its support) near 0 means the rows carry no ordering."""
    valid = (neighbors >= 0) & (probs > 0)
    p = np.where(valid, probs.astype(np.float64), 0.0)
    entropy = -np.where(valid, p * np.log(np.where(valid, p, 1.0)), 0.0).sum(axis=1)
    support = valid.sum(axis=1)
    top1 = p.max(axis=1)
    return {
        "target_support": float(support.mean()),
        "target_min_support": float(support.min()),
        "target_kl_uniform": float((np.log(np.maximum(support, 1)) - entropy).mean()),
        "target_top1": float(top1.mean()),
        "target_degenerate_count": int((top1 > 0.99).sum()),
    }


def _write_knn_graph_log(
    log_dir: str,
    row_neighbors: list[np.ndarray],
    row_probs: list[np.ndarray],
    row_scores: list[np.ndarray],
    fallback_flags: np.ndarray,
    stats: dict,
) -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "knn_graph_neighbors.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "summary", **stats}, sort_keys=True) + "\n")
        handle.writelines(
            json.dumps(
                {
                    "type": "node",
                    "idx": idx,
                    "fallback_used": bool(fallback_flags[idx]),
                    "neighbors": nbrs.tolist(),
                    "transition_probs": [float(v) for v in probs],
                    "cosine_scores": [float(v) for v in scores],
                },
                sort_keys=True,
            )
            + "\n"
            for idx, (nbrs, probs, scores) in enumerate(
                zip(row_neighbors, row_probs, row_scores)
            )
        )
    return path


_METADATA_KEYS = (
    "artifact_version",
    "n_items",
    "graph_k",
    "bandwidth",
    "knn_mode",
    "holdout_edge_frac",
    "holdout_seed",
    "holdout_bandwidth",
    "neighbor_source",
    "teacher_fingerprint",
)


def _metadata_matches(artifact: dict, metadata: dict) -> tuple[bool, str]:
    cached = artifact.get("metadata", {})
    for key in _METADATA_KEYS:
        if cached.get(key) != metadata.get(key):
            return (
                False,
                f"{key}: cached={cached.get(key)!r} requested={metadata.get(key)!r}",
            )
    return True, ""


def build_or_load_ggpkd_artifact(
    teacher_embeddings: torch.Tensor,
    cache_path: str,
    log_dir: str,
    graph_k: int,
    fixed_bandwidth: bool = False,
    knn_mode: str = "directed",
    holdout_edge_frac: float = 0.0,
    holdout_seed: int = 0,
    holdout_bandwidth: str = "full",
    neighbor_source: str = "teacher",
    neighbor_embeddings: Callable[[], torch.Tensor] | None = None,
) -> dict:
    """Build the graph, or load it when every metadata key matches.

    `neighbor_source` other than "teacher" labels another encoder whose kNN picks
    each row's columns; `neighbor_embeddings` produces its corpus embeddings and is
    only called on a build. Scores, targets and bandwidths stay the teacher's, and
    the label is part of the cache key.
    """
    n_items = int(teacher_embeddings.size(0))
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "n_items": n_items,
        "graph_k": int(graph_k),
        "bandwidth": "median" if fixed_bandwidth else "knn",
        "knn_mode": str(knn_mode),
        "holdout_edge_frac": float(holdout_edge_frac),
        "holdout_seed": int(holdout_seed) if holdout_edge_frac > 0.0 else 0,
        "holdout_bandwidth": (
            str(holdout_bandwidth) if holdout_edge_frac > 0.0 else "full"
        ),
        "neighbor_source": str(neighbor_source),
        "teacher_fingerprint": _fingerprint(teacher_embeddings),
    }

    artifact_path = Path(cache_path)
    if artifact_path.exists():
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        matches, reason = _metadata_matches(artifact, metadata)
        if matches:
            print(f"Loaded GGPKD artifact from: {artifact_path}")
            _print_graph_summary(artifact.get("graph_stats", {}))
            return artifact
        print(f"GGPKD artifact config mismatch, rebuilding: {artifact_path}")
        print(f"  first mismatch -> {reason}")
    if str(artifact_path.parent):
        os.makedirs(artifact_path.parent, exist_ok=True)

    device = _default_device(teacher_embeddings)
    teacher_indices, teacher_scores = _compute_topk_cosine(
        teacher_embeddings, k=graph_k, device=device
    )
    if neighbor_source == "teacher":
        top_indices, top_scores = teacher_indices, teacher_scores
    else:
        if neighbor_embeddings is None:
            raise ValueError(
                f"neighbor_source={neighbor_source!r} needs neighbor_embeddings"
            )
        neighbor_matrix = neighbor_embeddings()
        if int(neighbor_matrix.size(0)) != n_items:
            raise ValueError(
                f"neighbor embeddings have {int(neighbor_matrix.size(0))} rows but "
                f"there are {n_items} teacher embeddings"
            )
        top_indices, _ = _compute_topk_cosine(neighbor_matrix, k=graph_k, device=device)
        top_scores = _cosine_scores_at_indices(
            teacher_embeddings, top_indices, device=device
        )
        metadata["neighbor_fingerprint"] = _fingerprint(neighbor_matrix)

    # Bandwidths always come from the teacher's own top-k span.
    row_temps = _knn_bandwidths(teacher_scores, graph_k)
    if fixed_bandwidth:
        row_temps = np.full(n_items, float(np.median(row_temps)))

    row_neighbors, row_probs, row_scores, fallback_flags, row_temps, build_stats = (
        _build_transition(
            top_indices=top_indices,
            top_scores=top_scores,
            graph_k=graph_k,
            row_temps=row_temps,
            knn_mode=knn_mode,
            holdout_edge_frac=holdout_edge_frac,
            holdout_seed=holdout_seed,
            holdout_bandwidth=holdout_bandwidth,
        )
    )
    neighbors, probs = _pad_rows(row_neighbors, row_probs)
    graph_stats = {
        "n_items": n_items,
        "graph_k": int(graph_k),
        "fallback_count": int(fallback_flags.sum()),
        **build_stats,
        **_degree_stats(neighbors),
        **reciprocal_component_stats(neighbors),
        **_target_sharpness_stats(neighbors, probs),
    }
    log_path = _write_knn_graph_log(
        log_dir, row_neighbors, row_probs, row_scores, fallback_flags, graph_stats
    )
    artifact = {
        "transition_neighbors": torch.from_numpy(neighbors),
        "transition_probs": torch.from_numpy(probs),
        "row_temps": torch.from_numpy(row_temps.astype(np.float32)),
        "graph_log_path": log_path,
        "graph_stats": graph_stats,
        "metadata": metadata,
    }
    torch.save(artifact, artifact_path)
    print(f"Saved GGPKD artifact to: {artifact_path}")
    _print_graph_summary(graph_stats)
    return artifact


def _print_graph_summary(stats: dict) -> None:
    if not stats:
        return
    print(
        "GGPKD graph: "
        f"n={stats.get('n_items')} k={stats.get('graph_k')} "
        f"degree={float(stats.get('avg_degree', 0.0)):.1f} "
        f"fallback={stats.get('fallback_count', 0)} "
        f"reciprocal components={stats.get('reciprocal_components')} "
        f"(isolated {stats.get('reciprocal_isolated')}, largest "
        f"{float(stats.get('reciprocal_largest_frac', 0.0)):.3f})"
    )
    print(
        "GGPKD targets: "
        f"support={float(stats.get('target_support', 0.0)):.1f} "
        f"KL(p||uniform)={float(stats.get('target_kl_uniform', 0.0)):.4f} "
        f"top1={float(stats.get('target_top1', 0.0)):.4f} "
        f"tau_p50={float(stats.get('row_temp_p50', 0.0)):.4f}"
    )
    degenerate = int(stats.get("target_degenerate_count", 0))
    if degenerate:
        print(
            f"WARNING: GGPKD {degenerate} rows have a near one-hot target "
            f"(min support {int(stats.get('target_min_support', 0))})"
        )
    if float(stats.get("target_kl_uniform", 1.0)) < 0.05:
        print("WARNING: GGPKD rows are nearly uniform; graph_k is too large")
