"""Neighbour-composed batches (story.md, Table 3 rows 2 and 4).

    random    the loader's own i.i.d. shuffle. The method's setting.
    neighbor  each batch is filled from one graph neighbourhood, so a batch-local
              objective sees related texts in its batch.

The neighbour sampler is a partition: every corpus index appears exactly once per
epoch, so anchor set, step count and update count match the random arm and only
the grouping differs.
"""

from collections.abc import Iterator, Sequence

import numpy as np

BATCH_SAMPLERS = ("random", "neighbor")


def _neighbor_batches(
    neighbors: np.ndarray, batch_size: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Greedy partition: an unused random seed plus its nearest unused neighbours.

    A seed whose neighbours are used up is topped up from the unused pool at random,
    so every batch keeps the same size.
    """
    n_items = int(neighbors.shape[0])
    used = np.zeros(n_items, dtype=bool)
    order = rng.permutation(n_items)
    batches: list[np.ndarray] = []
    cursor = 0
    while True:
        while cursor < n_items and used[order[cursor]]:
            cursor += 1
        if cursor >= n_items:
            break
        seed = int(order[cursor])
        used[seed] = True
        members = [seed]
        for candidate in neighbors[seed]:
            if len(members) >= batch_size:
                break
            candidate = int(candidate)
            if candidate < 0 or used[candidate]:
                continue
            used[candidate] = True
            members.append(candidate)
        if len(members) < batch_size:
            remaining = np.flatnonzero(~used)
            take = min(batch_size - len(members), remaining.size)
            if take > 0:
                filler = rng.choice(remaining, size=take, replace=False)
                used[filler] = True
                members.extend(int(index) for index in filler)
        batches.append(np.asarray(members, dtype=np.int64))
    return batches


class NeighborBatchSampler:
    """A `batch_sampler` for `DataLoader`, regrouped every epoch from (seed, epoch)."""

    def __init__(
        self,
        neighbors: np.ndarray,
        batch_size: int,
        seed: int,
        drop_last: bool = True,
    ):
        if batch_size < 2:
            raise ValueError(
                f"batch composition needs batch_size >= 2, got {batch_size}"
            )
        self.neighbors = np.ascontiguousarray(neighbors, dtype=np.int32)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self._batches: list[list[int]] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._batches = None

    def _build(self) -> list[list[int]]:
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, 0xB47C])
        )
        batches = []
        for group in _neighbor_batches(self.neighbors, self.batch_size, rng):
            members = group.copy()
            rng.shuffle(members)
            batches.append([int(index) for index in members])
        rng.shuffle(batches)
        if self.drop_last:
            batches = [batch for batch in batches if len(batch) == self.batch_size]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        if self._batches is None:
            self._batches = self._build()
        return iter(self._batches)

    def __len__(self) -> int:
        if self._batches is None:
            self._batches = self._build()
        return len(self._batches)


def batch_relevance_stats(
    batches: Sequence[Sequence[int]], neighbors: np.ndarray
) -> dict[str, float]:
    """Share of each anchor's batch-mates that are its graph neighbours.

    The measured count for Table 3 ("nbrs / row"), so the neighbour arm reports the
    exposure it achieved rather than the one it intended.
    """
    membership = [set(int(j) for j in row if j >= 0) for row in neighbors]
    hits_per_row, precision = [], []
    for batch in batches:
        members = set(int(index) for index in batch)
        if len(members) < 2:
            continue
        for anchor in members:
            hits = len((members - {anchor}) & membership[anchor])
            hits_per_row.append(hits)
            precision.append(hits / (len(members) - 1))
    if not hits_per_row:
        return {}
    return {
        "in_batch_neighbors_per_row": float(np.mean(hits_per_row)),
        "in_batch_precision": float(np.mean(precision)),
        "n_batches": float(len(batches)),
    }
