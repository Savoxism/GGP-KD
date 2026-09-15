"""GGPKD's dataset and collate: a step's anchors and the pool they expand to.

The dataset yields corpus positions only. The collate turns a batch of anchors
into the step's pool -- the anchors plus their graph rows, deduplicated -- and
encodes each pool text once, grouped by length so a chunk pads to its own longest
member.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.ggpkd.graph_builder import heldout_edge_mask
from src.ggpkd.policy import ENCODE_CHUNK_SIZE, PAD_TO_MULTIPLE_OF


class GGPKDAnchorDataset(Dataset):
    """Corpus positions of the deduplicated anchor texts."""

    def __init__(self, n_items: int):
        self.n_items = int(n_items)

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, idx: int) -> dict:
        return {"idx": int(idx)}


class GGPKDCollate:
    """Pool construction and length-bucketed encoding for one step.

    Output, with P the pool size and B the number of anchors:

        pool_idx      [P]    sorted corpus indices of the pool
        anchor_pos    [B]    position of each anchor in pool_idx
        pool_chunks          tokenized pool texts in encode order
        pool_inverse  [P]    encode row of each pool position
        cal_exclude   [B, P] pairs withheld by the edge holdout (only with a holdout)

    `neighbors=None` is the in-batch baseline: the pool is the batch itself.
    """

    def __init__(
        self,
        tok_student,
        max_len: int,
        corpus_texts: list[str],
        neighbors: np.ndarray | None,
        holdout_edge_frac: float = 0.0,
        holdout_seed: int = 0,
        encode_chunk_size: int = ENCODE_CHUNK_SIZE,
        pad_to_multiple_of: int = PAD_TO_MULTIPLE_OF,
    ):
        self.neighbors = (
            None
            if neighbors is None
            else np.ascontiguousarray(neighbors, dtype=np.int64)
        )
        self.holdout_edge_frac = float(holdout_edge_frac)
        self.holdout_seed = int(holdout_seed)
        self.encode_chunk_size = int(encode_chunk_size)
        self.pad_to_multiple_of = int(pad_to_multiple_of)
        self.pad_id = tok_student.pad_token_id
        if self.pad_id is None:
            raise ValueError("student tokenizer has no pad_token_id")
        encoded = tok_student(
            [str(text) for text in corpus_texts], max_length=max_len, truncation=True
        )["input_ids"]
        self.corpus_ids = [np.asarray(ids, dtype=np.int64) for ids in encoded]
        self.corpus_len = np.asarray(
            [ids.size for ids in self.corpus_ids], dtype=np.int64
        )
        if self.neighbors is not None and self.neighbors.shape[0] != len(
            self.corpus_ids
        ):
            raise ValueError(
                f"graph has {self.neighbors.shape[0]} rows but the corpus has "
                f"{len(self.corpus_ids)} texts"
            )

    def _pad(self, nodes: np.ndarray) -> dict[str, torch.Tensor]:
        rows = [self.corpus_ids[int(node)] for node in nodes]
        width = max(ids.size for ids in rows)
        if self.pad_to_multiple_of > 1:
            multiple = self.pad_to_multiple_of
            width = ((width + multiple - 1) // multiple) * multiple
        input_ids = np.full((len(rows), width), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros((len(rows), width), dtype=np.int64)
        for row, ids in enumerate(rows):
            input_ids[row, : ids.size] = ids
            attention_mask[row, : ids.size] = 1
        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention_mask),
        }

    def __call__(self, batch: list[dict]) -> dict:
        idx = np.asarray([item["idx"] for item in batch], dtype=np.int64)
        if self.neighbors is None:
            if idx.size < 2:
                raise ValueError("in-batch relations need at least two texts per batch")
            nodes = idx
        else:
            rows = self.neighbors[idx]
            nodes = np.concatenate([idx, rows[rows >= 0]])
        pool_idx = np.unique(nodes)
        anchor_pos = np.searchsorted(pool_idx, idx)

        order = np.argsort(self.corpus_len[pool_idx], kind="stable")
        pool_inverse = np.empty(order.size, dtype=np.int64)
        pool_inverse[order] = np.arange(order.size, dtype=np.int64)
        ordered = pool_idx[order]
        pool_chunks = [
            self._pad(ordered[start : start + self.encode_chunk_size])
            for start in range(0, ordered.size, self.encode_chunk_size)
        ]

        out = {
            "idx": torch.from_numpy(idx),
            "pool_idx": torch.from_numpy(pool_idx),
            "anchor_pos": torch.from_numpy(anchor_pos),
            "pool_chunks": pool_chunks,
            "pool_inverse": torch.from_numpy(pool_inverse),
        }
        if self.holdout_edge_frac > 0.0:
            # Withheld pairs must not reach L_cal either, or "never supervised" is
            # false for the arms that keep the calibration term.
            rows = np.broadcast_to(idx[:, None], (idx.size, pool_idx.size))
            cols = np.broadcast_to(pool_idx[None, :], (idx.size, pool_idx.size))
            out["cal_exclude"] = torch.from_numpy(
                heldout_edge_mask(rows, cols, self.holdout_seed, self.holdout_edge_frac)
            )
        return out
