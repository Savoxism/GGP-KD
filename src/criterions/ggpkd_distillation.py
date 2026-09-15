"""GGPKD loss: row-conditional matching on a step's pool, plus pool calibration.

    L = row_weight * L_row + cal_weight * L_cal

L_row   For every supervised pool text j,
            KL( P^T_j|Omega_j || P^S_j|Omega_j ),   Omega_j = N(j) ∩ pool,
        both sides softmaxes at the row's own tau_j, renormalized on Omega_j. By
        Luce's choice axiom a partial row is an exact constraint: its zero set is
        s^S_ju - s^T_ju = const_j on Omega_j (story.md §3, Lemma 1). An anchor's
        Omega_i is its whole row.
L_cal   For every anchor i, KL between the teacher's and the student's softmax
        over the whole pool at one temperature (the median tau_j). It is the only
        term that scores pairs outside the graph (Proposition 2).

Ablation switches, each at the method's value by default (story.md, Tables 3-6):

    row_set        all | anchors | non_anchors   which pool texts are rows
    row_target     teacher | uniform             values on the same columns
    row_columns    graph | random                same width, random pool columns
    row_inclusion  None | P(j in pool)           weight rows by 1 / p_j
"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.ggpkd.policy import EPS_NORM

ROW_SETS = ("all", "anchors", "non_anchors")
ROW_TARGETS = ("teacher", "uniform")
ROW_COLUMNS = ("graph", "random")

_ROW_METRICS = (
    "row_count",
    "row_eff_denom",
    "row_exposed_mass",
    "row_exposed_mass_p10",
    "row_exposed_mass_p90",
    "row_teacher_entropy",
    "row_kl_p50",
    "row_kl_p90",
    "row_ess_ratio",
)


def _assert_finite(named: Sequence[tuple[str, torch.Tensor]]) -> None:
    for name, tensor in named:
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(
                f"GGPKD non-finite tensor {name!r}: shape={tuple(tensor.shape)}, "
                f"nan={int(torch.isnan(tensor).sum())}, inf={int(torch.isinf(tensor).sum())}"
            )


def _kl_rows(target: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """Row-wise KL(target || q); columns with zero target contribute nothing."""
    positive = target > 0
    log_p = torch.where(
        positive, target.clamp_min(1e-12).log(), torch.zeros_like(target)
    )
    return torch.where(
        positive, target * (log_p - log_q), torch.zeros_like(target)
    ).sum(-1)


def _entropy(p: torch.Tensor) -> torch.Tensor:
    positive = p > 0
    return -torch.where(
        positive, p * p.clamp_min(1e-12).log(), torch.zeros_like(p)
    ).sum(-1)


class GGPKDDistillation(nn.Module):
    def __init__(
        self,
        teacher_embeddings: torch.Tensor,
        transition_neighbors: torch.Tensor,
        transition_probs: torch.Tensor,
        row_temps: torch.Tensor,
        cal_temp: float,
        cal_weight: float = 0.5,
        row_weight: float = 1.0,
        row_set: str = "all",
        row_target: str = "teacher",
        row_columns: str = "graph",
        row_inclusion: torch.Tensor | None = None,
    ):
        super().__init__()
        for name, value in (("cal_weight", cal_weight), ("row_weight", row_weight)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        if cal_weight == 0 and row_weight == 0:
            raise ValueError(
                "at least one of cal_weight and row_weight must be positive"
            )
        for name, value, choices in (
            ("row_set", row_set, ROW_SETS),
            ("row_target", row_target, ROW_TARGETS),
            ("row_columns", row_columns, ROW_COLUMNS),
        ):
            if value not in choices:
                raise ValueError(f"{name} must be one of {choices}, got {value!r}")
        if not math.isfinite(cal_temp) or cal_temp <= 0:
            raise ValueError(f"cal_temp must be finite and positive, got {cal_temp}")

        n_items = int(teacher_embeddings.size(0))
        temps = row_temps.detach().float().reshape(-1)
        if temps.numel() != n_items or tuple(transition_neighbors.shape[:1]) != (
            n_items,
        ):
            raise ValueError(
                "graph arrays and teacher embeddings disagree on the corpus size"
            )
        if not bool(torch.isfinite(temps).all()) or bool((temps <= 0).any()):
            raise ValueError("row_temps must be finite and positive")

        self.cal_weight = float(cal_weight)
        self.row_weight = float(row_weight)
        self.cal_temp = float(cal_temp)
        self.row_set = row_set
        self.row_target = row_target
        self.row_columns = row_columns
        # Normalized once, in half precision: it only feeds cosines, and at corpus
        # scale it is the largest buffer the criterion owns.
        self.register_buffer(
            "teacher_bank",
            F.normalize(teacher_embeddings.float(), p=2, dim=-1, eps=EPS_NORM).half(),
            persistent=False,
        )
        self.register_buffer(
            "row_neighbors", transition_neighbors.long(), persistent=False
        )
        self.register_buffer("row_probs", transition_probs.float(), persistent=False)
        self.register_buffer("row_temps", temps, persistent=False)
        if row_inclusion is not None:
            p = row_inclusion.detach().float().reshape(-1)
            if p.numel() != n_items or bool((p <= 0).any()) or bool((p > 1).any()):
                raise ValueError(
                    "row_inclusion must hold one probability in (0, 1] per text"
                )
            self.register_buffer("row_weights", 1.0 / p, persistent=False)
        else:
            self.row_weights = None

    def forward(
        self,
        pool_embeddings: torch.Tensor,
        pool_idx: torch.Tensor,
        anchor_pos: torch.Tensor,
        cal_exclude: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pool_norm = F.normalize(pool_embeddings.float(), p=2, dim=-1, eps=EPS_NORM)
        zero = pool_norm.new_zeros(())
        entries: list[tuple[str, torch.Tensor]] = [
            ("pool_size", zero + pool_idx.numel()),
        ]

        loss_cal = zero
        if self.cal_weight > 0:
            loss_cal, cal_entries = self._calibration_loss(
                pool_norm, pool_idx, anchor_pos, cal_exclude
            )
            entries += cal_entries
        loss_row = zero
        if self.row_weight > 0:
            loss_row, row_entries = self._row_loss(pool_norm, pool_idx, anchor_pos)
            entries += row_entries

        cal_term = self.cal_weight * loss_cal
        row_term = self.row_weight * loss_row
        total = cal_term + row_term
        try:
            _assert_finite((("loss_cal", loss_cal), ("loss_row", loss_row)))
        except RuntimeError:
            # A non-finite loss almost always comes from the encoder; name it.
            _assert_finite((("pool_embeddings", pool_embeddings),))
            raise

        entries = [
            ("loss_total", total),
            ("loss_cal", loss_cal),
            ("loss_row", loss_row),
            ("loss_cal_weighted", cal_term),
            ("loss_row_weighted", row_term),
            ("row_share", row_term / total.clamp_min(1e-12)),
            *entries,
        ]
        # One device sync for every logged scalar.
        values = torch.stack(
            [value.detach().float().reshape(()) for _, value in entries]
        )
        return total, dict(zip((name for name, _ in entries), values.tolist()))

    def _calibration_loss(
        self,
        pool_norm: torch.Tensor,
        pool_idx: torch.Tensor,
        anchor_pos: torch.Tensor,
        cal_exclude: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]]]:
        student = pool_norm.index_select(0, anchor_pos) @ pool_norm.t() / self.cal_temp
        with torch.no_grad():
            teacher_pool = self.teacher_bank.index_select(0, pool_idx).float()
            teacher = (
                teacher_pool.index_select(0, anchor_pos)
                @ teacher_pool.t()
                / self.cal_temp
            )
            mask = torch.zeros_like(teacher, dtype=torch.bool)
            mask[torch.arange(anchor_pos.numel(), device=mask.device), anchor_pos] = (
                True
            )
            if cal_exclude is not None:
                mask |= cal_exclude
            target = F.softmax(teacher.masked_fill(mask, float("-inf")), dim=-1)
        log_q = F.log_softmax(student.masked_fill(mask, float("-inf")), dim=-1)
        loss = _kl_rows(target, log_q).mean()
        return loss, [
            ("cal_columns", (~mask).sum(-1).float().mean()),
            ("cal_teacher_entropy", _entropy(target).mean()),
            ("cal_student_entropy", _entropy(log_q.detach().exp()).mean()),
        ]

    def _row_loss(
        self,
        pool_norm: torch.Tensor,
        pool_idx: torch.Tensor,
        anchor_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]]]:
        zero = pool_norm.new_zeros(())
        pool_size = pool_idx.numel()
        device = pool_idx.device

        is_anchor = torch.zeros(pool_size, dtype=torch.bool, device=device)
        is_anchor[anchor_pos] = True
        if self.row_set == "anchors":
            centers = is_anchor
        elif self.row_set == "non_anchors":
            centers = ~is_anchor
        else:
            centers = torch.ones_like(is_anchor)
        positions = centers.nonzero(as_tuple=True)[0]
        nodes = pool_idx.index_select(0, positions)

        with torch.no_grad():
            neighbors = self.row_neighbors.index_select(0, nodes)
            probs = self.row_probs.index_select(0, nodes)
            column = torch.searchsorted(pool_idx, neighbors.clamp_min(0)).clamp_max(
                pool_size - 1
            )
            present = (
                (neighbors >= 0)
                & (pool_idx[column] == neighbors)
                & (probs > 0)
                & (column != positions.unsqueeze(1))
            )
            target = torch.zeros(nodes.numel(), pool_size, device=device)
            target.scatter_add_(
                1, column, torch.where(present, probs, torch.zeros_like(probs))
            )
            exposed = target.sum(-1)
            allowed = target > 0
            usable = allowed.sum(-1) >= 2

        if not bool(usable.any()):
            return zero, [(name, zero) for name in _ROW_METRICS]
        positions, nodes = positions[usable], nodes[usable]
        target, allowed, exposed = target[usable], allowed[usable], exposed[usable]
        tau = self.row_temps.index_select(0, nodes).unsqueeze(1)

        with torch.no_grad():
            target = target / target.sum(-1, keepdim=True)
            if self.row_columns == "random":
                allowed, target = self._random_columns(
                    positions, nodes, pool_idx, allowed, tau
                )
            if self.row_target == "uniform":
                target = allowed.float() / allowed.sum(-1, keepdim=True)

        logits = pool_norm.index_select(0, positions) @ pool_norm.t() / tau
        log_q = F.log_softmax(logits.masked_fill(~allowed, float("-inf")), dim=-1)
        kl = _kl_rows(target, log_q)
        if self.row_weights is not None:
            weights = self.row_weights.index_select(0, nodes)
            loss = (weights * kl).sum() / weights.sum()
            ess_ratio = weights.sum() ** 2 / (weights * weights).sum() / weights.numel()
        else:
            loss = kl.mean()
            ess_ratio = zero + 1.0

        detached = kl.detach()
        return loss, [
            ("row_count", zero + positions.numel()),
            ("row_eff_denom", allowed.sum(-1).float().mean()),
            ("row_exposed_mass", exposed.mean()),
            ("row_exposed_mass_p10", exposed.quantile(0.10)),
            ("row_exposed_mass_p90", exposed.quantile(0.90)),
            ("row_teacher_entropy", _entropy(target).mean()),
            ("row_kl_p50", detached.quantile(0.50)),
            ("row_kl_p90", detached.quantile(0.90)),
            ("row_ess_ratio", ess_ratio),
        ]

    @torch.no_grad()
    def _random_columns(
        self,
        positions: torch.Tensor,
        nodes: torch.Tensor,
        pool_idx: torch.Tensor,
        allowed: torch.Tensor,
        tau: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Each row keeps its width; its columns are drawn uniformly from the pool.

        The target is the teacher's softmax at tau_j over the drawn columns, which on
        the graph columns is exactly the renormalized transition row, so the only
        change is *which* texts row j is compared against (Table 3, row 8).
        """
        rows, pool_size = allowed.shape
        counts = allowed.sum(-1)
        scores = torch.rand(rows, pool_size, device=allowed.device)
        scores[torch.arange(rows, device=allowed.device), positions] = -1.0
        kth = scores.sort(dim=-1, descending=True).values.gather(
            1, (counts - 1).clamp_min(0).unsqueeze(1)
        )
        chosen = scores >= kth
        bank = self.teacher_bank
        logits = (
            bank.index_select(0, nodes).float()
            @ bank.index_select(0, pool_idx).float().t()
        ) / tau
        return chosen, F.softmax(logits.masked_fill(~chosen, float("-inf")), dim=-1)
