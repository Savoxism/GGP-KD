"""GGPKD: subgraph distillation over the teacher's kNN transition rows (story.md §2)."""

import numpy as np
import pandas as pd
import torch

from config import GGPKDConfig
from src.criterions.ggpkd_distillation import GGPKDDistillation
from src.data_utils.ggpkd_dataset import GGPKDAnchorDataset, GGPKDCollate
from src.distill.geometry import build_probe_index
from src.distill.steps.ggpkd import step
from src.ggpkd.graph_builder import (
    build_or_load_ggpkd_artifact,
    pool_inclusion_probability,
)
from src.ggpkd.student_neighbors import encode_base_student
from src.methods.spec import MethodSpec


def resolve_anchor_column(ctx, df: pd.DataFrame) -> str:
    cfg = ctx.config
    column = cfg.ggpkd_anchor_column
    if column is not None:
        if column not in df.columns:
            raise ValueError(
                f"ggpkd_anchor_column={column!r} is not a column of "
                f"{cfg.train_data_path} (have {list(df.columns)})"
            )
        return column

    column = {"single_cls": "text", "pair_cls": "premise"}.get(
        cfg.task_type, "sentence1"
    )
    if column not in df.columns:
        raise ValueError(
            f"GGPKD needs column {column!r} for task_type={cfg.task_type!r}"
        )
    # The graph is built over this column only; say so if a real second view exists.
    partner = {"pair_cls": "hypothesis", "pair_reg": "sentence2"}.get(cfg.task_type)
    if partner in df.columns and not df[column].equals(df[partner]):
        print(
            f"WARNING: GGPKD uses only {column!r}; {partner!r} differs from it and is "
            "not distilled. Set ggpkd_anchor_column explicitly if that is not intended."
        )
    return column


def prepare_frame(ctx, df: pd.DataFrame):
    """Resolve the anchor column and drop exact duplicate anchors.

    Two identical texts have cosine 1 for every parameter setting, so a duplicate
    is a column with no gradient that still takes a pool slot.
    """
    ctx.ggpkd_anchor_column = resolve_anchor_column(ctx, df)
    duplicated = (
        df[ctx.ggpkd_anchor_column].astype(str).duplicated(keep="first").to_numpy()
    )
    keep = np.flatnonzero(~duplicated).astype(np.int64)
    if duplicated.any():
        print(
            f"GGPKD corpus dedup on {ctx.ggpkd_anchor_column!r}: {len(df)} -> {keep.size} "
            f"rows ({int(duplicated.sum())} exact duplicates removed)"
        )
    return df.iloc[keep].reset_index(drop=True), keep


def build_data(ctx, df: pd.DataFrame, teacher_cls: torch.Tensor):
    """Build or load the graph, then the anchor dataset and the pool collate."""
    cfg = ctx.config
    anchor_texts = df[ctx.ggpkd_anchor_column].astype(str).tolist()
    neighbor_kwargs = {}
    if cfg.neighbor_source == "student":
        neighbor_kwargs = {
            # Everything that decides the student's embeddings is in the label, so
            # the artifact never loads under another source's name.
            "neighbor_source": (
                f"student:{cfg.student_model_name}:cls:max_length={cfg.max_length}"
            ),
            "neighbor_embeddings": lambda: encode_base_student(
                cfg.student_model_name,
                ctx.tok_student,
                anchor_texts,
                cfg.max_length,
                device=ctx.device_s,
            ),
        }
    ctx.ggpkd_artifact = build_or_load_ggpkd_artifact(
        teacher_embeddings=teacher_cls,
        cache_path=cfg.ggpkd_cache_path,
        log_dir=cfg.ggpkd_log_dir,
        graph_k=cfg.graph_k,
        fixed_bandwidth=cfg.fixed_bandwidth,
        knn_mode=cfg.knn_mode,
        holdout_edge_frac=cfg.holdout_edge_frac,
        holdout_seed=cfg.holdout_seed,
        **neighbor_kwargs,
    )

    # The probe is sampled from the deduplicated corpus, whose rows index the cache.
    probe_index = build_probe_index(len(anchor_texts), size=2048, seed=0)
    ctx.probe_texts = [anchor_texts[int(i)] for i in probe_index]
    ctx.probe_teacher = teacher_cls[torch.from_numpy(np.asarray(probe_index)).long()]

    neighbors = ctx.ggpkd_artifact["transition_neighbors"].numpy()
    ctx.train_ds = GGPKDAnchorDataset(len(anchor_texts))
    ctx.collate_fn = GGPKDCollate(
        ctx.tok_student,
        cfg.max_length,
        corpus_texts=anchor_texts,
        neighbors=None if cfg.batch_local else neighbors,
        holdout_edge_frac=cfg.holdout_edge_frac,
        holdout_seed=cfg.holdout_seed,
    )
    if cfg.batch_local:
        print(
            f"GGPKD in-batch baseline: the pool is the batch ({cfg.batch_size} texts)"
        )
    else:
        degree = (neighbors >= 0).sum(axis=1)
        print(
            f"GGPKD pool: {cfg.batch_size} anchors plus their graph rows "
            f"(row width mean={degree.mean():.1f} min={int(degree.min())})"
        )
    return ctx.train_ds, ctx.collate_fn


def build_criterion(ctx, config):
    artifact = ctx.ggpkd_artifact
    row_temps = artifact["row_temps"]
    # The calibration temperature is derived: the graph's median row bandwidth.
    config.cal_temp = float(row_temps.median())
    inclusion = None
    if config.row_reweight:
        p = pool_inclusion_probability(
            artifact["transition_neighbors"].numpy(), config.batch_size
        )
        weights = 1.0 / p
        print(
            "GGPKD row reweighting by 1/p_j: "
            f"p_j median={np.median(p):.3f} max/median={p.max() / np.median(p):.2f} "
            f"ESS/N={weights.sum() ** 2 / (weights**2).sum() / p.size:.2f}"
        )
        inclusion = torch.from_numpy(p)
    criterion = GGPKDDistillation(
        teacher_embeddings=ctx.teacher_cls_all,
        transition_neighbors=artifact["transition_neighbors"],
        transition_probs=artifact["transition_probs"],
        row_temps=row_temps,
        cal_temp=config.cal_temp,
        cal_weight=config.cal_weight,
        row_weight=config.row_weight,
        row_set=config.row_set,
        row_target=config.row_target,
        row_columns=config.row_columns,
        row_inclusion=inclusion,
    ).to(ctx.device_s)
    print(
        "GGPKD criterion: "
        f"cal_weight={config.cal_weight} row_weight={config.row_weight} "
        f"cal_temp={config.cal_temp:.4f} row_set={config.row_set} "
        f"row_target={config.row_target} row_columns={config.row_columns} "
        f"row_reweight={config.row_reweight} batch_local={config.batch_local} "
        f"neighbor_source={config.neighbor_source}"
    )
    return criterion


SPEC = MethodSpec(
    name="ggpkd",
    config_cls=GGPKDConfig,
    step=step,
    uses_teacher_cache=True,
    batch_relational=True,
    prepare_frame=prepare_frame,
    build_data=build_data,
    build_criterion=build_criterion,
)
