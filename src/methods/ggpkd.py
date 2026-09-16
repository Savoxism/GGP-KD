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
from src.methods.support import dedup_anchor_frame


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
        holdout_bandwidth=cfg.holdout_bandwidth,
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
        pool_source=cfg.pool_source,
        # The pool draw is part of the run, not of the graph: it moves with the
        # training seed so two seeds of the control see two different pools.
        pool_seed=int(getattr(cfg, "seed", 0) or 0),
    )
    if cfg.batch_local:
        print(
            f"GGPKD in-batch baseline: the pool is the batch ({cfg.batch_size} texts)"
        )
    elif cfg.pool_source == "random":
        print(
            f"GGPKD random-pool control: {cfg.batch_size} anchors plus uniformly "
            "drawn texts, as many as this batch's graph pool would have held"
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
        f"pool_source={config.pool_source} "
        f"neighbor_source={config.neighbor_source}"
    )
    return criterion


SPEC = MethodSpec(
    name="ggpkd",
    config_cls=GGPKDConfig,
    step=step,
    uses_teacher_cache=True,
    batch_relational=True,
    prepare_frame=dedup_anchor_frame,
    build_data=build_data,
    build_criterion=build_criterion,
)
