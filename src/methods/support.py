"""Helpers shared by the method specs."""

import numpy as np
import pandas as pd


def attach_parameters(ctx, parameters, lr):
    """Give the optimizer a new param group and rebuild the schedule over it.

    Three criteria own trainable parameters, and adding a param group *after* the
    scheduler exists leaves `LambdaLR` holding one `lr_lambda` for two groups;
    torch >= 2.6 zips them with `strict=True`, so the first `scheduler.step()`
    raises. Every method that adds a group must therefore rebuild the scheduler in
    the same breath, and keeping the two steps apart is what let CDM drift into
    that bug once already.
    """
    ctx.optimizer.add_param_group({"params": parameters, "lr": lr})
    ctx.scheduler = ctx.build_scheduler()


def resolve_anchor_column(ctx, df: pd.DataFrame) -> str:
    """The text column a corpus-graph method treats as the node of each row."""
    cfg = ctx.config
    column = getattr(cfg, "ggpkd_anchor_column", None)
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
        raise ValueError(f"need column {column!r} for task_type={cfg.task_type!r}")
    # The graph is built over this column only; say so if a real second view exists.
    partner = {"pair_cls": "hypothesis", "pair_reg": "sentence2"}.get(cfg.task_type)
    if partner in df.columns and not df[column].equals(df[partner]):
        print(
            f"WARNING: only {column!r} is distilled; {partner!r} differs from it. "
            "Set ggpkd_anchor_column explicitly if that is not intended."
        )
    return column


def dedup_anchor_frame(ctx, df: pd.DataFrame):
    """Resolve the anchor column and drop exact duplicate anchors.

    Two identical texts have cosine 1 under every parameter setting, so a duplicate
    is a column with no gradient that still takes a pool slot. Every method that
    reads the corpus graph uses this one function, so graph node i is dataset row i
    for all of them -- the pointwise neighbour-batching arm included.
    """
    ctx.ggpkd_anchor_column = resolve_anchor_column(ctx, df)
    duplicated = (
        df[ctx.ggpkd_anchor_column].astype(str).duplicated(keep="first").to_numpy()
    )
    keep = np.flatnonzero(~duplicated).astype(np.int64)
    if duplicated.any():
        print(
            f"Corpus dedup on {ctx.ggpkd_anchor_column!r}: {len(df)} -> {keep.size} "
            f"rows ({int(duplicated.sum())} exact duplicates removed)"
        )
    return df.iloc[keep].reset_index(drop=True), keep
