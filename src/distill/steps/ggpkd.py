"""The ggpkd training step.

Reads from the distiller context: config, device_s, model_student, criterion,
optimizer, scaler, scheduler, current_epoch, current_step.
"""

import math

import torch
from torch.amp import autocast

from src.distill.numerics import is_finite, total_grad_norm


def step(ctx, batch: dict) -> tuple[torch.Tensor, dict]:
    device = ctx.device_s
    pool_idx = batch["pool_idx"].to(device, non_blocking=True)
    anchor_pos = batch["anchor_pos"].to(device, non_blocking=True)
    pool_inverse = batch["pool_inverse"].to(device, non_blocking=True)
    cal_exclude = batch.get("cal_exclude")
    if cal_exclude is not None:
        cal_exclude = cal_exclude.to(device, non_blocking=True)

    ctx.optimizer.zero_grad(set_to_none=True)

    # Encoder budget, counted where the encoding happens. Read off the CPU batch:
    # int() on a CUDA tensor would sync every step for a counter.
    encoded_texts = 0
    encoded_tokens = 0
    with autocast("cuda", enabled=torch.cuda.is_available()):
        chunk_embeddings = []
        for chunk in batch["pool_chunks"]:
            out = ctx.model_student(
                input_ids=chunk["input_ids"].to(device, non_blocking=True),
                attention_mask=chunk["attention_mask"].to(device, non_blocking=True),
                return_dict=True,
                output_hidden_states=False,
            )
            chunk_embeddings.append(out.last_hidden_state[:, 0, :])
            encoded_texts += int(chunk["input_ids"].size(0))
            encoded_tokens += int(chunk["attention_mask"].sum())
        # Back from encode order to pool order. The gather is differentiable, so a
        # text shared by several rows accumulates all of their gradient.
        pool_embeddings = torch.cat(chunk_embeddings, dim=0).index_select(
            0, pool_inverse
        )
        loss, metrics = ctx.criterion(
            pool_embeddings=pool_embeddings,
            pool_idx=pool_idx,
            anchor_pos=anchor_pos,
            cal_exclude=cal_exclude,
        )
        loss = loss.float()

    ctx.encoded_texts_total = getattr(ctx, "encoded_texts_total", 0) + encoded_texts
    ctx.encoded_tokens_total = getattr(ctx, "encoded_tokens_total", 0) + encoded_tokens

    if not is_finite(loss):
        raise RuntimeError(
            f"GGPKD loss NaN/Inf at epoch={ctx.current_epoch} step={ctx.current_step}"
        )

    ctx.scaler.scale(loss).backward()
    ctx.scaler.unscale_(ctx.optimizer)
    # Reported, not enforced; the norm is also the finiteness check.
    metrics["grad_norm"] = float(total_grad_norm(ctx.optimizer))
    if not math.isfinite(metrics["grad_norm"]):
        ctx.optimizer.zero_grad(set_to_none=True)
        ctx.scaler.update()
        # Keep the schedule aligned with the step count it was built for.
        ctx.scheduler.step()
        return loss, {**metrics, "skip": "grad_inf"}

    ctx.scaler.step(ctx.optimizer)
    ctx.scaler.update()
    ctx.scheduler.step()
    return loss, metrics
