"""Frozen base-student corpus embeddings for student-selected kNN support.

Only graph columns come from these embeddings. The artifact gathers teacher
cosines on those columns and uses teacher-derived row temperatures, so every
target remains teacher-valued even though retrieval comes from the student.

The support is frozen at the base student, before any distillation: following
the student as it trains would make the arm a moving target rather than a
fixed method definition.
"""

import torch


@torch.no_grad()
def encode_base_student(
    model_name: str,
    tokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int = 128,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """CLS-pooled embeddings from a freshly loaded pre-distillation student.

    Takes the training run's tokenizer rather than loading one by name: the
    MiniLMv2 checkpoints ship a stub tokenizer that `AutoTokenizer` resolves
    wrongly, and the run has already resolved the right one.
    """
    from transformers import AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    outputs = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        result = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        )
        # CLS pooling, matching the student path in training and evaluation.
        outputs.append(result.last_hidden_state[:, 0, :].float().cpu())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(outputs, dim=0)
