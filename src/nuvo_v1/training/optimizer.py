"""AdamW with proper weight-decay grouping (no decay on norms/embeddings)."""

import torch
from torch import nn


def make_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.1,
) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "embed" in name or "norm" in name or "bias" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
