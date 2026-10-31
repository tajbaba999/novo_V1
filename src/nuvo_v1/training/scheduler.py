"""Cosine LR schedule with linear warmup."""

import math


def get_lr(step: int, lr: float, warmup: int, max_steps: int) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))
