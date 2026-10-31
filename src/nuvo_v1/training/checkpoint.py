"""Save / load / resume checkpoints."""

from pathlib import Path

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    best_val: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "step": step,
            "best_val": best_val,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", 0), ckpt.get("best_val", float("inf"))
