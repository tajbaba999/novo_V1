import torch
from torch import nn

from nuvo_v1.config import ModelConfig

from .attention import GroupedQueryAttention
from .mlp import SwiGLU
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        cache: tuple | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h, new_cache = self.attn(self.norm1(x), cos, sin, start_pos, cache)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, new_cache
