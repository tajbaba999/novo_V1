"""Grouped-Query Attention with QK-norm, RoPE, and KV-cache."""

import torch
import torch.nn.functional as F
from torch import nn

from nuvo_v1.config import ModelConfig

from .rmsnorm import RMSNorm
from .rope import apply_rope


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0, (
            "n_heads must be divisible by n_kv_heads"
        )
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim or cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.group_size = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        cache: tuple | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, cos, sin, offset=start_pos)
        k = apply_rope(k, cos, sin, offset=start_pos)

        if cache is not None:
            prev_k, prev_v = cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)

        new_cache = (k, v)

        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

        kv_len = k.shape[2]
        if T > 1 and kv_len != T:
            attn_mask = torch.ones(T, kv_len, dtype=torch.bool, device=x.device).tril(
                diagonal=kv_len - T
            )
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        out = self.wo(out)
        return out, new_cache
