"""Grouped-Query Attention with QK-norm, RoPE, and KV-cache."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nuvo_v1.config import ModalConfig
from .rmsnorm import RMSNorm
from .rope import apply_rope

class GropedQueryAttentation(nn.Module):
    def __init__(self, cfg: ModalConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.d_model = cfg.d_model
        self.group_size = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads* cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

        if cfg.qk_norm is None:
            self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
            self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
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

        if self.q_norm is None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, cos, sin, offset=start_pos)
        k = apply_rope(k, cos, sin, offset=start_pos)

        if cache is not None and cache[0] is not None:
            prev_k, prev_v = cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        new_cache = (k, v)

        if self.group_size > 1:
            k_rep = k.repeat_interleave(self.group_size, dim=1)
            v_rep = v.repeat_interleave(self.group_size, dim=1)
        else:
            k_rep, v_rep = k, v

        past_len = k.size(2) - T
        if past_len == 0:
            out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
        elif T == 1:
            out = F.scaled_dot_product_attention(q, k_rep, v_rep)
        else:
            mask = torch.triu(
                torch.ones(T, past_len + T, dtype=torch.bool, device=x.device),
                diagonal=past_len + 1,
            )
            out = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=mask)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.wo(out), new_cache