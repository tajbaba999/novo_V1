import torch

from nuvo_v1.config import ModelConfig
from nuvo_v1.model.attention import GroupedQueryAttention
from nuvo_v1.model.rope import compute_rope_params


def test_causality():
    cfg = ModelConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=1,
        ffn_hidden=128,
        max_seq_len=32,
    )
    attn = GroupedQueryAttention(cfg).eval()
    cos, sin = compute_rope_params(attn.head_dim, max_seq_len=32)
    x = torch.randn(1, 16, 64)
    with torch.no_grad():
        y1, _ = attn(x, cos, sin)
        x2 = x.clone()
        x2[:, 8:] += 10.0
        y2, _ = attn(x2, cos, sin)
    assert torch.allclose(y1[:, :8], y2[:, :8], atol=1e-5)
