import torch

from nuvo_v1.model.rope import apply_rope, compute_rope_params


def test_rope_preserves_norm():
    cos, sin = compute_rope_params(head_dim=64, max_seq_len=32)
    x = torch.randn(2, 4, 32, 64)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)


def test_rope_position_zero_identity():
    cos, sin = compute_rope_params(head_dim=64, max_seq_len=32)
    x = torch.randn(1, 1, 1, 64)
    y = apply_rope(x, cos, sin, offset=0)
    assert torch.allclose(x, y, atol=1e-5)
