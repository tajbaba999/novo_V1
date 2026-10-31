import torch


def compute_rope_params(
    head_dim: int,
    theta_base: float = 10000.0,
    max_seq_len: int = 2048,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) each of shape [max_seq_len, head_dim]."""
    assert head_dim % 2 == 0, "head_dim must be even"

    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype) / head_dim))
    positions = torch.arange(max_seq_len, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)   # [seq, head_dim//2]
    angles = torch.cat([angles, angles], dim=1)                # [seq, head_dim]

    return torch.cos(angles), torch.sin(angles)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0,
) -> torch.Tensor:
    """Apply RoPE to a Q or K tensor.

    Args:
        x:        [B, n_heads, T, head_dim]
        cos, sin: [max_seq, head_dim]
        offset:   starting position (for KV-cache decoding)
    """
    B, H, T, D = x.shape
    assert D % 2 == 0, "head_dim must be even"

    x1 = x[..., : D // 2]
    x2 = x[..., D // 2 :]

    cos = cos[offset : offset + T].unsqueeze(0).unsqueeze(0)   # [1, 1, T, D]
    sin = sin[offset : offset + T].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos + rotated * sin).to(x.dtype)