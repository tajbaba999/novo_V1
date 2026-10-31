import torch

from nuvo_v1.config import ModelConfig
from nuvo_v1.model import Nuvo


def test_overfit_tiny():
    """The single most important test. If this fails, model is broken."""
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        ffn_hidden=128,
        max_seq_len=16,
    )
    model = Nuvo(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(0, 100, (10, 16))

    loss = None
    for _ in range(300):
        _, loss = model(x, targets=x)
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert loss is not None
    assert loss.item() < 0.5, f"failed: loss={loss.item():.4f}"
