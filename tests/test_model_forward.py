import torch

from nuvo_v1.config import ModelConfig
from nuvo_v1.model import Nuvo


def test_forward_shapes():
    cfg = ModelConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        ffn_hidden=128,
        max_seq_len=32,
    )
    model = Nuvo(cfg)
    x = torch.randint(0, 100, (2, 16))
    logits, loss = model(x, targets=x)
    assert logits.shape == (2, 16, 100)
    assert loss is not None


def test_param_count():
    model = Nuvo(ModelConfig())
    n = model.num_params()
    print(f"params: {n / 1e6:.2f}M")
    assert 40e6 < n < 60e6
