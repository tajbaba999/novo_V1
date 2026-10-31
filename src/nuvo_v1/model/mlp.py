import torch.nn.functional as F
from torch import nn

from nuvo_v1.config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        h = cfg.ffn_hidden
        self.w1 = nn.Linear(cfg.d_model, h, bias=False)
        self.w3 = nn.Linear(cfg.d_model, h, bias=False)
        self.w2 = nn.Linear(h, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
