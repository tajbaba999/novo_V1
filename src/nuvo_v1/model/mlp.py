import torch.nn as nn
import torch.nn.functional as F

from nuvo_v1.config import ModalConfig

class SwiGLU(nn.Module):
    def __init__(self, cfg:ModalConfig):
        super().__init__()
        h = cfg.ffn_head
        self.w1 = nn.Linear(cfg.d_model, h, bias=False)
        self.w3 = nn.Linear(cfg.d_model, h, bias=False)
        self.w2 = nn.Linear(h, cfg.d_model, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x) * self.w3(x)))