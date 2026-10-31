import torch
import torch.nn.functional as F
from torch import nn

from nuvo_v1.config import ModelConfig

from .block import TransformerBlock
from .kv_cache import KVCache
from .rmsnorm import RMSNorm
from .rope import compute_rope_params


class Nuvo(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        head_dim = cfg.head_dim or cfg.d_model // cfg.n_heads
        cos, sin = compute_rope_params(head_dim, cfg.rope_theta, cfg.max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init)

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.embed(idx)

        for i, block in enumerate(self.blocks):
            blk_cache = cache.get(i) if cache is not None else None
            x, new_kv = block(x, self.cos, self.sin, start_pos, blk_cache)
            if cache is not None:
                cache.update(i, new_kv)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        cache = KVCache(self.cfg.n_layers)

        prompt = idx[:, -self.cfg.max_seq_len :]
        _, _ = self(prompt, cache=cache, start_pos=0)
        pos = prompt.size(1)
        idx = prompt

        for _ in range(max_new_tokens):
            if pos >= self.cfg.max_seq_len:
                break

            logits, _ = self(idx[:, -1:], cache=cache, start_pos=pos)
            pos += 1
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = probs.cumsum(dim=-1)
                mask = cumprobs - probs > top_p
                sorted_logits[mask] = -float("inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return idx
