"""Text generation from a trained checkpoint."""

import torch
from transformers import PreTrainedTokenizerFast

from nuvo_v1.config import CONFIG
from nuvo_v1.model import Nuvo


def generate(
    model: Nuvo,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    device: str = "auto",
) -> str:
    g = CONFIG.generation
    max_new_tokens = max_new_tokens or g.max_new_tokens
    temperature = temperature if temperature is not None else g.temperature
    top_k = top_k if top_k is not None else g.top_k
    top_p = top_p if top_p is not None else g.top_p

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


if __name__ == "__main__":
    from pathlib import Path

    from nuvo_v1.training.checkpoint import load_checkpoint

    tok = PreTrainedTokenizerFast.from_pretrained(str(CONFIG.data.tokenizer_dir))
    model = Nuvo(CONFIG.model)
    ckpt = Path(CONFIG.train.out_dir) / "final.pt"
    load_checkpoint(ckpt, model)

    prompts = [
        "Once upon a time",
        "The little dog",
        "Lily was sad because",
    ]
    for p in prompts:
        print("=" * 70)
        print(f"PROMPT: {p}")
        print("-" * 70)
        print(generate(model, tok, p))
