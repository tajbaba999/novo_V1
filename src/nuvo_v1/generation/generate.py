"""Text generation with optional Alpaca chat format."""

import torch
from transformers import PreTrainedTokenizerFast

from nuvo_v1.config import CONFIG
from nuvo_v1.model import Nuvo


def format_alpaca_prompt(instruction: str, input_text: str = "") -> str:
    s = f"### Instruction:\n{instruction.strip()}\n"
    if input_text.strip():
        s += f"\n### Input:\n{input_text.strip()}\n"
    s += "\n### Response:\n"
    return s


def generate(
    model: Nuvo,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    device: str = "auto",
    chat: bool = False,
) -> str:
    g = CONFIG.generation
    max_new_tokens = max_new_tokens or g.max_new_tokens
    temperature = temperature if temperature is not None else g.temperature
    top_k = top_k if top_k is not None else g.top_k
    top_p = top_p if top_p is not None else g.top_p

    if chat:
        prompt = format_alpaca_prompt(prompt)

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
    return tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True)


if __name__ == "__main__":
    from pathlib import Path

    from nuvo_v1.training.checkpoint import load_checkpoint

    tok = PreTrainedTokenizerFast.from_pretrained(str(CONFIG.data.tokenizer_dir))

    ckpt = Path(CONFIG.sft.out_dir) / "best.pt"
    if not ckpt.exists():
        ckpt = Path(CONFIG.train.out_dir) / "final.pt"

    model = Nuvo(CONFIG.model)
    load_checkpoint(ckpt, model)
    print(f"Loaded from {ckpt}")

    prompts = [
        "What is the capital of France?",
        "Give 3 tips for staying healthy.",
        "Explain what a transformer is in one sentence.",
    ]

    for p in prompts:
        print("=" * 70)
        print(f"PROMPT: {p}")
        print("-" * 70)
        print(generate(model, tok, p, chat=True))
