"""Gradio demo for the Nuvo model."""

import json
import os

import gradio as gr
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_model
from transformers import PreTrainedTokenizerFast

from nuvo_v1.config import ModelConfig
from nuvo_v1.model import Nuvo

HF_REPO = os.environ.get("HF_REPO", "tajbaba999/nuvo-300m")

print(f"Loading {HF_REPO}...")
local_dir = snapshot_download(repo_id=HF_REPO, local_dir="/tmp/nuvo")

with open(f"{local_dir}/config.json") as f:
    raw = json.load(f)

cfg = ModelConfig(
    vocab_size=raw["vocab_size"],
    d_model=raw["d_model"],
    n_layers=raw["n_layers"],
    n_heads=raw["n_heads"],
    n_kv_heads=raw["n_kv_heads"],
    head_dim=raw["head_dim"],
    ffn_hidden=raw["ffn_hidden"],
    max_seq_len=raw["max_seq_len"],
    rope_theta=raw["rope_theta"],
    qk_norm=raw["qk_norm"],
    tie_embeddings=raw["tie_embeddings"],
)

model = Nuvo(cfg)
load_model(model, f"{local_dir}/model.safetensors")
model.eval()

tok = PreTrainedTokenizerFast.from_pretrained(local_dir)
print(f"Model loaded: {model.num_params() / 1e6:.1f}M params")


def _format_prompt(instruction: str) -> str:
    return f"### Instruction:\n{instruction.strip()}\n\n### Response:\n"


@torch.no_grad()
def chat(message: str, _history: list) -> str:
    if not message.strip():
        return "Please enter a prompt."

    prompt = _format_prompt(message)
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    out = model.generate(
        ids,
        max_new_tokens=200,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        eos_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)


demo = gr.ChatInterface(
    fn=chat,
    title="Nuvo-300M",
    description=(
        "A 300M parameter language model trained from scratch on 6B tokens, "
        "then instruction-tuned on Alpaca."
    ),
    examples=[
        "What is the capital of France?",
        "Give 3 tips for staying healthy.",
        "Explain what a transformer is in one sentence.",
    ],
)

demo.launch()
