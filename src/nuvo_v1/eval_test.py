"""Final test evaluation. Run ONCE after ALL training is complete.

This number goes in the model card. Do NOT use it for model selection.
"""

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from nuvo_v1.config import CONFIG
from nuvo_v1.model import Nuvo
from nuvo_v1.training.checkpoint import load_checkpoint


class NpyDataset(Dataset):
    def __init__(self, ids, labels):
        self.ids, self.labels = ids, labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {
            "input_ids": torch.from_numpy(self.ids[i].astype(np.int64)),
            "labels": torch.from_numpy(self.labels[i].astype(np.int64)),
        }


@torch.no_grad()
def eval_test_sft(ckpt: str | Path, batch_size: int = 8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Nuvo(CONFIG.model).to(device)
    load_checkpoint(ckpt, model)
    model.eval()

    ids = np.load(CONFIG.data.processed_dir / "sft_test_ids.npy")
    labels = np.load(CONFIG.data.processed_dir / "sft_test_labels.npy")
    print(f"SFT test: {ids.shape[0]:,} examples")

    loader = DataLoader(
        NpyDataset(ids, labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if device.type == "cuda" else 0,
    )

    total, n = 0.0, 0
    for b in loader:
        x = b["input_ids"].to(device)
        y = b["labels"].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            _, loss = model(x, targets=y)
        if not math.isnan(loss.item()):
            total += loss.item()
            n += 1

    loss = total / max(n, 1)
    print("-" * 50)
    print(f"  Test loss:       {loss:.4f}")
    print(f"  Test perplexity: {math.exp(min(loss, 20)):.2f}")
    print("-" * 50)
    return {"test_loss": loss, "test_ppl": math.exp(min(loss, 20))}


@torch.no_grad()
def eval_test_pretrain(ckpt: str | Path, batch_size: int = 8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Nuvo(CONFIG.model).to(device)
    load_checkpoint(ckpt, model)
    model.eval()

    data = np.memmap(CONFIG.data.test_bin, dtype=np.uint16, mode="r")
    L = CONFIG.model.max_seq_len
    n_chunks = (len(data) - 1) // L
    print(f"Pretrain test: {len(data) / 1e6:.1f}M tokens -> {n_chunks:,} chunks")

    all_ids, all_labels = [], []
    for i in range(n_chunks):
        s = i * L
        chunk = data[s : s + L + 1].astype(np.int64)
        all_ids.append(chunk[:-1])
        all_labels.append(chunk[1:])

    loader = DataLoader(
        NpyDataset(np.array(all_ids), np.array(all_labels)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if device.type == "cuda" else 0,
    )

    total, n = 0.0, 0
    for b in loader:
        x = b["input_ids"].to(device)
        y = b["labels"].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            _, loss = model(x, targets=y)
        if not math.isnan(loss.item()):
            total += loss.item()
            n += 1

    loss = total / max(n, 1)
    print("-" * 50)
    print(f"  Test loss:       {loss:.4f}")
    print(f"  Test perplexity: {math.exp(min(loss, 20)):.2f}")
    print("-" * 50)
    return {"test_loss": loss, "test_ppl": math.exp(min(loss, 20))}


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "sft"

    if which == "sft":
        eval_test_sft(CONFIG.sft.out_dir / "best.pt")
    elif which == "pretrain":
        eval_test_pretrain(CONFIG.train.out_dir / "final.pt")
    else:
        print("Usage: python -m nuvo_v1.eval_test [sft|pretrain]")
