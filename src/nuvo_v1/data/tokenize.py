"""Tokenize a HF dataset into uint16 .bin shards."""

from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from nuvo_v1.config import CONFIG


def tokenize_dataset(
    dataset_name: str | None = None,
    tokenizer_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    num_proc: int | None = None,
) -> None:
    dataset_name = dataset_name or CONFIG.data.dataset_name
    tokenizer_path = tokenizer_path or CONFIG.data.tokenizer_dir
    out_dir = Path(out_dir or CONFIG.data.processed_dir)
    num_proc = num_proc or CONFIG.data.num_proc
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    if len(tok) > 65535:
        raise ValueError(f"vocab size {len(tok)} does not fit in uint16 (max 65535)")
    eos = tok.eos_token_id
    print(f"Tokenizer: vocab={len(tok):,}, eos={eos}")

    ds = load_dataset(dataset_name)
    for split_src, split_out in [("train", "train"), ("validation", "val")]:
        print(f"Tokenizing {split_src}...")
        data = ds[split_src]

        def encode(ex):
            ids = tok(ex["text"], add_special_tokens=False)["input_ids"]
            ids.append(eos)
            return {"ids": ids}

        tokenized = data.map(
            encode,
            remove_columns=data.column_names,
            num_proc=num_proc,
        )

        all_ids = []
        for row in tokenized["ids"]:
            all_ids.extend(row)
        arr = np.array(all_ids, dtype=np.uint16)

        path = out_dir / f"{split_out}.bin"
        arr.tofile(path)
        print(f"  {path}: {len(arr) / 1e6:.2f}M tokens")


if __name__ == "__main__":
    tokenize_dataset()
