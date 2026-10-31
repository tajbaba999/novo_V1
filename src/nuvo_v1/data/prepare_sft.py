"""Prepare Stanford Alpaca for SFT with train/val/test splits.

Format:
    ### Instruction:
    {instruction}

    ### Input:
    {input}             (only if non-empty)

    ### Response:
    {output}{eos}

Splits: 90/5/5 by example count.
Loss mask: only the response tokens contribute to loss.
"""

from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from nuvo_v1.config import CONFIG


def _prefix(instruction: str, input_text: str) -> str:
    s = f"### Instruction:\n{instruction.strip()}\n"
    if input_text and input_text.strip():
        s += f"\n### Input:\n{input_text.strip()}\n"
    s += "\n### Response:\n"
    return s


def format_alpaca(
    instruction: str,
    input_text: str,
    output: str,
    eos_token: str = "</s>",
) -> str:
    return f"{_prefix(instruction, input_text)}{output.strip()}{eos_token}"


def prepare_sft(
    tokenizer_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
):
    tokenizer_path = tokenizer_path or CONFIG.data.tokenizer_dir
    out_dir = Path(out_dir or CONFIG.data.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    if hasattr(tok, "add_bos_token"):
        tok.add_bos_token = False
    eos_token = tok.eos_token or ""

    print(f"Loading {CONFIG.sft.data_source}...")
    ds = load_dataset(CONFIG.sft.data_source, split="train")
    ds = ds.filter(lambda x: bool(x["output"] and x["output"].strip()))

    seen = set()

    def is_unique(ex):
        key = (ex["instruction"], ex["input"], ex["output"])
        if key in seen:
            return False
        seen.add(key)
        return True

    ds = ds.filter(is_unique)
    ds = ds.shuffle(seed=42)
    n = len(ds)
    print(f"  {n:,} usable examples")

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    train_ds = ds.select(range(n_test + n_val, n))
    val_ds = ds.select(range(n_test, n_test + n_val))
    test_ds = ds.select(range(n_test))

    print(f"  train: {len(train_ds):,} | val: {len(val_ds):,} | test: {len(test_ds):,}")

    def tokenize_batch(batch):
        out_ids, out_labels = [], []
        for inst, inp, out in zip(
            batch["instruction"], batch["input"], batch["output"]
        ):
            prefix = _prefix(inst, inp)
            text = f"{prefix}{out.strip()}{eos_token}"
            enc = tok(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            ids = enc["input_ids"]
            offsets = enc["offset_mapping"]

            boundary = len(ids)
            for i, (start, _) in enumerate(offsets):
                if start >= len(prefix):
                    boundary = i
                    break

            labels = [-100] * boundary + list(ids[boundary:])

            L = CONFIG.model.max_seq_len
            ids = ids[:L] + [tok.eos_token_id] * max(0, L - len(ids))
            labels = labels[:L] + [-100] * max(0, L - len(labels))

            out_ids.append(ids)
            out_labels.append(labels)
        return {"input_ids": out_ids, "labels": out_labels}

    print("Tokenizing...")
    for split, split_ds in [
        ("train", train_ds),
        ("val", val_ds),
        ("test", test_ds),
    ]:
        tok_ds = split_ds.map(
            tokenize_batch,
            batched=True,
            batch_size=1000,
            remove_columns=split_ds.column_names,
            num_proc=1,
        )
        ids = np.array(tok_ds["input_ids"], dtype=np.uint16)
        labels = np.array(tok_ds["labels"], dtype=np.int32)

        keep = (labels == -100).any(axis=1) & (labels != -100).any(axis=1)
        dropped = int((~keep).sum())
        ids, labels = ids[keep], labels[keep]
        if dropped:
            print(f"  {split}: dropped {dropped} unusable examples")

        np.save(out_dir / f"sft_{split}_ids.npy", ids)
        np.save(out_dir / f"sft_{split}_labels.npy", labels)
        n_assistant = (labels >= 0).sum()
        print(
            f"  {split}: {ids.shape[0]:,} examples | "
            f"{n_assistant / 1e6:.2f}M assistant tokens"
        )


if __name__ == "__main__":
    prepare_sft()
