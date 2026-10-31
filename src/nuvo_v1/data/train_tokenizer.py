"""Train a BPE tokenizer on the configured dataset."""

from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from nuvo_v1.config import CONFIG


def train_tokenizer(
    dataset_name: str | None = None,
    vocab_size: int | None = None,
    out_dir: str | Path | None = None,
    max_samples: int | None = None,
) -> None:
    dataset_name = dataset_name or CONFIG.data.dataset_name
    vocab_size = vocab_size or CONFIG.data.vocab_size
    out_dir = Path(out_dir or CONFIG.data.tokenizer_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {dataset_name}...")
    ds = load_dataset(dataset_name, split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    print(f"Training BPE (vocab={vocab_size}) on {len(ds):,} docs...")
    tok = Tokenizer(BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|unk|>"],
        show_progress=True,
    )
    tok.train_from_iterator(ds["text"], trainer=trainer)
    tok.save(str(out_dir / "tokenizer.json"))

    wrapped = PreTrainedTokenizerFast(
        tokenizer_file=str(out_dir / "tokenizer.json"),
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|unk|>",
    )
    wrapped.save_pretrained(str(out_dir))
    print(f"Saved tokenizer to {out_dir}/")


if __name__ == "__main__":
    train_tokenizer()
