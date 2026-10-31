"""Load the Mistral 7B tokenizer from HuggingFace Hub (no training needed)."""

from pathlib import Path

from transformers import AutoTokenizer

from nuvo_v1.config import CONFIG


def load_tokenizer(
    model_id: str = "mistralai/Mistral-7B-v0.1",
    out_dir: str | Path | None = None,
) -> None:
    out_dir = Path(out_dir or CONFIG.data.tokenizer_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id)

    if hasattr(tok, "add_bos_token"):
        tok.add_bos_token = False
    if hasattr(tok, "add_eos_token"):
        tok.add_eos_token = False

    assert len(tok) == CONFIG.model.vocab_size, (
        f"Vocab mismatch: {len(tok):,} vs config {CONFIG.model.vocab_size:,}"
    )

    print(f"Vocab: {len(tok):,}")
    print(f"EOS: {tok.eos_token!r} (id={tok.eos_token_id})")

    tok.save_pretrained(str(out_dir))
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    load_tokenizer()
