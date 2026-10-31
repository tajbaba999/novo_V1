"""Tokenize datasets into separate .bin files, then merge them."""

from pathlib import Path
import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from nuvo_v1.config import CONFIG

MIX = [
    {"id": "HuggingFaceFW/fineweb-edu",   "config": "sample-10BT", "tokens": 3_000_000_000, "name": "fineweb"},
    {"id": "HuggingFaceTB/smollm-corpus", "config": "cosmopedia-v2", "tokens": 1_800_000_000, "name": "smollm"},
    {"id": "wikimedia/wikipedia",         "config": "20231101.en", "tokens": 900_000_000, "name": "wiki"},
    {"id": "open-web-math/open-web-math", "config": None, "tokens": 300_000_000, "name": "math"},
]

def tokenize_dataset():
    out_dir = Path(CONFIG.data.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = PreTrainedTokenizerFast.from_pretrained(str(CONFIG.data.tokenizer_dir))
    if hasattr(tok, "add_bos_token"): tok.add_bos_token = False
    eos = tok.eos_token_id

    for spec in MIX:
        out_path = out_dir / f"{spec['name']}.bin"
        
        # RESUME LOGIC: If file already exists and is large enough, skip it!
        if out_path.exists() and out_path.stat().st_size > (spec['tokens'] * 1.8):
            print(f"✓ {spec['name']}.bin already exists. Skipping.")
            continue

        print(f"\n→ Processing {spec['name']} (target: {spec['tokens']/1e6:.0f}M tokens)")
        kwargs = {"split": "train", "streaming": True}
        if spec["config"]: kwargs["name"] = spec["config"]
        
        ds = load_dataset(spec["id"], **kwargs)
        
        # Write directly to disk in chunks to save RAM
        with open(out_path, "wb") as f:
            collected, running_total = [], 0
            
            for example in ds:
                text = example.get("text", "")
                if not text or not text.strip(): continue
                ids = tok(text, add_special_tokens=False)["input_ids"]
                ids.append(eos)
                collected.append(np.array(ids, dtype=np.uint16))
                running_total += len(ids)
                
                # Flush to disk every 5M tokens
                if running_total % 5_000_000 < len(ids):
                    np.concatenate(collected).tofile(f)
                    collected = []
                    print(f"   {running_total/1e6:.1f}M tokens...")
                    
                if running_total >= spec["tokens"]:
                    break
            
            if collected:
                np.concatenate(collected).tofile(f)
                
        print(f"✓ Saved {spec['name']}.bin")

    # ── MERGE & SHUFFLE ──
    print("\nMerging and shuffling all datasets...")
    shards = [np.memmap(out_dir / f"{s['name']}.bin", dtype=np.uint16, mode="r") for s in MIX]
    combined = np.concatenate(shards)
    
    rng = np.random.default_rng(seed=42)
    rng.shuffle(combined)
    
    n_total = len(combined)
    n_test, n_val = int(n_total * 0.02), int(n_total * 0.02)
    
    combined[:n_test].tofile(out_dir / "test.bin")
    combined[n_test:n_test+n_val].tofile(out_dir / "val.bin")
    combined[n_test+n_val:].tofile(out_dir / "train.bin")
    
    print(f"✓ Final files saved! Total tokens: {n_total/1e6:.1f}M")

if __name__ == "__main__":
    tokenize_dataset()