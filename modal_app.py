"""Modal app: tokenize, H100 pretrain, SFT, and final test evaluation."""

import modal

app = modal.App("nuvo-v1")

volume = modal.Volume.from_name("nuvo-data", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.44",
        "datasets>=2.20",
        "tokenizers>=0.19",
        "numpy>=2.0",
        "pyyaml",
        "tqdm",
        "wandb",
        "einops",
        "huggingface_hub",
        "safetensors",
        "python-dotenv",
    )
    .add_local_dir("src", "/root/nuvo/src")
)


def _setup():
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root/nuvo/src")
    os.chdir("/root/nuvo")

    from nuvo_v1.config import CONFIG

    CONFIG.data.train_bin = Path("/root/nuvo/data/processed/train.bin")
    CONFIG.data.val_bin = Path("/root/nuvo/data/processed/val.bin")
    CONFIG.data.test_bin = Path("/root/nuvo/data/processed/test.bin")
    CONFIG.data.tokenizer_dir = Path("/root/nuvo/data/tokenizer")
    CONFIG.data.processed_dir = Path("/root/nuvo/data/processed")
    CONFIG.train.device = "cuda"
    CONFIG.train.out_dir = Path("/root/nuvo/data/checkpoints/nuvo-300m")
    CONFIG.sft.out_dir = Path("/root/nuvo/data/checkpoints/nuvo-300m-sft")
    CONFIG.sft.pretrain_ckpt = Path("/root/nuvo/data/checkpoints/nuvo-300m/final.pt")
    return CONFIG


def _setup_local_ssd(pretrain=False, sft=False, test=False):
    """Copy only the files a stage needs from the Volume to local SSD."""
    import shutil
    from pathlib import Path

    CONFIG = _setup()
    local = Path("/tmp/processed")
    local.mkdir(parents=True, exist_ok=True)

    files = []
    if pretrain:
        files += ["train.bin", "val.bin"]
    if test:
        files += ["test.bin"]
    if sft:
        files += [
            "sft_train_ids.npy",
            "sft_train_labels.npy",
            "sft_val_ids.npy",
            "sft_val_labels.npy",
            "sft_test_ids.npy",
            "sft_test_labels.npy",
        ]

    for name in files:
        dst = local / name
        if dst.exists():
            continue
        src = CONFIG.data.processed_dir / name
        if not src.exists():
            raise FileNotFoundError(
                f"missing {src}; upload the prepared data to the nuvo-data volume"
            )
        print(f"Copying {name} to local SSD...")
        shutil.copy2(src, dst)

    CONFIG.data.train_bin = local / "train.bin"
    CONFIG.data.val_bin = local / "val.bin"
    CONFIG.data.test_bin = local / "test.bin"
    CONFIG.data.processed_dir = local
    CONFIG.data.tokenizer_dir = Path("/root/nuvo/data/tokenizer")
    return CONFIG


@app.function(
    image=image,
    cpu=8,
    memory=65536,
    timeout=60 * 60 * 6,
    volumes={"/root/nuvo/data": volume},
    secrets=[hf_secret],
)
def tokenize():
    import os, sys
    sys.path.insert(0, "/root/nuvo/src")
    os.chdir("/root/nuvo")

    from nuvo_v1.config import CONFIG
    from nuvo_v1.data.tokenize import tokenize_dataset

    CONFIG.data.tokenizer_dir = "/root/nuvo/data/tokenizer"
    CONFIG.data.processed_dir = "/root/nuvo/data/processed"

    tokenize_dataset()
    volume.commit()         


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=60 * 10,
    volumes={"/root/nuvo/data": volume},
    secrets=[hf_secret],
)
def download_tokenizer():
    CONFIG = _setup()
    from nuvo_v1.data.load_tokenizer import load_tokenizer

    load_tokenizer(out_dir=CONFIG.data.tokenizer_dir)
    volume.commit()
    return "tokenizer ready"


@app.function(
    image=image,
    cpu=8,
    memory=65536,
    timeout=60 * 60,
    volumes={"/root/nuvo/data": volume},
    secrets=[hf_secret],
)
def prepare_sft_remote():
    CONFIG = _setup()
    from nuvo_v1.data.prepare_sft import prepare_sft

    prepare_sft(
        tokenizer_path=CONFIG.data.tokenizer_dir,
        out_dir=CONFIG.data.processed_dir,
    )
    volume.commit()
    return "sft data ready"


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 14,
    volumes={"/root/nuvo/data": volume},
    secrets=[hf_secret],
)
def pretrain():
    _setup_local_ssd(pretrain=True)
    from nuvo_v1.training.train import train

    result = train(
        use_wandb=True,
        run_name="nuvo-300m-pretrain",
        on_save=volume.commit,
    )
    volume.commit()
    return result


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 2,
    volumes={"/root/nuvo/data": volume},
    secrets=[hf_secret],
)
def sft():
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    CONFIG = _setup_local_ssd(sft=True)
    from nuvo_v1.model import Nuvo
    from nuvo_v1.training.checkpoint import load_checkpoint, save_checkpoint
    from nuvo_v1.training.optimizer import make_optimizer
    from nuvo_v1.training.scheduler import get_lr
    from nuvo_v1.utils.seed import set_seed

    set_seed(42)
    tr_ids = np.load("/tmp/processed/sft_train_ids.npy")
    tr_labels = np.load("/tmp/processed/sft_train_labels.npy")
    vl_ids = np.load("/tmp/processed/sft_val_ids.npy")
    vl_labels = np.load("/tmp/processed/sft_val_labels.npy")
    print(f"SFT train: {tr_ids.shape} | val: {vl_ids.shape}")

    class Ds(Dataset):
        def __init__(self, i, l):
            self.ids, self.labels = i, l

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            return {
                "input_ids": torch.from_numpy(self.ids[i].astype(np.int64)),
                "labels": torch.from_numpy(self.labels[i].astype(np.int64)),
            }

    tr_loader = DataLoader(
        Ds(tr_ids, tr_labels),
        batch_size=CONFIG.sft.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    vl_loader = DataLoader(
        Ds(vl_ids, vl_labels),
        batch_size=CONFIG.sft.batch_size,
        shuffle=False,
        num_workers=4,
    )

    model = Nuvo(CONFIG.model).cuda()
    load_checkpoint(CONFIG.sft.pretrain_ckpt, model)
    opt = make_optimizer(model, CONFIG.sft.lr, CONFIG.sft.weight_decay)

    @torch.no_grad()
    def eval_val():
        model.eval()
        total, n = 0.0, 0
        for b in vl_loader:
            x = b["input_ids"].cuda()
            y = b["labels"].cuda()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, targets=y)
            total += loss.item()
            n += 1
        model.train()
        return total / max(n, 1)

    model.train()
    step, best_val = 0, float("inf")
    max_steps = CONFIG.sft.max_steps

    for _epoch in range(CONFIG.sft.epochs):
        for batch in tr_loader:
            if step >= max_steps:
                break

            x = batch["input_ids"].cuda()
            y = batch["labels"].cuda()
            cur_lr = get_lr(step, CONFIG.sft.lr, CONFIG.sft.warmup_steps, max_steps)
            for g in opt.param_groups:
                g["lr"] = cur_lr

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, targets=y)
                loss = loss / CONFIG.sft.grad_accum
            loss.backward()

            if (step + 1) % CONFIG.sft.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.sft.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if step % CONFIG.sft.log_every == 0:
                print(
                    f"SFT step {step:5d} | "
                    f"loss {loss.item() * CONFIG.sft.grad_accum:.4f} | "
                    f"lr {cur_lr:.2e}"
                )

            if step > 0 and step % CONFIG.sft.eval_every == 0:
                v = eval_val()
                print(f"  [val] step {step} | val_loss {v:.4f}")
                if v < best_val:
                    best_val = v
                    save_checkpoint(
                        f"{CONFIG.sft.out_dir}/best.pt",
                        model,
                        opt,
                        step,
                        best_val,
                    )
                    print(f"  -> saved best (val={v:.4f})")

            step += 1

    save_checkpoint(f"{CONFIG.sft.out_dir}/final.pt", model, opt, step, best_val)
    volume.commit()
    print(f"SFT done. best_val={best_val:.4f}")

    _push_to_hub(f"{CONFIG.sft.out_dir}/best.pt")
    return {"sft_steps": step, "best_val": best_val}


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes={"/root/nuvo/data": volume},
)
def eval_final():
    CONFIG = _setup_local_ssd(test=True, sft=True)
    from nuvo_v1.eval_test import eval_test_pretrain, eval_test_sft

    print("PRETRAIN TEST")
    pt = eval_test_pretrain(f"{CONFIG.train.out_dir}/final.pt")
    print("SFT TEST")
    sft_r = eval_test_sft(f"{CONFIG.sft.out_dir}/best.pt")
    return {"pretrain": pt, "sft": sft_r}


def _push_to_hub(ckpt_path: str):
    import json
    import os
    import shutil
    import tempfile

    from huggingface_hub import HfApi
    from safetensors.torch import save_model

    from nuvo_v1.config import CONFIG
    from nuvo_v1.model import Nuvo
    from nuvo_v1.training.checkpoint import load_checkpoint

    model = Nuvo(CONFIG.model)
    load_checkpoint(ckpt_path, model)

    tmp = tempfile.mkdtemp()
    save_model(model, f"{tmp}/model.safetensors")

    with open(f"{tmp}/config.json", "w") as f:
        json.dump(
            {
                "architectures": ["Nuvo"],
                "model_type": "nuvo",
                "vocab_size": CONFIG.model.vocab_size,
                "d_model": CONFIG.model.d_model,
                "n_layers": CONFIG.model.n_layers,
                "n_heads": CONFIG.model.n_heads,
                "n_kv_heads": CONFIG.model.n_kv_heads,
                "head_dim": CONFIG.model.head_dim,
                "ffn_hidden": CONFIG.model.ffn_hidden,
                "max_seq_len": CONFIG.model.max_seq_len,
                "rope_theta": CONFIG.model.rope_theta,
                "qk_norm": CONFIG.model.qk_norm,
                "tie_embeddings": CONFIG.model.tie_embeddings,
            },
            f,
            indent=2,
        )

    for f in os.listdir("/root/nuvo/data/tokenizer"):
        shutil.copy(f"/root/nuvo/data/tokenizer/{f}", tmp)

    with open(f"{tmp}/README.md", "w") as f:
        f.write(
            "---\n"
            "language: en\n"
            "license: apache-2.0\n"
            "tags: [text-generation, small-language-model]\n"
            "---\n\n"
            "# Nuvo-300M\n\n"
            "~299M parameter model, trained from scratch on 6B tokens "
            "plus Alpaca SFT.\n\n"
            "Pretrain mix: FineWeb-Edu 50% + Cosmopedia v2 30% + "
            "Wikipedia 15% + OpenWebMath 5%.\n\n"
            "## Usage\n\n"
            "```python\n"
            'prompt = "### Instruction:\\nWhat is the capital of France?'
            '\\n\\n### Response:\\n"\n'
            "```\n"
        )

    api = HfApi()
    api.create_repo(CONFIG.hf.model_repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=tmp,
        repo_id=CONFIG.hf.model_repo_id,
        repo_type="model",
    )
    print(f"https://huggingface.co/{CONFIG.hf.model_repo_id}")


@app.local_entrypoint()
def main(stage: str = "pretrain"):
    if stage == "download_tokenizer":
        print(download_tokenizer.remote())
    elif stage == "tokenize":
        print(tokenize.remote())
    elif stage == "prepare_sft":
        print(prepare_sft_remote.remote())
    elif stage == "pretrain":
        print(pretrain.remote())
    elif stage == "sft":
        print(sft.remote())
    elif stage == "eval":
        print(eval_final.remote())
    else:
        print(
            "Usage: modal run modal_app.py --stage "
            "[download_tokenizer|tokenize|prepare_sft|pretrain|sft|eval]"
        )
