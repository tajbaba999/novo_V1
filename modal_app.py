"""Modal app for Nuvo-V1: train on GPU + auto-upload to HF Hub."""

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
    )
    .add_local_dir("src", "/root/nuvo/src")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 12,
    volumes={"/root/nuvo/data": volume},
    secrets=[hf_secret],
)
def train(max_steps: int | None = None):
    import os
    import sys

    sys.path.insert(0, "/root/nuvo/src")
    os.chdir("/root/nuvo")

    from nuvo_v1.config import CONFIG
    from nuvo_v1.training.train import train as train_fn

    CONFIG.data.train_bin = "/root/nuvo/data/processed/train.bin"
    CONFIG.data.val_bin = "/root/nuvo/data/processed/val.bin"
    CONFIG.data.tokenizer_dir = "/root/nuvo/data/tokenizer"
    CONFIG.train.out_dir = "/root/nuvo/data/checkpoints/nuvo-30m"
    CONFIG.train.device = "cuda"
    CONFIG.train.resume = True
    if max_steps is not None:
        CONFIG.train.max_steps = max_steps

    result = train_fn(use_wandb=True, run_name="nuvo-modal")

    volume.commit()

    _push_to_hub()

    return result


def _push_to_hub():
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
    ckpt = "/root/nuvo/data/checkpoints/nuvo-30m/final.pt"
    load_checkpoint(ckpt, model)

    tmp = tempfile.mkdtemp()
    save_model(model, f"{tmp}/model.safetensors")

    for f in os.listdir(CONFIG.data.tokenizer_dir):
        shutil.copy(f"{CONFIG.data.tokenizer_dir}/{f}", tmp)

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

    api = HfApi()
    api.create_repo(CONFIG.hf.model_repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=tmp,
        repo_id=CONFIG.hf.model_repo_id,
        repo_type="model",
    )
    print(f"https://huggingface.co/{CONFIG.hf.model_repo_id}")


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    volumes={"/root/nuvo/data": volume},
)
def smoke():
    import os
    import sys

    sys.path.insert(0, "/root/nuvo/src")
    os.chdir("/root/nuvo")

    from nuvo_v1.config import CONFIG
    from nuvo_v1.training.train import train as train_fn

    CONFIG.data.train_bin = "/root/nuvo/data/processed/train.bin"
    CONFIG.data.val_bin = "/root/nuvo/data/processed/val.bin"
    CONFIG.data.tokenizer_dir = "/root/nuvo/data/tokenizer"
    CONFIG.train.out_dir = "/root/nuvo/data/checkpoints/smoke"
    CONFIG.train.max_steps = 500
    CONFIG.train.device = "cuda"

    return train_fn(use_wandb=False)


@app.local_entrypoint()
def main(smoke_test: bool = False, max_steps: int = 60000):
    if smoke_test:
        print(smoke.remote())
    else:
        print(train.remote(max_steps=max_steps))
