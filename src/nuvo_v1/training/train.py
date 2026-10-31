"""Main training loop with W&B logging + checkpoint resume."""

import math
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nuvo_v1.config import CONFIG, ModelConfig, TrainConfig
from nuvo_v1.data.dataset import PackedDataset
from nuvo_v1.model import Nuvo
from nuvo_v1.training.checkpoint import load_checkpoint, save_checkpoint
from nuvo_v1.training.optimizer import make_optimizer
from nuvo_v1.training.scheduler import get_lr
from nuvo_v1.utils.seed import set_seed


def _init_wandb(enabled: bool, run_name: str | None):
    if not enabled:
        return None
    try:
        import wandb

        return wandb.init(project="nuvo-v1", name=run_name)
    except Exception as e:  # noqa: BLE001
        print(f"wandb init failed ({e}); continuing without logging")
        return None


def _wandb_log(run, data: dict, step: int) -> None:
    if run is not None:
        run.log(data, step=step)


def train(
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
    use_wandb: bool = True,
    run_name: str | None = None,
    on_save: Callable[[], None] | None = None,
) -> dict:
    model_cfg = model_cfg or CONFIG.model
    train_cfg = train_cfg or CONFIG.train

    set_seed(train_cfg.seed)

    device = train_cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Device: {device}")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    out = Path(train_cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = Nuvo(model_cfg).to(device)
    n_params = model.num_params()
    print(f"Params: {n_params / 1e6:.2f}M")

    train_ds = PackedDataset(str(CONFIG.data.train_bin), model_cfg.max_seq_len)
    val_ds = PackedDataset(str(CONFIG.data.val_bin), model_cfg.max_seq_len)
    print(f"Train chunks: {len(train_ds):,}, Val chunks: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        drop_last=True,
    )

    opt = make_optimizer(model, train_cfg.lr, train_cfg.weight_decay)

    step, best_val = 0, float("inf")
    resume_path = out / "last.pt"
    if train_cfg.resume and resume_path.exists():
        step, best_val = load_checkpoint(resume_path, model, opt)
        print(f"Resumed from {resume_path} at step {step}")

    wandb_run = _init_wandb(use_wandb, run_name)
    if wandb_run is not None:
        import wandb

        wandb.config.update(
            {
                "model": model_cfg.__dict__,
                "train": {
                    k: v
                    for k, v in train_cfg.__dict__.items()
                    if not isinstance(v, Path)
                },
                "n_params": n_params,
            }
        )

    model.train()
    t0 = time.time()
    tokens_seen = 0
    step_loss = 0.0
    log_steps = 0
    accum_count = 0
    losses = []

    pbar = tqdm(total=train_cfg.max_steps, initial=step, desc="train")
    while step < train_cfg.max_steps:
        for batch in train_loader:
            if step >= train_cfg.max_steps:
                break

            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            cur_lr = get_lr(
                step, train_cfg.lr, train_cfg.warmup_steps, train_cfg.max_steps
            )
            for g in opt.param_groups:
                g["lr"] = cur_lr

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                _, loss = model(x, y)
                loss = loss / train_cfg.grad_accum

            loss.backward()
            step_loss += loss.item() * train_cfg.grad_accum
            tokens_seen += x.numel()
            accum_count += 1
            log_steps += 1

            if accum_count < train_cfg.grad_accum:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg.grad_clip
            ).item()
            opt.step()
            opt.zero_grad(set_to_none=True)
            accum_count = 0
            step += 1
            pbar.update(1)

            if step % train_cfg.log_every == 0:
                dt = time.time() - t0
                tok_s = tokens_seen / dt if dt > 0 else 0
                avg_loss = step_loss / log_steps
                losses.append(avg_loss)

                print(
                    f"step {step:6d} | loss {avg_loss:.4f} "
                    f"| lr {cur_lr:.2e} | {tok_s:.0f} tok/s | grad {grad_norm:.3f}"
                )
                _wandb_log(
                    wandb_run,
                    {
                        "train/loss": avg_loss,
                        "train/lr": cur_lr,
                        "train/tok_per_sec": tok_s,
                        "train/grad_norm": grad_norm,
                        "train/perplexity": math.exp(min(avg_loss, 20)),
                    },
                    step,
                )

                step_loss = 0.0
                log_steps = 0
                t0 = time.time()
                tokens_seen = 0

            if step % train_cfg.eval_every == 0:
                val_loss = evaluate(model, val_loader, device, use_bf16)
                ppl = math.exp(min(val_loss, 20))
                print(f"  [eval] step {step} | val_loss {val_loss:.4f} | ppl {ppl:.2f}")
                _wandb_log(
                    wandb_run,
                    {"val/loss": val_loss, "val/perplexity": ppl},
                    step,
                )

                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(out / "best.pt", model, opt, step, best_val)
                    if on_save is not None:
                        on_save()
                    print("  -> saved best")

            if step % train_cfg.save_every == 0:
                save_checkpoint(out / "last.pt", model, opt, step, best_val)
                if on_save is not None:
                    on_save()

    pbar.close()
    save_checkpoint(out / "final.pt", model, opt, step, best_val)
    if on_save is not None:
        on_save()
    print(f"Done. best val_loss={best_val:.4f}")

    if wandb_run is not None:
        wandb_run.finish()

    return {
        "steps": step,
        "best_val": best_val,
        "final_loss": losses[-1] if losses else None,
    }


@torch.no_grad()
def evaluate(
    model: Nuvo,
    loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            _, loss = model(x, y)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


if __name__ == "__main__":
    train()
