# Nuvo-V1

A ~299M parameter decoder-only language model trained from scratch on a 6B
token mix, then instruction-tuned on Stanford Alpaca.

## Architecture

| Component | Value |
|---|---|
| Parameters | ~298.6M (tied embeddings) |
| Layers | 24 |
| d_model | 1024 |
| Heads / KV heads | 16 / 4 (GQA) |
| Head dim | 64 |
| FFN hidden | 2752 (SwiGLU) |
| Context | 1024 |
| Positional encoding | RoPE (theta 10000) |
| Norm | RMSNorm, QK-norm |
| Tokenizer | Mistral-7B-v0.1 (32,000 vocab) |

## Data

Pretraining mix (streamed and tokenized locally or on Modal):

| Source | Tokens | Mix |
|---|---|---|
| HuggingFaceFW/fineweb-edu (sample-10BT) | 3.0B | 50% |
| HuggingFaceTB/smollm-corpus (cosmopedia-v2) | 1.8B | 30% |
| wikimedia/wikipedia (20231101.en) | 900M | 15% |
| open-web-math/open-web-math | 300M | 5% |
| **Total** | **6.0B** | **100%** |

Splits:

- Pretrain: **96 / 2 / 2** by token count (`train.bin`, `val.bin`, `test.bin`)
- SFT: **90 / 5 / 5** by example count (`sft_{train,val,test}_{ids,labels}.npy`)

Shuffling is done in 1M-token windows so document structure inside each window
is preserved. SFT labels mask the instruction prefix with `-100`, so only the
response tokens contribute to the loss.

## Setup

```bash
uv sync
cp .env.example .env   # fill in HF_TOKEN and WANDB_API_KEY
```

`.env` is gitignored and loaded automatically at runtime. Modal containers
cannot see your local `.env`, so push it once to the Modal secret store:

```bash
uv run modal secret create --force --from-dotenv .env huggingface-secret
```

## Data preparation

The tokenizer is tiny and can be downloaded locally:

```bash
uv run python -m nuvo_v1.data.load_tokenizer
```

The 6B-token corpus needs about **25 GB of free disk** while tokenizing and
**12 GB** for the final binaries. If your machine does not have that, run
tokenization on Modal instead.

### Option A: run everything on Modal (recommended)

No local tokenizer download or data upload needed:

```bash
uv run modal volume create nuvo-data
uv run modal run --detach modal_app.py --stage download_tokenizer
uv run modal run --detach modal_app.py --stage tokenize
uv run modal run --detach modal_app.py --stage prepare_sft
```

`download_tokenizer` writes the Mistral tokenizer to the volume,
`tokenize` streams the 6B-token mix and writes `train.bin`, `val.bin`, and
`test.bin`, and `prepare_sft` builds the Alpaca splits. Temporary files stay
on the container's local SSD.

### Option B: tokenize locally

```bash
uv run python -m nuvo_v1.data.load_tokenizer   # Mistral tokenizer, ~5s
uv run python -m nuvo_v1.data.tokenize         # ~3 hours, ~25 GB peak disk
uv run python -m nuvo_v1.data.prepare_sft
uv run modal volume put --force nuvo-data data/tokenizer/ /tokenizer/
uv run modal volume put --force nuvo-data data/processed/ /processed/
```

Expected outputs:

```text
data/tokenizer/tokenizer.json
data/tokenizer/tokenizer_config.json
data/processed/train.bin   # ~10.7 GiB (uint16)
data/processed/val.bin     # ~229 MiB
data/processed/test.bin    # ~229 MiB
data/processed/sft_train_ids.npy
data/processed/sft_train_labels.npy
data/processed/sft_val_ids.npy
data/processed/sft_val_labels.npy
data/processed/sft_test_ids.npy
data/processed/sft_test_labels.npy
```

The split tests skip automatically until the files above exist locally.

## Training

### Pretrain

On Modal (H100, batch 32 x grad-accum 8, ~6B tokens):

```bash
uv run modal run --detach modal_app.py --stage pretrain
```

Or locally:

```bash
uv run python -m nuvo_v1.training.train
```

The Modal stage copies `train.bin`/`val.bin` from the volume to local SSD
before training to avoid slow network reads during the run.

### SFT

```bash
uv run modal run modal_app.py --stage sft
```

SFT selects `best.pt` by validation loss and pushes it to the Hugging Face
model repo configured in `CONFIG.hf.model_repo_id`.

### Final test evaluation

Run once after all training, on the held-out test sets:

```bash
uv run modal run modal_app.py --stage eval
```

or locally:

```bash
uv run python -m nuvo_v1.eval_test pretrain
uv run python -m nuvo_v1.eval_test sft
```

The test numbers are for reporting only; do not use them for model selection.

## Generation

```bash
uv run python -m nuvo_v1.generation.generate
```

The script prefers the SFT `best.pt`, falling back to the pretrain `final.pt`,
and uses the Alpaca prompt format.

## Deployment

The Gradio demo lives in `spaces/gradio/`. It downloads the model from the Hub,
loads the tokenizer, and serves an Alpaca-formatted chat interface.

```bash
hf repo create YOUR_USERNAME/nuvo-300m-demo --type space --space_sdk gradio
hf upload YOUR_USERNAME/nuvo-300m-demo spaces/gradio/ --repo-type space
```

## Project layout

```text
src/nuvo_v1/
  config.py              # all hyperparameters and paths
  data/
    load_tokenizer.py    # download Mistral tokenizer
    tokenize.py          # streaming tokenize + 96/2/2 split
    prepare_sft.py       # Alpaca formatting, 90/5/5 split, loss masks
    dataset.py           # memmapped pretraining dataset
  model/                 # RMSNorm, RoPE, GQA, SwiGLU, block, Nuvo
  training/              # optimizer, schedule, checkpoints, loop
  generation/generate.py # sampling + Alpaca chat prompt
  eval_test.py           # one-time held-out test evaluation
modal_app.py             # tokenize, pretrain, SFT, final eval on Modal
spaces/gradio/           # Hugging Face Space
tests/                   # model, RoPE, attention, split tests
```

## Configuration

All knobs live in `src/nuvo_v1/config.py`:

- `CONFIG.model` — architecture
- `CONFIG.data` — tokenizer and binary paths
- `CONFIG.train` — pretraining schedule
- `CONFIG.sft` — instruction-tuning schedule
- `CONFIG.generation` — sampling defaults
- `CONFIG.hf` — Hub repo IDs
