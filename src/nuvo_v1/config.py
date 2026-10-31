"""Single source of truth for all Nuvo-V1 configuration."""

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int | None = None
    ffn_hidden: int = 2752
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm: bool = True
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.d_model // self.n_heads
        assert self.d_model == self.n_heads * self.head_dim
        assert self.n_heads % self.n_kv_heads == 0


@dataclass
class DataConfig:
    dataset_name: str = "mixed"
    vocab_size: int = 32000
    num_proc: int = 8
    tokenizer_dir: Path = Path("data/tokenizer")
    processed_dir: Path = Path("data/processed")
    train_bin: Path = Path("data/processed/train.bin")
    val_bin: Path = Path("data/processed/val.bin")
    test_bin: Path = Path("data/processed/test.bin")


@dataclass
class TrainConfig:
    batch_size: int = 32
    grad_accum: int = 8
    lr: float = 4e-4
    warmup_steps: int = 1000
    max_steps: int = 22900
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 20
    eval_every: int = 2000
    save_every: int = 4000
    num_workers: int = 8
    device: str = "auto"
    seed: int = 42
    out_dir: Path = Path("checkpoints/nuvo-300m")
    resume: bool = True


@dataclass
class SFTConfig:
    data_source: str = "tatsu-lab/alpaca"
    epochs: int = 3
    batch_size: int = 16
    grad_accum: int = 2
    lr: float = 2e-5
    warmup_steps: int = 50
    max_steps: int = 1500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 20
    eval_every: int = 250
    out_dir: Path = Path("checkpoints/nuvo-300m-sft")
    pretrain_ckpt: Path = Path("checkpoints/nuvo-300m/final.pt")


@dataclass
class GenerationConfig:
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9


@dataclass
class HFConfig:
    model_repo_id: str = "tajbaba999/nuvo-300m"
    space_repo_id: str = "tajbaba999/nuvo-300m-demo"


@dataclass
class Config:
    project_name: str = "nuvo-v1"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    hf: HFConfig = field(default_factory=HFConfig)


CONFIG = Config()
