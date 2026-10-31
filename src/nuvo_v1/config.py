from dataclasses import dataclass, field
from pathlib import Path


#Modal
@dataclass
class ModalConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    head_dim: int | None = None
    ffn_head: int = 1408
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5 
    qk_norm: bool = True
    tie_embeddings: bool = True

#  DATA
@dataclass
class DataConfig:
    dataset_name: str = "roneneldan/TinyStories"
    vocab_size: int = 32000
    num_proc: int = 8

    tokenizer_path: Path = Path("data/tokenizer")
    processed_dir: Path = Path("data/processed")
    train_bin: Path = Path("data/processed/train.bin")
    val_bin: Path = Path("data/processed/val.bin")


#  TRAINING
@dataclass
class TrainConfig:
    batch_size: int = 16
    grad_accum: int = 4
    lr: float = 6e-4
    warmup_steps: int = 500
    max_steps: int = 60000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 20
    eval_every: int = 1000
    save_every: int = 2000
    device: str = "auto"           # "auto" | "cpu" | "cuda"
    seed: int = 42
    out_dir: Path = Path("checkpoints/nuvo-30m")
    resume: bool = True            

#  GENERATION

@dataclass
class GenerationConfig:
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95

#  HUGGING FACE
@dataclass
class HFConfig:
    model_repo_id: str = "tajbaba999/nuvo-30m"
    space_repo_id: str = "tajbaba999/nuvo-30m-demo"


#  TOP-LEVEL
@dataclass
class Config:
    project_name: str = "nuvo-v1"
    model: ModalConfig = field(default_factory=ModalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    hf: HFConfig = field(default_factory=HFConfig)


CONFIG = Config() 