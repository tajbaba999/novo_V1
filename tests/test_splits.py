"""Verify train/val/test splits are disjoint and correctly sized."""

from pathlib import Path

import numpy as np
import pytest

from nuvo_v1.config import CONFIG

_REQUIRED = [
    "train.bin",
    "val.bin",
    "test.bin",
    "sft_train_ids.npy",
    "sft_train_labels.npy",
    "sft_val_ids.npy",
    "sft_val_labels.npy",
    "sft_test_ids.npy",
    "sft_test_labels.npy",
]

pytestmark = pytest.mark.skipif(
    not all((CONFIG.data.processed_dir / n).exists() for n in _REQUIRED),
    reason="run tokenize and prepare_sft first",
)


def _require(*names: str) -> None:
    missing = [
        name for name in names if not (CONFIG.data.processed_dir / name).exists()
    ]
    if missing:
        pytest.skip(f"missing prepared data: {missing}")


def test_pretrain_splits_exist():
    for name in ["train", "val", "test"]:
        p = Path(CONFIG.data.processed_dir) / f"{name}.bin"
        assert p.exists(), f"missing {p}"


def test_pretrain_splits_disjoint():
    """Sample 1000 windows from each split, verify zero overlap."""
    _require("train.bin", "val.bin", "test.bin")
    L = CONFIG.model.max_seq_len
    splits = {}
    for name in ["train", "val", "test"]:
        arr = np.memmap(
            CONFIG.data.processed_dir / f"{name}.bin",
            dtype=np.uint16,
            mode="r",
        )
        n_chunks = len(arr) // L
        idx = np.random.default_rng(0).choice(
            n_chunks, min(1000, n_chunks), replace=False
        )
        windows = {tuple(arr[i * L : (i + 1) * L].tolist()) for i in idx}
        splits[name] = windows

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = splits[a] & splits[b]
        assert not overlap, f"{len(overlap)} windows overlap between {a} and {b}"


def test_pretrain_splits_size():
    """Verify ratios are roughly 96/2/2."""
    _require("train.bin", "val.bin", "test.bin")
    sizes = {}
    for name in ["train", "val", "test"]:
        arr = np.memmap(
            CONFIG.data.processed_dir / f"{name}.bin",
            dtype=np.uint16,
            mode="r",
        )
        sizes[name] = len(arr)

    total = sum(sizes.values())
    assert 0.94 < sizes["train"] / total < 0.98, (
        f"train ratio {sizes['train'] / total:.3f}"
    )
    assert 0.01 < sizes["val"] / total < 0.04
    assert 0.01 < sizes["test"] / total < 0.04


def test_sft_splits_exist():
    for name in ["train", "val", "test"]:
        for suffix in ["ids", "labels"]:
            p = Path(CONFIG.data.processed_dir) / f"sft_{name}_{suffix}.npy"
            assert p.exists(), f"missing {p}"


def test_sft_no_leakage():
    """Verify SFT train/val/test don't share examples."""
    _require(
        "sft_train_ids.npy",
        "sft_val_ids.npy",
        "sft_test_ids.npy",
    )
    sets = {}
    for name in ["train", "val", "test"]:
        ids = np.load(CONFIG.data.processed_dir / f"sft_{name}_ids.npy")
        hashes = {hash(ids[i].tobytes()) for i in range(len(ids))}
        sets[name] = hashes

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = sets[a] & sets[b]
        assert not overlap, f"{len(overlap)} SFT examples leak between {a} and {b}"


def test_sft_loss_masking():
    """Verify assistant tokens are trained, instruction tokens are masked."""
    _require("sft_train_labels.npy")
    labels = np.load(CONFIG.data.processed_dir / "sft_train_labels.npy")
    masked = (labels == -100).any(axis=1)
    trained = (labels != -100).any(axis=1)
    assert masked.all(), "some examples have no masked prefix"
    assert trained.all(), "some examples have no trained suffix"
