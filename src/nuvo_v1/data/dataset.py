"""Memory-mapped dataset over token shards."""

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedDataset(Dataset):
    """Fixed-length chunks from a uint16 token stream."""

    def __init__(self, bin_path: str, seq_len: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.n = (len(self.data) - 1) // seq_len
        if self.n == 0:
            raise ValueError(f"{bin_path} too small for seq_len={seq_len}")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        s = i * self.seq_len
        chunk = self.data[s : s + self.seq_len + 1].astype(np.int64)
        return {
            "input_ids": torch.from_numpy(chunk[:-1]),
            "labels": torch.from_numpy(chunk[1:]),
        }
