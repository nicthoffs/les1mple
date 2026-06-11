from __future__ import annotations

import torch
from torch.utils.data import random_split


def split_dataset(dataset, val_split: float, seed: int):
    val_len = max(1, int(len(dataset) * val_split))
    train_len = len(dataset) - val_len
    if train_len < 1:
        raise SystemExit("dataset is too small for a train/val split")
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len], generator=generator)
