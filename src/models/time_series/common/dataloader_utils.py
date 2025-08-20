from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class _NumpyOptionalCatDataset(Dataset):
    """
    Dataset wrapping numeric features and targets with an optional categorical index array.

    Yields (x_num, y, cat_idx) where cat_idx can be None per sample batch consumer expectations.
    """

    def __init__(self, *, X_num: np.ndarray, y: np.ndarray, cat_idx: Optional[np.ndarray] = None) -> None:
        assert X_num.dtype == np.float32, "X_num must be float32"
        self.X_num = X_num
        self.y = y.astype(np.float32)
        self.cat_idx = cat_idx.astype(np.int64) if cat_idx is not None else None

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X_num[idx])
        y = torch.tensor(self.y[idx])
        if self.cat_idx is not None:
            c = torch.tensor(self.cat_idx[idx])
            return x, y, c
        return x, y, None


def create_dataloader_from_numpy(
    *,
    X_num: np.ndarray,
    y: np.ndarray,
    cat_idx: Optional[np.ndarray] = None,
    batch_size: int = 1024,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    """
    Create a DataLoader from numpy arrays with optional categorical indices.
    Arguments are designed for tabular models that accept (x_num, y, cat_idx) batches,
    where cat_idx may be None.
    """
    ds = _NumpyOptionalCatDataset(X_num=X_num.astype(np.float32), y=y, cat_idx=cat_idx)
    pin = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )


