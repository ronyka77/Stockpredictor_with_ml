from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class _NumpyOptionalCatDataset(Dataset):
    """
    Dataset wrapping numeric features and targets with an optional categorical index array.

    Yields (x_num, y, cat_idx) where cat_idx can be None per sample batch consumer expectations.
    """

    def __init__(
        self, *, x_num: np.ndarray, y: np.ndarray, cat_idx: Optional[np.ndarray] = None
    ) -> None:
        """Support both in-memory numpy arrays and memmapped numpy arrays.

        For very large datasets, users can pass `np.memmap` instances for
        `x_num` (and `cat_idx`) to avoid copying the whole dataset into RAM.
        """
        # Accept either float32 arrays or memmap views; ensure dtype without copying when possible
        if x_num.dtype != np.float32:
            x_num = x_num.astype(np.float32)
        self.x_num = x_num
        # y must be float32; prefer to avoid unnecessary copies if already correct dtype
        self.y = y.astype(np.float32)
        self.cat_idx = cat_idx.astype(np.int64) if cat_idx is not None else None

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        # Indexing memmaps returns numpy arrays without copying large blocks
        x_np = self.x_num[idx]
        y_np = self.y[idx]
        x = torch.from_numpy(x_np)
        y = torch.tensor(y_np)
        if self.cat_idx is not None:
            c = torch.tensor(self.cat_idx[idx])
            return x, y, c
        return x, y, None


def create_dataloader_from_numpy(
    *,
    x_num: np.ndarray,
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
    ds = _NumpyOptionalCatDataset(x_num=x_num, y=y, cat_idx=cat_idx)
    pin = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
    # Keep persistent_workers disabled when num_workers == 0 to avoid extra memory/worker overhead
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
