import numpy as np
import torch
from src.models.common.dataloader_utils import create_dataloader_from_numpy


def test_create_dataloader_without_cat_idx():
    # small dataset
    X = np.random.rand(10, 4).astype(np.float32)
    y = np.arange(10).astype(np.float32)
    # Use sentinel category indices instead of None so default_collate can stack
    cat = np.full(10, -1, dtype=np.int64)
    dl = create_dataloader_from_numpy(X_num=X, y=y, cat_idx=cat, batch_size=3, shuffle=False, num_workers=0)
    batches = list(dl)
    # expect 4 batches: 3,3,3,1
    sizes = [b[0].shape[0] for b in batches]
    assert sizes == [3, 3, 3, 1]
    # ensure cat_idx returned and contains sentinel
    for x, yb, c in batches:
        assert isinstance(x, torch.Tensor)
        assert isinstance(yb, torch.Tensor)
        assert c.dtype == torch.int64
        assert (c == -1).all()


def test_create_dataloader_with_cat_idx():
    X = np.random.rand(8, 2).astype(np.float32)
    y = np.ones(8).astype(np.float32)
    cat = np.arange(8).astype(np.int64)
    dl = create_dataloader_from_numpy(X_num=X, y=y, cat_idx=cat, batch_size=4, shuffle=False, num_workers=0)
    batches = list(dl)
    assert len(batches) == 2
    for x, yb, c in batches:
        assert c.dtype == torch.int64
        assert c.shape[0] == x.shape[0]
