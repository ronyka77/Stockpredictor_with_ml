"""GPU-friendly PyTorch autoencoder and training utilities for feature compression.

Designed to train on large tabular datasets in batches with AMP and early stopping.
Exports small, well-documented functions so the orchestrator can call them.
"""

import os
import time
from typing import Tuple, Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims=(256, 128)):
        super().__init__()
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(nn.ReLU())
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(nn.ReLU())
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def _make_dataloader(
    array: np.ndarray, batch_size: int, shuffle: bool = True, num_workers: int = 4
) -> DataLoader:
    tensor = torch.from_numpy(array.astype(np.float32))
    ds = TensorDataset(tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train_autoencoder(
    X_train: np.ndarray,
    x_val: np.ndarray,
    latent_dim: int = 40,
    device: str = "cuda",
    batch_size: int = 2048,
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 5,
    hidden_dims=(512, 256),
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train autoencoder and return the encoder model plus training history.

    All arrays must be numpy float arrays already scaled by the caller.
    """
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    input_dim = X_train.shape[1]
    model = DenseAutoencoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    criterion = nn.MSELoss()

    train_loader = _make_dataloader(X_train, batch_size=batch_size, shuffle=True)
    val_loader = _make_dataloader(x_val, batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    best_epoch = -1
    hist = {"train_loss": [], "val_loss": []}

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        t0 = time.time()
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                recon = model(batch)
                loss = criterion(recon, batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu().numpy()))

        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    recon = model(batch)
                    loss = criterion(recon, batch)
                val_losses.append(float(loss.detach().cpu().numpy()))

        avg_val = float(np.mean(val_losses)) if val_losses else float("nan")
        hist["train_loss"].append(avg_train)
        hist["val_loss"].append(avg_val)

        logger.info(
            f"AE epoch {epoch}/{epochs} train_loss={avg_train:.6f} val_loss={avg_val:.6f} elapsed={time.time() - t0:.1f}s"
        )

        # early stopping
        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            logger.info("Early stopping triggered")
            break

    # restore best
    try:
        model.load_state_dict(best_state)
    except Exception:
        logger.warning("Failed to restore best state; returning last state")

    # return only encoder as the compressor
    encoder = nn.Sequential(*list(model.encoder.children()))
    encoder = encoder.to(device)

    meta = {
        "model_state": model.state_dict(),
        "history": hist,
        "config": {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "hidden_dims": hidden_dims,
            "batch_size": batch_size,
        },
    }

    return encoder, meta


def encode_df(
    encoder: nn.Module, X: np.ndarray, batch_size: int = 2048, device: str = "cuda"
) -> np.ndarray:
    """Encode a numpy array X using encoder in batches and return numpy array of latent embeddings.

    Encoder is expected to be a torch.nn.Module that maps input -> latent.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    encoder.eval()

    loader = _make_dataloader(X, batch_size=batch_size, shuffle=False)
    latents = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            z = encoder(batch)
            latents.append(z.detach().cpu().numpy())

    return np.vstack(latents)


def save_encoder(encoder: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(encoder.state_dict(), path)
