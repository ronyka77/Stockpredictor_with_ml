from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.time_series.realmlp.realmlp_architecture import RealMLPModule
from src.models.time_series.common.dataloader_utils import create_dataloader_from_numpy
from src.models.time_series.realmlp.realmlp_preprocessing import RealMLPPreprocessor
from src.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class RealMLPTrainingConfig:
    epochs: int = 30
    batch_size: int = 1024
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    dropout: float = 0.1
    activation: str = "gelu"
    batch_norm: bool = True
    use_diagonal: bool = True
    use_numeric_embedding: bool = True
    numeric_embedding_dim: int = 16
    use_huber: bool = True
    huber_delta: float = 0.05
    grad_clip: float = 1.0
    use_amp: bool = True
    amp_dtype_bf16: bool = False
    num_workers: int = 2
    max_consecutive_skips: int = 10
    target_clip: Optional[Tuple[float, float]] = None


class RealMLPTrainingMixin:
    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        learning_rate = float(self.config.get("learning_rate", 1e-3))
        weight_decay = float(self.config.get("weight_decay", 1e-5))
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        epochs = int(self.config.get("epochs", 30))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    def _build_loaders(
        self,
        *,
        X_train_num: np.ndarray,
        y_train: np.ndarray,
        train_cat_idx: Optional[np.ndarray],
        X_val_num: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        val_cat_idx: Optional[np.ndarray]) -> Tuple[DataLoader, Optional[DataLoader]]:
        batch_size = int(self.config.get("batch_size", 1024))
        num_workers = int(self.config.get("num_workers", 2))
        train_loader = create_dataloader_from_numpy(
            X_num=X_train_num,
            y=y_train,
            cat_idx=train_cat_idx,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=None,
        )
        val_loader = None
        if X_val_num is not None and y_val is not None:
            val_loader = create_dataloader_from_numpy(
                X_num=X_val_num,
                y=y_val,
                cat_idx=val_cat_idx,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=None,
            )
        return train_loader, val_loader

    def _train_one_epoch(
        self,
        *,
        model: RealMLPModule,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scaler: Optional["torch.amp.GradScaler"],
        grad_clip: float) -> Tuple[float, int]:
        model.train()
        running = 0.0
        steps = 0
        skipped_consecutive = 0
        max_skips = int(self.config.get("max_consecutive_skips", 10))
        for batch_idx, batch in enumerate(loader):
            x_num, y, cat_idx = batch
            x_num = x_num.to(device)
            y = y.to(device)
            cat_idx = cat_idx.to(device) if cat_idx is not None else None

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                # Allow BF16 autocast if requested via config; otherwise default to FP16
                autocast_dtype = torch.bfloat16 if bool(self.config.get("amp_dtype_bf16", False)) else torch.float16
                with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                    preds = model(x_num, cat_idx)
                    loss = criterion(preds.squeeze(), y)
                if not torch.isfinite(loss):
                    skipped_consecutive += 1
                    # log offending batch stats
                    try:
                        x_cpu = x_num.detach().cpu().numpy()
                        y_cpu = y.detach().cpu().numpy()
                        stats = {
                            "x_min": float(np.nanmin(x_cpu)),
                            "x_max": float(np.nanmax(x_cpu)),
                            "x_mean": float(np.nanmean(x_cpu)),
                            "y_min": float(np.nanmin(y_cpu)),
                            "y_max": float(np.nanmax(y_cpu)),
                            "y_mean": float(np.nanmean(y_cpu)),
                        }
                    except Exception:
                        stats = {}
                    logger.warning(
                        f"Skipping batch {batch_idx} due to non-finite loss (AMP). stats={stats} skipped_consecutive={skipped_consecutive}/{max_skips}"
                    )
                    if skipped_consecutive >= max_skips:
                        raise RuntimeError(
                            f"Too many consecutive skipped batches due to non-finite loss (>= {max_skips}). Aborting to debug."
                        )
                    continue
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(x_num, cat_idx)
                loss = criterion(preds.squeeze(), y)
                if not torch.isfinite(loss):
                    skipped_consecutive += 1
                    try:
                        x_cpu = x_num.detach().cpu().numpy()
                        y_cpu = y.detach().cpu().numpy()
                        stats = {
                            "x_min": float(np.nanmin(x_cpu)),
                            "x_max": float(np.nanmax(x_cpu)),
                            "x_mean": float(np.nanmean(x_cpu)),
                            "y_min": float(np.nanmin(y_cpu)),
                            "y_max": float(np.nanmax(y_cpu)),
                            "y_mean": float(np.nanmean(y_cpu)),
                        }
                    except Exception:
                        stats = {}
                    logger.warning(
                        f"Skipping batch {batch_idx} due to non-finite loss. stats={stats} skipped_consecutive={skipped_consecutive}/{max_skips}"
                    )
                    if skipped_consecutive >= max_skips:
                        raise RuntimeError(
                            f"Too many consecutive skipped batches due to non-finite loss (>= {max_skips}). Aborting to debug."
                        )
                    continue
                loss.backward()
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            # successful step -> reset consecutive skip counter
            skipped_consecutive = 0
            running += float(loss.item())
            steps += 1
        avg = running / max(1, steps)
        return avg, steps

    @torch.no_grad()
    def _evaluate_mse(self, 
        model: RealMLPModule, 
        loader: Optional[DataLoader], 
        device: torch.device) -> float:
        if loader is None:
            return float("nan")
        model.eval()
        crit = nn.MSELoss()
        vals: List[float] = []
        for batch in loader:
            x_num, y, cat_idx = batch
            x_num = x_num.to(device)
            y = y.to(device)
            cat_idx = cat_idx.to(device) if cat_idx is not None else None
            preds = model(x_num, cat_idx)
            loss = crit(preds.squeeze(), y)
            if torch.isfinite(loss):
                vals.append(float(loss.item()))
        return float(np.mean(vals)) if vals else float("nan")

    def fit(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        preprocessor: RealMLPPreprocessor) -> "RealMLPTrainingMixin":
        """
        Fit the RealMLP model using the provided training data and preprocessor.
        """
        # Ensure model
        if getattr(self, "model", None) is None:
            self.model = self._create_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.preprocessor = preprocessor

        # Numeric columns order from preprocessor
        numeric_cols = list(preprocessor.feature_names)
        X_train_num, train_cat_idx = preprocessor.transform(X_train, numeric_cols=numeric_cols)
        X_val_num = y_val_array = val_cat_idx = None
        if X_val is not None and y_val is not None:
            X_val_num, val_cat_idx = preprocessor.transform(X_val, numeric_cols=numeric_cols)
            y_val_array = np.asarray(y_val.values, dtype=np.float32)

        y_train_array = np.asarray(y_train.values, dtype=np.float32)
        # Optional target clipping (robustness against extreme outliers)
        target_clip = self.config.get("target_clip")
        if target_clip is not None:
            try:
                lo, hi = target_clip
                y_train_array = np.clip(y_train_array, lo, hi)
                if y_val_array is not None:
                    y_val_array = np.clip(y_val_array, lo, hi)
                logger.info(f"Applied target clipping to range [{lo}, {hi}]")
            except Exception as e:
                logger.warning(f"Failed to apply target clipping: {e}")

        # Guard against non-finite
        train_mask = np.isfinite(X_train_num).all(axis=1) & np.isfinite(y_train_array)
        dropped_train = int(len(train_mask) - int(train_mask.sum()))
        if dropped_train:
            logger.warning(f"Dropping {dropped_train} non-finite train rows")
        X_train_num = X_train_num[train_mask]
        y_train_array = y_train_array[train_mask]
        if train_cat_idx is not None:
            train_cat_idx = train_cat_idx[train_mask]

        if X_val_num is not None:
            val_mask = np.isfinite(X_val_num).all(axis=1) & np.isfinite(y_val_array)
            dropped_val = int(len(val_mask) - int(val_mask.sum()))
            if dropped_val:
                logger.warning(f"Dropping {dropped_val} non-finite val rows")
            X_val_num = X_val_num[val_mask]
            y_val_array = y_val_array[val_mask]
            if val_cat_idx is not None:
                val_cat_idx = val_cat_idx[val_mask]

        # Loaders
        train_loader, val_loader = self._build_loaders(
            X_train_num=X_train_num,
            y_train=y_train_array,
            train_cat_idx=train_cat_idx,
            X_val_num=X_val_num,
            y_val=y_val_array,
            val_cat_idx=val_cat_idx,
        )

        # Optimization pieces
        optimizer = self._create_optimizer(self.model)
        use_huber = bool(self.config.get("use_huber", False))
        huber_delta = float(self.config.get("huber_delta", 0.1))
        criterion: nn.Module = nn.HuberLoss(delta=huber_delta) if use_huber else nn.MSELoss()
        scheduler = self._create_scheduler(optimizer)
        use_amp = bool(self.config.get("use_amp", True)) and torch.cuda.is_available()
        scaler = torch.amp.GradScaler(device=device) if use_amp else None
        epochs = int(self.config.get("epochs", 30))
        grad_clip = float(self.config.get("grad_clip", 1.0))

        logger.info(f"Starting RealMLP training for {epochs} epochs...")
        for epoch in range(epochs):
            train_loss, steps = self._train_one_epoch(
                model=self.model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scaler=scaler,
                grad_clip=grad_clip,
            )
            val_mse = self._evaluate_mse(self.model, val_loader, device)
            logger.info(f"Epoch {epoch+1}/{epochs}  train_loss={train_loss:.6f}  val_mse={val_mse:.6f}")
            if steps > 0 and scheduler is not None:
                scheduler.step()
            elif steps == 0:
                logger.warning("No optimizer steps performed; scheduler step skipped.")

        self.is_trained = True
        return self


