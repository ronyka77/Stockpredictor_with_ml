from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

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
    num_workers: int = 1
    max_consecutive_skips: int = 10


class RealMLPTrainingMixin:
    def _calculate_gradient_norm(self, model: nn.Module) -> float:
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = float(p.grad.data.norm(2).item())
                total_norm_sq += param_norm * param_norm
        return float(total_norm_sq ** 0.5)

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
        num_workers = int(self.config.get("num_workers", 1))
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
        use_amp: bool,
        grad_clip: float,
        fallback_consecutive_nonfinite: int) -> Tuple[float, int, int, int, float, bool, Optional[Dict[str, Any]]]:
        model.train()
        running = 0.0
        steps = 0
        skipped_consecutive = 0
        skipped_batches = 0
        total_batches = 0
        grad_norm_sum = 0.0
        fallback_triggered = False
        last_nonfinite_stats: Optional[Dict[str, Any]] = None
        max_skips = int(self.config.get("max_consecutive_skips", 10))
        for batch_idx, batch in enumerate(loader):
            x_num, y, cat_idx = batch
            x_num = x_num.to(device)
            y = y.to(device)
            cat_idx = cat_idx.to(device) if cat_idx is not None else None

            optimizer.zero_grad(set_to_none=True)
            total_batches += 1
            if use_amp:
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    preds = model(x_num, cat_idx)
                    loss = criterion(preds.squeeze(), y)
                if not torch.isfinite(loss):
                    skipped_consecutive += 1
                    skipped_batches += 1
                    # log offending batch stats
                    try:
                        x_cpu = x_num.detach().cpu().numpy()
                        y_cpu = y.detach().cpu().numpy()
                        last_nonfinite_stats = {
                            "x_min": float(np.nanmin(x_cpu)),
                            "x_max": float(np.nanmax(x_cpu)),
                            "x_mean": float(np.nanmean(x_cpu)),
                            "y_min": float(np.nanmin(y_cpu)),
                            "y_max": float(np.nanmax(y_cpu)),
                            "y_mean": float(np.nanmean(y_cpu)),
                        }
                    except Exception:
                        last_nonfinite_stats = {}
                    logger.warning(
                        f"Skipping batch {batch_idx} due to non-finite loss (AMP). stats={last_nonfinite_stats} skipped_consecutive={skipped_consecutive}/{max_skips}"
                    )
                    if skipped_consecutive >= fallback_consecutive_nonfinite:
                        fallback_triggered = True
                    if skipped_consecutive >= max_skips:
                        raise RuntimeError(
                            f"Too many consecutive skipped batches due to non-finite loss (>= {max_skips}). Aborting to debug."
                        )
                    continue
                loss.backward()
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                try:
                    grad_norm_sum += self._calculate_gradient_norm(model)
                except Exception:
                    pass
                optimizer.step()
            else:
                preds = model(x_num, cat_idx)
                loss = criterion(preds.squeeze(), y)
                if not torch.isfinite(loss):
                    skipped_consecutive += 1
                    skipped_batches += 1
                    try:
                        x_cpu = x_num.detach().cpu().numpy()
                        y_cpu = y.detach().cpu().numpy()
                        last_nonfinite_stats = {
                            "x_min": float(np.nanmin(x_cpu)),
                            "x_max": float(np.nanmax(x_cpu)),
                            "x_mean": float(np.nanmean(x_cpu)),
                            "y_min": float(np.nanmin(y_cpu)),
                            "y_max": float(np.nanmax(y_cpu)),
                            "y_mean": float(np.nanmean(y_cpu)),
                        }
                    except Exception:
                        last_nonfinite_stats = {}
                    logger.warning(
                        f"Skipping batch {batch_idx} due to non-finite loss. stats={last_nonfinite_stats} skipped_consecutive={skipped_consecutive}/{max_skips}"
                    )
                    if skipped_consecutive >= max_skips:
                        raise RuntimeError(
                            f"Too many consecutive skipped batches due to non-finite loss (>= {max_skips}). Aborting to debug."
                        )
                    continue
                loss.backward()
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                try:
                    grad_norm_sum += self._calculate_gradient_norm(model)
                except Exception:
                    pass
                optimizer.step()
            # successful step -> reset consecutive skip counter
            skipped_consecutive = 0
            running += float(loss.item())
            steps += 1
        avg = running / max(1, steps)
        avg_grad_norm = grad_norm_sum / max(1, steps)
        return avg, steps, skipped_batches, total_batches, avg_grad_norm, fallback_triggered, last_nonfinite_stats

    @torch.no_grad()
    def _evaluate_metrics(self, 
        model: RealMLPModule, 
        loader: Optional[DataLoader], 
        device: torch.device) -> Dict[str, float]:
        if loader is None:
            return {"mse": float("nan"), "mae": float("nan"), "r2": float("nan")}
        model.eval()
        preds_all: List[np.ndarray] = []
        y_all: List[np.ndarray] = []
        for batch in loader:
            x_num, y, cat_idx = batch
            x_num = x_num.to(device)
            y = y.to(device)
            cat_idx = cat_idx.to(device) if cat_idx is not None else None
            preds = model(x_num, cat_idx)
            preds_all.append(preds.detach().cpu().numpy().reshape(-1))
            y_all.append(y.detach().cpu().numpy().reshape(-1))
        if not preds_all:
            return {"mse": float("nan"), "mae": float("nan"), "r2": float("nan")}
        p = np.concatenate(preds_all)
        t = np.concatenate(y_all)
        mse = float(np.mean((p - t) ** 2))
        mae = float(np.mean(np.abs(p - t)))
        t_var = float(np.var(t))
        if t_var <= 0.0:
            r2 = float("nan")
        else:
            sse = float(np.sum((p - t) ** 2))
            sst = float(len(t)) * t_var
            r2 = 1.0 - (sse / sst)
        return {"mse": mse, "mae": mae, "r2": r2}

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
        use_amp_base = bool(self.config.get("use_amp", True)) and torch.cuda.is_available()
        epochs = int(self.config.get("epochs", 30))
        grad_clip = float(self.config.get("grad_clip", 1.0))

        logger.info(f"Starting RealMLP training for {epochs} epochs...")
        # Early stopping configuration
        patience = int(self.config.get("early_stopping_patience", 10))
        min_delta = float(self.config.get("early_stopping_min_delta", 1e-4))
        warmup = int(self.config.get("early_stopping_warmup", 3))
        best_val = float("inf")
        best_epoch = -1
        epochs_no_improve = 0
        best_ckpt_path = None
        checkpoint_dir = str(self.config.get("checkpoint_dir", "checkpoints/realmlp"))

        for epoch in range(epochs):
            # Determine AMP mode this epoch (BF16 only); supports fallback to disable AMP
            use_amp = use_amp_base
            logger.info(
                f"AMP epoch config: use_amp={use_amp} dtype=bf16 loss_scaler_active=False"
            )

            fallback_consecutive_nonfinite = int(self.config.get("amp_fallback_consecutive_nonfinite", 3))

            train_loss, steps, skipped_batches, total_batches, avg_grad_norm, fallback_triggered, last_nonfinite = self._train_one_epoch(
                model=self.model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
                grad_clip=grad_clip,
                fallback_consecutive_nonfinite=fallback_consecutive_nonfinite,
            )

            # Fallback: if FP16 scaler path triggered too many non-finite, switch to BF16 or disable AMP
            if fallback_triggered and use_amp:
                logger.warning("AMP fallback triggered under BF16: disabling AMP for subsequent epochs")
                use_amp_base = False

            # Validation metrics
            val_metrics = self._evaluate_metrics(self.model, val_loader, device)
            lr_now = None
            try:
                for g in optimizer.param_groups:
                    lr_now = float(g.get("lr", None))
                    break
            except Exception:
                pass
            skipped_pct = (skipped_batches / max(1, total_batches)) * 100.0
            logger.info(
                f"Epoch {epoch+1}/{epochs} train_loss={train_loss:.6f} val_mse={val_metrics['mse']:.6f} val_mae={val_metrics['mae']:.6f} val_r2={val_metrics['r2']:.4f} "
                f"lr={lr_now} grad_norm={avg_grad_norm:.4f} skipped_batches={skipped_batches}/{total_batches} ({skipped_pct:.2f}%)"
            )
            if skipped_batches > 0 and last_nonfinite is not None:
                logger.info(f"Last non-finite batch stats: {last_nonfinite}")
            # Early stopping check
            val_mse = val_metrics["mse"]
            if np.isfinite(val_mse) and (epoch + 1) >= warmup:
                improved = (best_val - val_mse) > min_delta
                if improved:
                    best_val = float(val_mse)
                    best_epoch = epoch
                    epochs_no_improve = 0
                    # Save checkpoint
                    try:
                        import os
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        best_ckpt_path = os.path.join(checkpoint_dir, "realmlp_best.pt")
                        torch.save({
                            "model_state": self.model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "epoch": epoch + 1,
                            "val_mse": best_val,
                        }, best_ckpt_path)
                        logger.info(f"🧩 Saved best checkpoint at epoch {epoch+1} to {best_ckpt_path} (val_mse={best_val:.6f})")
                    except Exception as e:
                        logger.warning(f"Could not save checkpoint: {e}")
                else:
                    epochs_no_improve += 1
                    logger.info(f"⏳ No improvement ({epochs_no_improve}/{patience})")
                    if epochs_no_improve >= patience:
                        logger.info(f"⛔ Early stopping at epoch {epoch+1}; best epoch was {best_epoch+1} (val_mse={best_val:.6f})")
                        # Restore the best checkpoint if available
                        if best_ckpt_path is not None:
                            try:
                                state = torch.load(best_ckpt_path, map_location=device)
                                self.model.load_state_dict(state.get("model_state", {}))
                                logger.info("✅ Restored best model weights from checkpoint")
                            except Exception as e:
                                logger.warning(f"Could not restore best checkpoint: {e}")
                        break
            if steps > 0 and scheduler is not None:
                scheduler.step()
            elif steps == 0:
                logger.warning("No optimizer steps performed; scheduler step skipped.")

        # Finalize by restoring best weights if training finished without trigger
        if best_ckpt_path is not None:
            try:
                state = torch.load(best_ckpt_path, map_location=device)
                self.model.load_state_dict(state.get("model_state", {}))
                logger.info(f"✅ Finalized with best checkpoint weights (val_mse={best_val:.6f})")
            except Exception as e:
                logger.warning(f"Could not finalize restore of best checkpoint: {e}")

        self.is_trained = True
        # Optionally precompute latent stats for latent_mahalanobis confidence
        try:
            if hasattr(self, "set_latent_stats") and X_train is not None:
                # Use the same numeric feature alignment as during training
                self.set_latent_stats(X_ref=X_train)
                logger.info("✅ Precomputed latent stats for latent_mahalanobis confidence")
        except Exception as e:
            logger.warning(f"Could not precompute latent stats: {e}")
        return self


