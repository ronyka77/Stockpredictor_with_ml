"""MLP Predictor with training, evaluation, optimization, and MLflow integration."""

import warnings
import torch

import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from sklearn.preprocessing import StandardScaler

from src.models.time_series.base_pytorch_model import PyTorchBasePredictor
from src.models.evaluation import ThresholdEvaluator
from src.utils.core.logger import get_logger, timeit

logger = get_logger(__name__)

warnings.filterwarnings(
    "ignore", message=r".*torch.distributed.reduce_op.*", category=FutureWarning
)


def load_artifact(run_id: str, artifact_path: str, loader_fn):
    """Generic artifact loader with fallback."""
    try:
        local_path = __import__("mlflow").artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/{artifact_path}"
        )
        return loader_fn(local_path)
    except Exception as e:
        logger.info(f"Artifact not found ({artifact_path}): {e}")
        return None


class MLPPredictor(PyTorchBasePredictor):
    """MLP Predictor with training, feature selection, optimization, and MLflow integration."""

    def __init__(
        self,
        model_name: str = "MLP",
        config: Optional[Dict[str, Any]] = None,
        threshold_evaluator: Optional[ThresholdEvaluator] = None,
    ):
        super().__init__(model_name, config, threshold_evaluator=threshold_evaluator)
        self.scaler = None
        self.default_confidence_method = "variance"

        if not config:
            defaults = {
                "layer_sizes": [128, 64, 32],
                "activation": "relu",
                "dropout": 0.2,
                "batch_norm": False,
                "residual": False,
                "task": "regression",
                "learning_rate": 1e-3,
                "epochs": 50,
                "batch_size": 1024,
                "optimizer": "adam",
                "weight_decay": 0.0,
                "gradient_clip": None,
                "early_stopping_patience": 10,
                "early_stopping_min_delta": 1e-4,
                "lr_scheduler": None,
                "lr_scheduler_params": {},
            }
            self.config.update(defaults)

        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.early_stopping_counter = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "gradient_norm": [],
            "epoch": [],
        }

    def select_features(
        self, X: pd.DataFrame, y: pd.Series, n_features_to_select: int = 50
    ) -> List[str]:
        """Feature selection using correlation."""
        X_clean, y_clean = self._clean_data_simple(X, y)
        selected = X_clean.corrwith(y_clean).abs().nlargest(n_features_to_select).index.tolist()
        important = [
            c for c in ("close", "date_int", "ticker_id") if c in X.columns and c not in selected
        ]
        overflow = max(0, len(selected) + len(important) - n_features_to_select)
        if overflow:
            selected = selected[:-overflow]
        selected.extend(important)
        logger.info(f"Selected {len(selected)} features, top 10: {selected[:10]}")
        return selected

    def _clean_data_simple(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Simple data cleaning for feature selection."""
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        valid_mask = y.notna() & ~np.isinf(y)
        return X_clean[valid_mask], y[valid_mask]

    @timeit
    def fit(
        self,
        train_loader,
        val_loader=None,
        scaler=None,
        feature_names=None,
        trial: Optional[Any] = None,
    ):
        """Train MLP with early stopping and optional LR scheduling."""
        self.scaler = scaler
        self.feature_names = list(feature_names) if feature_names is not None else None

        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for MLPPredictor.fit")

        self.config["use_amp"] = True
        self.config["non_blocking_transfer"] = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        if self.model is None:
            self.model = self._create_model()
            self.model.to(self.device)

        epochs = self.config.get("epochs", 50)
        early_stopping_patience = self.config.get("early_stopping_patience", 10)
        early_stopping_min_delta = self.config.get("early_stopping_min_delta", 1e-4)
        gradient_clip = self.config.get("gradient_clip", None)
        criterion = nn.MSELoss()
        optimizer = self._create_optimizer(self.model)
        self.optimizer = optimizer
        scheduler = self._create_scheduler(optimizer)

        use_amp = bool(self.config.get("use_amp", True))
        non_blocking_transfer = bool(self.config.get("non_blocking_transfer", True))
        amp_enabled = bool(use_amp and self.device.type == "cuda" and torch.cuda.is_available())
        grad_scaler = torch.amp.GradScaler(enabled=amp_enabled)

        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0

        val_loss = None
        for epoch in range(epochs):
            self.model.train()
            total_loss = torch.zeros((), device=self.device)
            total_grad_norm = 0.0
            num_batches = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device, non_blocking=non_blocking_transfer)
                batch_y = batch_y.to(self.device, non_blocking=non_blocking_transfer)
                optimizer.zero_grad(set_to_none=True)

                if amp_enabled:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(batch_X)
                        loss = criterion(outputs.view(-1), batch_y.view(-1))
                    grad_scaler.scale(loss).backward()
                    if gradient_clip:
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.view(-1), batch_y.view(-1))
                    loss.backward()
                    if gradient_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                    optimizer.step()

                total_loss = total_loss + loss.detach()
                num_batches += 1

            train_loss = (total_loss / max(1, num_batches)).item()
            if any(p.grad is not None for p in self.model.parameters()):
                with torch.no_grad():
                    grad_sq_sum = torch.zeros((), device=self.device)
                    for p in self.model.parameters():
                        if p.grad is None:
                            continue
                        grad_sq_sum = grad_sq_sum + p.grad.detach().pow(2).sum()
                    total_grad_norm = float(torch.sqrt(grad_sq_sum).item())
            avg_gradient_norm = total_grad_norm

            val_loss = None
            if val_loader is not None:
                self.model.eval()
                total_val_loss = torch.zeros((), device=self.device)
                val_batches = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device, non_blocking=non_blocking_transfer)
                        batch_y = batch_y.to(self.device, non_blocking=non_blocking_transfer)
                        if amp_enabled:
                            with torch.amp.autocast("cuda"):
                                outputs = self.model(batch_X)
                                batch_loss = criterion(outputs.view(-1), batch_y.view(-1))
                        else:
                            outputs = self.model(batch_X)
                            batch_loss = criterion(outputs.view(-1), batch_y.view(-1))

                        total_val_loss = total_val_loss + batch_loss.detach()
                        val_batches += 1
                val_loss = (total_val_loss / max(1, val_batches)).item()

                if trial is not None:
                    try:
                        import optuna
                    except Exception as e:
                        logger.info(f"Optuna pruning unavailable: {e}")
                    else:
                        trial.report(val_loss, epoch)
                        if trial.should_prune():
                            logger.info(
                                f"Optuna pruned trial at epoch {epoch + 1} with val_loss={val_loss:.6f}"
                            )
                            raise optuna.TrialPruned()

                if val_loss < self.best_val_loss - early_stopping_min_delta:
                    self.best_val_loss = val_loss
                    self.early_stopping_counter = 0
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

            current_lr = (
                self.optimizer.param_groups[0]["lr"]
                if self.optimizer
                else self.config.get("learning_rate", 1e-3)
            )
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loss is not None:
                        scheduler.step(val_loss)
                else:
                    scheduler.step()
                if self.optimizer:
                    current_lr = self.optimizer.param_groups[0]["lr"]

            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(
                val_loss if val_loss is not None else float("inf")
            )
            self.training_history["learning_rate"].append(current_lr)
            self.training_history["gradient_norm"].append(avg_gradient_norm)
            self.training_history["epoch"].append(epoch + 1)

            log_msg = f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}"
            if val_loss is not None:
                log_msg += f", Val Loss: {val_loss:.6f}"
            log_msg += f", LR: {current_lr:.2e}"
            if avg_gradient_norm > 0:
                log_msg += f", Grad Norm: {avg_gradient_norm:.4f}"
            logger.info(log_msg)

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        self.is_trained = True
        return self

    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        lr = self.config.get("learning_rate", 1e-3)
        wd = self.config.get("weight_decay", 0.0)
        opt_name = self.config.get("optimizer", "adam").lower()

        optimizers = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "rmsprop": lambda p, **kw: torch.optim.RMSprop(p, momentum=0.9, **kw),
            "sgd": lambda p, **kw: torch.optim.SGD(p, momentum=0.9, **kw),
        }
        if opt_name not in optimizers:
            raise ValueError(f"Unsupported optimizer: {opt_name}")
        return optimizers[opt_name](model.parameters(), lr=lr, weight_decay=wd)

    def _create_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        scheduler_type = self.config.get("lr_scheduler")
        if scheduler_type is None:
            return None

        params = self.config.get("lr_scheduler_params", {})
        if scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=params.get("T_max", self.config.get("epochs", 50))
            )
        if scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=params.get("factor", 0.1),
                patience=params.get("patience", 5),
                min_lr=params.get("min_lr", 1e-6),
            )
        return None

    def _create_model(self, params: Optional[Dict[str, Any]] = None) -> nn.Module:
        """Create MLPModule from config."""
        from src.models.time_series.mlp.mlp_architecture import MLPModule

        config = {**self.config, **(params or {})}
        input_size = config.get("input_size")
        if input_size is None:
            raise ValueError("Config must include 'input_size'")

        layer_sizes = config.get("layer_sizes", [128, 64, 32])
        if isinstance(layer_sizes, (tuple, str)):
            layer_sizes = (
                list(layer_sizes)
                if isinstance(layer_sizes, tuple)
                else [int(x.strip()) for x in layer_sizes.split(",")]
            )

        return MLPModule(
            input_size=input_size,
            layer_sizes=layer_sizes,
            output_size=config.get("output_size", 1),
            activation=config.get("activation", "relu"),
            dropout=config.get("dropout", 0.2),
            batch_norm=config.get("batch_norm", False),
            residual=config.get("residual", False),
            task=config.get("task", "regression"),
        )

    def _preprocess_for_prediction(
        self, X: pd.DataFrame, for_confidence: bool = False
    ) -> torch.FloatTensor:
        """Centralized preprocessing with scaler."""
        from src.models.time_series.mlp.mlp_architecture import MLPDataUtils

        cleaned_X = MLPDataUtils.validate_and_clean_data(X)

        if self.scaler is None:
            raise ValueError(
                "Scaler is not set. Fit with a train-fitted scaler or load a model that includes the saved scaler."
            )
        X_scaled, _ = MLPDataUtils.scale_data(cleaned_X, self.scaler, False)

        return torch.FloatTensor(X_scaled.values)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model must be trained before predictions")

        self.model.eval()
        X_tensor = self._preprocess_for_prediction(X, for_confidence=False).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()

        return predictions.flatten() if predictions.ndim > 1 else predictions

    def get_prediction_confidence(self, X: pd.DataFrame, method: str = "variance") -> np.ndarray:
        """Get prediction confidence scores."""
        preds = np.asarray(self.predict(X)).flatten()
        if method == "variance":
            conf = np.abs(preds)
            m = conf.max() if conf.size else 0
            return conf / m if m > 0 else conf
        if method == "simple":
            return 1 / (1 + np.exp(-np.abs(preds)))
        if method == "margin":
            return np.tanh(np.abs(preds))
        raise ValueError(f"Unsupported confidence method: {method}")

    def set_scaler(self, scaler):
        """Set feature scaler."""
        self.scaler = scaler

    def objective(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        x_val_scaled: pd.DataFrame,
        y_val: pd.Series,
        fitted_scaler=StandardScaler,
    ) -> callable:
        """Create Optuna objective using a validation holdout for tuning."""
        self.best_investment_success_rate = -np.inf
        self.best_trial_model = None
        self.best_trial_params = None
        self.best_threshold_info = None
        self.feature_names = x_train.columns

        if fitted_scaler is not None:
            self.fitted_scaler = fitted_scaler

        def objective_fn(trial):
            from src.models.time_series.mlp.mlp_architecture import MLPDataUtils

            import optuna

            # Persist trial context into Optuna sqlite via user_attrs.
            # Keep values scalar/JSON-serializable.
            self._optuna_log_dataset_sizes(trial, x_train=x_train, x_test=x_val)

            def _round_to_multiple(x: float, multiple: int = 8) -> int:
                if multiple <= 0:
                    return int(round(x))
                return int(round(x / multiple) * multiple)

            batch_size = trial.suggest_categorical("batch_size", (512, 1024, 2048, 4096, 8192))
            base_lr = trial.suggest_float("base_lr", 4e-4, 7e-3, log=True)
            lr_scale = float(np.sqrt(float(batch_size) / 4096.0))
            learning_rate = float(base_lr * lr_scale)
            trial.set_user_attr("effective_learning_rate", learning_rate)

            n_layers = trial.suggest_int("n_layers", 2, 5)
            hidden_dim = trial.suggest_categorical("hidden_dim", (128, 256, 512, 768))
            taper = trial.suggest_float("taper", 0.55, 0.9)

            layer_sizes: List[int] = []
            prev = None
            for i in range(int(n_layers)):
                raw = float(hidden_dim) * float(taper**i)
                size = max(32, _round_to_multiple(raw, 8))
                if prev is not None and size > prev:
                    size = prev
                layer_sizes.append(int(size))
                prev = int(size)

            activation = trial.suggest_categorical("activation", ("relu", "gelu"))
            batch_norm = trial.suggest_categorical("batch_norm", (True, False))
            residual = (
                trial.suggest_categorical("residual", (True, False)) if n_layers >= 3 else False
            )

            use_gradient_clip = trial.suggest_categorical("use_gradient_clip", (True, False))
            gradient_clip = (
                trial.suggest_float("gradient_clip", 0.8, 5.0, log=True)
                if use_gradient_clip
                else None
            )

            model_params = {
                "input_size": len(x_train.columns),
                "layer_sizes": layer_sizes,
                "learning_rate": learning_rate,
                "dropout": trial.suggest_float("dropout", 0.0, 0.12),
                "optimizer": "adamw",
                "weight_decay": trial.suggest_float("weight_decay", 1e-9, 1e-4, log=True),
                "activation": activation,
                "epochs": trial.suggest_int("epochs", 5, 30),
                "batch_norm": batch_norm,
                "residual": residual,
                "gradient_clip": gradient_clip,
                "early_stopping_patience": trial.suggest_int("early_stopping_patience", 2, 6),
                "early_stopping_min_delta": trial.suggest_float(
                    "early_stopping_min_delta", 1e-5, 5e-4, log=True
                ),
                "lr_scheduler": "cosine",
            }
            data_params = {
                "batch_size": batch_size,
                "num_workers": 2,
                "pin_memory": torch.cuda.is_available(),
            }

            try:
                trial_model = MLPPredictor(
                    model_name=f"mlp_trial_{trial.number}",
                    config=model_params,
                    threshold_evaluator=self.threshold_evaluator,
                )
                trial_model.disable_mlflow = True
                trial_model.model = trial_model._create_model(params=model_params)
                trial_model.model.to(trial_model.device)

                self.current_trial_scaler = getattr(self, "fitted_scaler", None) or fitted_scaler

                bs, nw, pm = (
                    data_params["batch_size"],
                    data_params["num_workers"],
                    data_params["pin_memory"],
                )
                train_loader = MLPDataUtils.create_dataloader_from_dataframe(
                    x_train, y_train, bs, True, nw, pm
                )
                val_loader = MLPDataUtils.create_dataloader_from_dataframe(
                    x_val_scaled, y_val, bs, False, nw, pm
                )

                trial_model.fit(
                    train_loader,
                    val_loader,
                    scaler=self.fitted_scaler,
                    feature_names=self.feature_names,
                    trial=trial,
                )

                prices = x_val["close"].values if "close" in x_val.columns else np.ones(len(x_val))
                thr = trial_model.optimize_prediction_threshold(
                    x_test=x_val,
                    y_test=y_val,
                    current_prices_test=prices,
                    confidence_method="variance",
                )
                best = thr["best_result"]
                optimized_profit_score = best["test_profit_per_investment"]
                threshold_info = {
                    "optimal_threshold": thr["optimal_threshold"],
                    "samples_kept_ratio": best["test_samples_ratio"],
                    "investment_success_rate": best["investment_success_rate"],
                    "test_profit_per_investment": best["test_profit_per_investment"],
                    "custom_accuracy": best["test_custom_accuracy"],
                    "total_threshold_profit": best["test_profit"],
                    "profitable_investments": best["profitable_investments"],
                }

                self._optuna_log_trial_core_metrics(
                    trial,
                    profit_per_investment=optimized_profit_score,
                    best_iteration=getattr(trial_model, "best_iteration", None),
                    best_score=getattr(trial_model, "best_score", None),
                )
                self._optuna_log_threshold_metrics(
                    trial,
                    optimal_threshold=threshold_info.get("optimal_threshold"),
                    samples_kept_ratio=threshold_info.get("samples_kept_ratio"),
                    investment_success_rate=threshold_info.get("investment_success_rate"),
                    custom_accuracy=threshold_info.get("custom_accuracy"),
                    total_profit=threshold_info.get("total_threshold_profit"),
                    profitable_investments=threshold_info.get("profitable_investments"),
                )

                if optimized_profit_score > self.best_investment_success_rate:
                    self.best_investment_success_rate = optimized_profit_score
                    self.best_trial_model = trial_model
                    self.best_trial_params = {**model_params, **data_params}
                    self.best_threshold_info = threshold_info
                    self.model = trial_model.model
                    self.feature_names = trial_model.feature_names

                    if threshold_info["optimal_threshold"] is not None:
                        self.optimal_threshold = threshold_info["optimal_threshold"]
                        self.confidence_method = "variance"

                    thr_val = threshold_info.get("optimal_threshold")
                    thr_str = (
                        f"{float(thr_val):.3f}"
                        if isinstance(thr_val, (int, float, np.floating))
                        else "NA"
                    )
                    logger.info(
                        f"NEW BEST TRIAL {trial.number}: profit={optimized_profit_score:.3f}, threshold={thr_str}"
                    )
                    self.previous_best = optimized_profit_score

                return optimized_profit_score

            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return -1e6

        return objective_fn

    def get_best_trial_info(self) -> Dict[str, Any]:
        """Get best trial info."""
        if not hasattr(self, "best_investment_success_rate"):
            return {"message": "No hyperparameter optimization run yet"}

        scaler = getattr(self, "scaler", None)
        threshold_info = getattr(self, "best_threshold_info", None)
        return {
            "best_investment_success_rate": self.best_investment_success_rate,
            "best_trial_params": self.best_trial_params,
            "has_best_model": self.best_trial_model is not None,
            "model_updated": getattr(self, "model", None) is not None,
            "scaler_info": {
                "scaler_type": "StandardScaler" if scaler else "None",
                "scaler_available": scaler is not None,
                "scaler_fitted": bool(
                    scaler and hasattr(scaler, "mean_") and hasattr(scaler, "scale_")
                ),
            },
            "threshold_optimization": (
                {
                    **threshold_info,
                    "confidence_method": getattr(self, "confidence_method", "variance"),
                }
                if threshold_info
                else None
            ),
        }

    def finalize_best_model(self) -> None:
        """Finalize best model after hyperparameter optimization."""
        best_model = getattr(self, "best_trial_model", None)
        if not best_model:
            logger.warning("No best model to finalize")
            return

        self.model = best_model.model
        self.feature_names = best_model.feature_names
        if getattr(best_model, "scaler", None) is not None:
            self.scaler = best_model.scaler

        threshold_info = getattr(self, "best_threshold_info", None) or {}
        if threshold_info.get("optimal_threshold") is not None:
            self.optimal_threshold = threshold_info["optimal_threshold"]
            self.confidence_method = getattr(self, "confidence_method", "variance")

        logger.info(f"Best model finalized: profit={self.best_investment_success_rate:.3f}")
        if getattr(self, "disable_mlflow", False):
            return

        if self.best_trial_params:
            self.log_params({f"best_{k}": v for k, v in self.best_trial_params.items()})

        scaler = getattr(self, "scaler", None)
        metrics = {
            "best_investment_success_rate": self.best_investment_success_rate,
            "hypertuning_completed": 1,
            "scaler_used": 1 if scaler else 0,
            "scaler_type": "StandardScaler" if scaler else "None",
        }
        if threshold_info.get("optimal_threshold") is not None:
            metrics.update(
                {
                    "best_optimal_threshold": threshold_info.get("optimal_threshold"),
                    "best_samples_kept_ratio": threshold_info.get("samples_kept_ratio"),
                    "best_investment_success_rate": threshold_info.get("investment_success_rate"),
                    "best_custom_accuracy": threshold_info.get("custom_accuracy"),
                    "best_total_threshold_profit": threshold_info.get("total_threshold_profit"),
                    "best_profitable_investments": threshold_info.get("profitable_investments"),
                }
            )
        self.log_metrics(metrics)

    def save_model(
        self,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        x_eval: pd.DataFrame,
        experiment_name: str = None,
        scaler=None,
    ) -> str:
        """Save trained MLP model to MLflow."""
        import mlflow
        import mlflow.pytorch

        if self.model is None:
            raise RuntimeError("No trained model to save")

        experiment_name = experiment_name or "mlp_stock_predictor"

        try:
            mlflow.end_run()
        except Exception:
            pass

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(
            run_name=f"mlp_final_{datetime.now().strftime('%Y%m%d_%H%M')}"
        ) as run:
            exclude = {"device", "objective", "verbosity", "seed", "nthread", "verbose"}
            mlflow.log_params({k: v for k, v in params.items() if k not in exclude})
            mlflow.log_metrics(metrics)

            if scaler is not None:
                temp_dir = "src/models/scalers"
                os.makedirs(temp_dir, exist_ok=True)
                scaler_path = os.path.join(temp_dir, "scaler.pkl")
                with open(scaler_path, "wb") as f:
                    pickle.dump(scaler, f)
                mlflow.log_artifact(scaler_path, artifact_path="preprocessor")
                # os.unlink(scaler_path)

            input_example = x_eval.iloc[:5].copy()
            for col in input_example.columns:
                if input_example[col].dtype.kind == "i":
                    input_example[col] = input_example[col].astype("float64")

            self.model.to(self.device)
            self.model.eval()
            with torch.no_grad():
                input_tensor = torch.as_tensor(
                    input_example.values, dtype=torch.float32, device=self.device
                )
                predictions_example = self.model(input_tensor).detach().cpu().numpy()
            signature = mlflow.models.infer_signature(input_example, predictions_example)

            mlflow.log_dict(
                {"feature_names": list(x_eval.columns)}, "preprocessor/feature_names.json"
            )

            mlflow.pytorch.log_model(
                pytorch_model=self.model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )

            logger.info(f"MLP model saved. Run ID: {run.info.run_id}")
            return run.info.run_id

    def load_model(self, run_id: str, experiment_name: str = None) -> bool:
        """Load trained MLP model from MLflow."""
        import mlflow
        import mlflow.pytorch

        mlflow.set_experiment(experiment_name or "mlp_stock_predictor")

        model_uri = f"runs:/{run_id}/model"
        self.model = mlflow.pytorch.load_model(model_uri)
        self.model.to(self.device)

        self.scaler = load_artifact(
            run_id, "preprocessor/scaler.pkl", lambda p: pickle.load(open(p, "rb"))
        )

        feature_data = load_artifact(
            run_id,
            "preprocessor/feature_names.json",
            lambda p: json.load(open(p, "r", encoding="utf-8")),
        )
        if isinstance(feature_data, dict):
            self.feature_names = feature_data.get("feature_names", [])

        if not hasattr(self, "feature_names") or not self.feature_names:
            self._extract_feature_names_from_model(model_uri)

        self.is_trained = True
        logger.info(f"Model loaded from run {run_id}")
        return True

    def _extract_feature_names_from_model(self, model_uri: str):
        """Extract feature names from model signature."""
        try:
            from mlflow.models import get_model_info

            model_info = get_model_info(model_uri)
            sig = getattr(model_info, "signature", None)
            inputs = getattr(sig, "inputs", None) if sig else None

            feature_names: List[str] = []
            if inputs is not None and hasattr(inputs, "inputs"):
                feature_names = [i.name for i in inputs.inputs if getattr(i, "name", None)]
            elif (
                inputs is not None
                and hasattr(inputs, "schema")
                and hasattr(inputs.schema, "input_names")
            ):
                feature_names = list(inputs.schema.input_names)

            if feature_names:
                self.feature_names = feature_names
        except Exception as e:
            logger.warning(f"Could not extract feature names: {e}")
