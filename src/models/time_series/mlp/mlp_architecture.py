"""MLP Architecture: neural network, data utilities, model wrapper."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Sequence, Tuple
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class MLPModule(nn.Module):
    """Configurable MLP for stock prediction."""

    ACTIVATIONS = {
        "relu": torch.relu,
        "leaky_relu": torch.nn.functional.leaky_relu,
        "elu": torch.nn.functional.elu,
        "gelu": torch.nn.functional.gelu,
    }

    def __init__(
        self,
        input_size: int,
        layer_sizes: Sequence[int],
        output_size: int = 1,
        activation: str = "relu",
        dropout: float = 0.2,
        batch_norm: bool = False,
        residual: bool = False,
        task: str = "regression",
    ):
        super().__init__()
        if not layer_sizes:
            raise ValueError("layer_sizes must be non-empty")
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {activation}")

        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.layer_sizes = list(layer_sizes)
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.task = task

        # Print initialization parameters for debugging/visibility
        print(
            f"MLP model created: input_size={self.input_size}, output_size={self.output_size}, "
            f"activation={self.activation}, layer_sizes={self.layer_sizes}, dropout={self.dropout}, "
            f"batch_norm={self.batch_norm}, residual={self.residual}, task={self.task}"
        )

        sizes = [input_size, *self.layer_sizes]
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(sizes, sizes[1:])])
        self.batch_norms = (
            nn.ModuleList([nn.BatchNorm1d(s) for s in self.layer_sizes]) if batch_norm else None
        )
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in self.layer_sizes])

        head = nn.Linear(self.layer_sizes[-1], output_size)
        self.output_layer = nn.Sequential(head, nn.Sigmoid()) if task == "classification" else head

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            is_output_linear = False
            try:
                if m is self.output_layer:
                    is_output_linear = True
                elif isinstance(self.output_layer, nn.Sequential) and len(self.output_layer) > 0:
                    is_output_linear = m is self.output_layer[0]
            except Exception:
                is_output_linear = False

            if is_output_linear:
                nn.init.xavier_uniform_(m.weight)
            elif self.activation in ("relu", "leaky_relu"):
                nonlinearity = "leaky_relu" if self.activation == "leaky_relu" else "relu"
                a = 0.01 if self.activation == "leaky_relu" else 0.0
                nn.init.kaiming_uniform_(m.weight, a=a, nonlinearity=nonlinearity)
            else:
                nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0) if x.dim() == 1 else x
        original_input, activation_fn = x, self.ACTIVATIONS[self.activation]

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            x = activation_fn(x)
            if self.residual and i == 0 and original_input.size(-1) == x.size(-1):
                x = x + original_input
            x = self.dropouts[i](x)

        return self.output_layer(x)

    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size,
            "layer_sizes": self.layer_sizes,
            "output_size": self.output_size,
            "activation": self.activation,
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
            "residual": self.residual,
            "task": self.task,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


class MLPDataUtils:
    """Data validation, scaling, DataLoader creation."""

    @staticmethod
    def validate_and_clean_data(data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            raise ValueError("Input data is empty")
        cleaned = data.copy().replace([np.inf, -np.inf], np.nan).fillna(0)
        nan_count = int(cleaned.isnull().sum().sum())
        inf_count = int(np.isinf(cleaned.to_numpy()).sum())
        if nan_count or inf_count:
            raise ValueError(f"Validation failed: {nan_count} NaNs, {inf_count} Infs")
        return cleaned

    @staticmethod
    def scale_data(
        data: pd.DataFrame, scaler: Optional[StandardScaler] = None, fit_scaler: bool = False
    ) -> Tuple[pd.DataFrame, StandardScaler]:
        if data.empty:
            raise ValueError("Input data is empty")
        tmp = data.replace([np.inf, -np.inf], np.nan)
        if tmp.isnull().sum().sum() > 0:
            tmp = MLPDataUtils.validate_and_clean_data(data)

        if fit_scaler:
            scaler = StandardScaler()
            scaled_array = scaler.fit_transform(tmp.astype(np.float32))
        else:
            if scaler is None:
                raise ValueError("scaler required when fit_scaler=False")
            scaled_array = scaler.transform(tmp.astype(np.float32))

        return pd.DataFrame(scaled_array, columns=data.columns, index=data.index), scaler

    @staticmethod
    def create_dataloader_from_dataframe(
        X: pd.DataFrame,
        y: pd.Series,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(np.ascontiguousarray(X.to_numpy(dtype=np.float32, copy=False))),
            torch.from_numpy(np.asarray(y, dtype=np.float32)),
        )
        kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": bool(pin_memory and torch.cuda.is_available()),
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = 2
        return DataLoader(dataset, **kwargs)


class MLPModelWrapper(nn.Module):
    """Wrapper for MLP model with prediction utilities."""

    CONFIDENCE_FN = {
        "variance": lambda c: c / c.max() if c.max() > 0 else c,
        "simple": lambda c: 1 / (1 + np.exp(-c)),
        "margin": lambda c: np.tanh(c),
    }

    def __init__(
        self,
        model: MLPModule,
        scaler: Optional[StandardScaler] = None,
        feature_names: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model, self.scaler, self.config = model, scaler, config or {}
        self.feature_names = list(feature_names) if feature_names else []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("No model for prediction")
        self.model.eval()
        x_scaled, _ = MLPDataUtils.scale_data(X, self.scaler, False)
        with torch.no_grad():
            preds = (
                self.model(torch.as_tensor(x_scaled.to_numpy(), dtype=torch.float32)).cpu().numpy()
            )
        return preds.flatten() if preds.ndim > 1 else preds

    def get_prediction_confidence(self, X: pd.DataFrame, method: str = "variance") -> np.ndarray:
        if method not in self.CONFIDENCE_FN:
            raise ValueError(f"Unsupported confidence method: {method}")
        return self.CONFIDENCE_FN[method](np.abs(self.predict(X)))

    def get_architecture_info(self) -> Dict[str, Any]:
        return self.model.get_architecture_info() if self.model else {}

    def to_device(self, device: str):
        if self.model is not None:
            self.model = self.model.to(device)
        return self
