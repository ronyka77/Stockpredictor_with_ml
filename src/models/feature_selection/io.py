"""I/O helpers for feature-selection pipeline: JSON save/load, scaler, encoder persistence and MLflow logging."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import torch

from src.utils.mlops.mlflow_integration import MLflowIntegration
from src.utils.core.logger import get_logger
from src.models.feature_selection.autoencoder import DenseAutoencoder

logger = get_logger(__name__)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_scaler(scaler: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)


def save_torch_state(state_dict: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)


def save_encoder_metadata(config: dict, path: str) -> None:
    # ensure lists serializable
    cfg = dict(config)
    if "hidden_dims" in cfg and not isinstance(cfg["hidden_dims"], list):
        cfg["hidden_dims"] = list(cfg["hidden_dims"])
    save_json(cfg, path)


def load_encoder(state_path: str, meta_path: str, device: str = "cpu") -> torch.nn.Module:
    """Reconstruct encoder model from saved state dict and meta JSON.

    Returns the encoder module (torch.nn.Module) moved to `device`.
    """
    meta = load_json(meta_path)
    input_dim = int(meta["input_dim"])
    latent_dim = int(meta["latent_dim"])
    hidden_dims = tuple(meta.get("hidden_dims", []))

    model = DenseAutoencoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)
    state = torch.load(state_path, map_location=device)
    # state may be full model.state_dict(); load into model
    try:
        model.load_state_dict(state)
    except Exception:
        # if state contains only encoder state, try loading keys that match
        model_state = {k: v for k, v in state.items() if k in model.state_dict()}
        model.load_state_dict(model_state, strict=False)

    # return encoder part
    encoder = torch.nn.Sequential(*list(model.encoder.children()))
    return encoder.to(device)


def log_artifacts_to_mlflow(local_paths: list[str], run_name: str = "feature-selection") -> None:
    import mlflow

    mlflow_int = MLflowIntegration()
    _ = mlflow_int.setup_experiment("feature-selection")
    _run = mlflow_int.start_run(run_name=run_name)
    try:
        for local in local_paths:
            if Path(local).exists():
                mlflow_int.log_artifact(local, artifact_path="feature_selection_artifacts")
            else:
                logger.warning(f"Artifact not found: {local}")
    finally:
        try:
            mlflow.end_run()
        except Exception:
            logger.warning("Failed to end MLflow run cleanly")
