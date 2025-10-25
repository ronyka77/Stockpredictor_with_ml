"""Orchestrator for feature selection pipeline.

Loads data via `prepare_common_training_data`, applies deterministic filters,
fits scaler, trains autoencoder, runs SHAP selector, persists artifacts and logs
to MLflow.
"""
import os
import time
from typing import Dict, Any

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.common.training_data_prep import prepare_common_training_data
from src.models.feature_selection.autoencoder import train_autoencoder, encode_df
from src.models.feature_selection.shap_selector import select_features_with_shap
from src.models.feature_selection.io import save_json, save_scaler, save_torch_state, log_artifacts_to_mlflow
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_feature_selection(
    prediction_horizon: int = 20,
    group_col: str = "year",
    target_dim: int = 40,
    final_k: int = 40,
    out_dir: str = "predictions",
    shap_sample_size: int = 100_000,
) -> Dict[str, Any]:
    start_time = time.time()

    data = prepare_common_training_data(prediction_horizon=prediction_horizon)
    x_train: pd.DataFrame = data["x_train"]
    y_train: pd.Series = data["y_train"]

    # Optional: drop group_col if present in features
    if group_col in x_train.columns:
        x_train = x_train.drop(columns=[group_col])

    # Basic deterministic filters: drop columns with > 0.95 missing or zero variance
    missing_frac = x_train.isna().mean()
    drop_missing = missing_frac[missing_frac > 0.95].index.tolist()
    nunique = x_train.nunique(dropna=True)
    drop_const = nunique[nunique <= 1].index.tolist()
    drop_cols = list(set(drop_missing + drop_const))
    if drop_cols:
        logger.info(f"Dropping {len(drop_cols)} columns due to missing/const: {drop_cols}")
        x_train = x_train.drop(columns=drop_cols)

    # Build small validation split from end of train (chronological)
    val_size = max(1, int(0.1 * len(x_train)))
    x_train_df = x_train.iloc[:-val_size]
    x_val_df = x_train.iloc[-val_size:]
    y_train_trim = y_train.iloc[:-val_size]

    # Fit scaler on train-only (no leakage) and transform train/val
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_df.values.astype(float))
    x_val_scaled = scaler.transform(x_val_df.values.astype(float))

    # Train autoencoder
    encoder, ae_meta = train_autoencoder(
        X_train=x_train_scaled,
        x_val=x_val_scaled,
        latent_dim=target_dim,
        device="cuda",
        batch_size=2048,
        epochs=30,
        patience=5,
    )

    # Encode train and val in batches
    x_enc_train = encode_df(encoder, x_train_scaled, batch_size=2048, device="cuda")
    x_enc_val = encode_df(encoder, x_val_scaled, batch_size=2048, device="cuda")

    # Select features via SHAP mapping
    # For correlation mapping we need original features scaled on the SAME rows used for x_enc_train
    x_train_scaled_df = pd.DataFrame(x_train_scaled, columns=x_train.columns)

    selected = select_features_with_shap(
        x_orig_scaled=x_train_scaled_df,
        x_enc_train=x_enc_train,
        x_enc_val=x_enc_val,
        y_train=y_train_trim,
        final_k=final_k,
        shap_sample_size=shap_sample_size,
    )

    ts = int(time.time())
    os.makedirs(out_dir, exist_ok=True)
    sel_path = f"{out_dir}/selected_features_{ts}.json"
    save_json({"selected_features": selected}, sel_path)

    # Save scaler and encoder state and metadata
    scaler_path = f"{out_dir}/scaler_{ts}.joblib"
    save_scaler(scaler, scaler_path)
    encoder_path = f"{out_dir}/encoder_{ts}.pt"
    save_torch_state(ae_meta.get("model_state", {}), encoder_path)
    encoder_meta_path = f"{out_dir}/encoder_meta_{ts}.json"
    save_encoder_metadata = None
    try:
        # import locally to avoid circular import at module load time
        from src.models.feature_selection.io import save_encoder_metadata

        save_encoder_metadata(ae_meta.get("config", {}), encoder_meta_path)
    except Exception:
        logger.warning("Failed to write encoder metadata")

    # Log artifacts to MLflow
    log_artifacts_to_mlflow([sel_path, scaler_path, encoder_path])

    elapsed = time.time() - start_time
    logger.info(f"Feature selection completed in {elapsed/60:.2f} minutes. Selected {len(selected)} features.")

    return {
        "selected_features": selected,
        "selected_path": sel_path,
        "scaler_path": scaler_path,
        "encoder_path": encoder_path,
        "elapsed_seconds": elapsed,
    }


def main():
    # Minimal CLI-less entrypoint for manual runs. Configure via environment
    # variables or edit the defaults below.
    prediction_horizon = int(os.environ.get("FS_PREDICTION_HORIZON", "20"))
    target_dim = int(os.environ.get("FS_TARGET_DIM", "40"))
    final_k = int(os.environ.get("FS_FINAL_K", "40"))
    out_dir = os.environ.get("FS_OUT_DIR", "predictions")
    shap_sample_size = int(os.environ.get("FS_SHAP_SAMPLE", "100000"))

    res = run_feature_selection(
        prediction_horizon=prediction_horizon,
        target_dim=target_dim,
        final_k=final_k,
        out_dir=out_dir,
        shap_sample_size=shap_sample_size,
    )
    logger.info(f"Feature selection main completed: {res}")


if __name__ == "__main__":
    main()


