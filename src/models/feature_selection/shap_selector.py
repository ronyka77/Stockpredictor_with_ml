"""SHAP-based selector: uses a LightGBM surrogate on encoded features
and maps latent-dimension importances back to original features via correlation.

Expectations:
- `X_orig_scaled` is a pandas DataFrame of original features already scaled
  (fit scaler on train and transform before calling selector).
- `X_enc_train` and `X_enc_val` are numpy arrays (train/val encodings) aligned
  with `X_orig_scaled` rows used for correlation mapping.
"""
from typing import List

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor

from src.utils.logger import get_logger

logger = get_logger(__name__)


def select_features_with_shap(
    x_orig_scaled: pd.DataFrame,
    x_enc_train: np.ndarray,
    x_enc_val: np.ndarray,
    y_train: pd.Series,
    final_k: int = 40,
    shap_sample_size: int = 100_000,
) -> List[str]:
    """Return top `final_k` original feature names selected via SHAP mapping.

    Steps:
    1. Train LightGBM on X_enc_train -> y_train
    2. Compute SHAP values on a chronological sample of X_enc_val
    3. Rank latent dims by mean(|SHAP|)
    4. Compute Pearson correlation between each latent dim and each original feature
    5. For each original feature aggregate score = sum_over_latents(|corr| * latent_shap_importance)
    6. Return top-k features by aggregated score
    """
    if x_enc_train.ndim != 2:
        raise ValueError("x_enc_train must be 2D numpy array")

    # 1) Train LightGBM surrogate
    try:
        model = LGBMRegressor(n_estimators=500, learning_rate=0.05, n_jobs=-1)
        model.fit(x_enc_train, y_train)
    except Exception as e:
        logger.error(f"LightGBM training failed: {e}")
        raise

    # 2) Prepare SHAP sample
    n_val = x_enc_val.shape[0]
    sample_n = min(shap_sample_size, n_val)
    if sample_n <= 0:
        raise ValueError("No validation rows available for SHAP computation")

    # Use the last `sample_n` rows (chronological) if dataset is time-ordered
    if sample_n == n_val:
        x_shap = x_enc_val
    else:
        x_shap = x_enc_val[-sample_n:]

    # 3) Compute SHAP values using TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_shap)
    # LightGBM regressor shap_values is a (n_rows, n_features) array for regression
    if isinstance(shap_values, list):
        shap_vals = np.array(shap_values[0])
    else:
        shap_vals = np.array(shap_values)

    # mean absolute shap per latent dim
    latent_importances = np.mean(np.abs(shap_vals), axis=0)
    # normalize
    if latent_importances.sum() > 0:
        latent_importances = latent_importances / float(latent_importances.sum())

    n_latents = x_enc_train.shape[1]
    if len(latent_importances) != n_latents:
        # Fallback: reshape if needed
        latent_importances = latent_importances.reshape(n_latents)

    # 4) Correlate latents with original features using Pearson on the aligned train rows
    # Ensure x_orig_scaled is aligned with x_enc_train rows (caller responsibility)
    x_f = x_orig_scaled.values
    z = x_enc_train  # (n_samples, n_latents)

    # compute correlation matrix (n_latents x n_features)
    # corr = (cov(Z, Xf) / (std(Z)*std(Xf)))
    # We'll compute via numpy for speed
    try:
        z_mean = z.mean(axis=0)
        z_std = z.std(axis=0)
        x_mean = x_f.mean(axis=0)
        x_std = x_f.std(axis=0)

        # avoid zero-std columns
        z_std[z_std == 0] = 1.0
        x_std[x_std == 0] = 1.0

        cov = (z.T @ x_f) / float(z.shape[0]) - np.outer(z_mean, x_mean)
        corr = cov / np.outer(z_std, x_std)
    except Exception as e:
        logger.warning(f"Correlation compute failed: {e}; falling back to numpy.corrcoef per-column")
        # fallback slower method
        corr = np.zeros((z.shape[1], x_f.shape[1]), dtype=float)
        for i in range(z.shape[1]):
            for j in range(x_f.shape[1]):
                zi = z[:, i]
                xj = x_f[:, j]
                if np.std(zi) == 0 or np.std(xj) == 0:
                    corr[i, j] = 0.0
                else:
                    corr[i, j] = np.corrcoef(zi, xj)[0, 1]

    # 5) Aggregate scores per original feature
    # score_j = sum_i ( |corr_ij| * latent_importances[i] )
    scores = np.zeros(x_f.shape[1], dtype=float)
    for i in range(corr.shape[0]):
        scores += np.abs(corr[i, :]) * float(latent_importances[i])

    # Map back to feature names
    feature_names = list(x_orig_scaled.columns)
    feature_scores = list(zip(feature_names, scores.tolist()))
    feature_scores.sort(key=lambda x: x[1], reverse=True)

    top_k = [f for f, _ in feature_scores[:final_k]]
    logger.info(f"Selected top {len(top_k)} features via SHAP mapping")
    return top_k


