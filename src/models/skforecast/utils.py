from typing import Any
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def cast_numeric_columns_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns in-place to float32 to reduce memory usage.

    Non-numeric columns are left untouched.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) == 0:
        return df
    df[numeric_cols] = df[numeric_cols].astype(np.float32)
    return df


def build_last_window_from_series(y: pd.Series, lags: int) -> np.ndarray:
    """Return the last `lags` values from a pandas Series as a numpy array (float32).

    If the series is shorter than `lags` this pads the left side with nan.
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    arr = y.values.astype(np.float32)
    if len(arr) >= lags:
        return arr[-lags:]
    # pad with nans on the left
    pad = np.full((lags - len(arr),), np.nan, dtype=np.float32)
    return np.concatenate([pad, arr])


def regressor_factory(
    name: str, *, random_state: int = 42, prefer_gpu: bool = True, **kwargs
) -> Any:
    """Construct a regressor by name with sensible GPU-aware defaults.

    Falls back to `sklearn.linear_model.Ridge` if the preferred library is not available.
    Supported names (best-effort): 'lgbm', 'xgb', 'rf', 'ridge', 'svr'.
    """
    name = name.lower()
    try:
        if name == "lgbm":
            try:
                import lightgbm as _lgb

                # limit threads to reduce memory/CPU contention
                params = dict(random_state=random_state, n_jobs=4, num_threads=4)
                if prefer_gpu:
                    # keep GPU device option when requested and available
                    params.update({"device": "gpu"})
                params.update(kwargs)
                return _lgb.LGBMRegressor(**params)
            except Exception:
                logger.warning("LightGBM not available, falling back to Ridge for 'lgbm'.")

        if name == "xgb":
            try:
                from xgboost import XGBRegressor as _XGB

                # limit CPU threads
                params = dict(random_state=random_state, n_jobs=4)
                if prefer_gpu:
                    params.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"})
                params.update(kwargs)
                return _XGB(**params)
            except Exception:
                logger.warning("XGBoost not available, falling back to Ridge for 'xgb'.")

        if name == "rf":
            try:
                from sklearn.ensemble import RandomForestRegressor as _RF

                # limit parallelism to reduce RAM spikes
                params = dict(n_jobs=4, random_state=random_state)
                params.update(kwargs)
                return _RF(**params)
            except Exception:
                logger.warning("RandomForest not available, falling back to Ridge for 'rf'.")

        if name == "svr":
            try:
                from sklearn.svm import SVR as _SVR

                params = dict()
                params.update(kwargs)
                return _SVR(**params)
            except Exception:
                logger.warning("SVR not available, falling back to Ridge for 'svr'.")

    except Exception:
        logger.exception("Unexpected error while creating regressor factory for %s", name)

    # Default / fallback
    from sklearn.linear_model import Ridge as _Ridge

    params = dict(random_state=random_state)
    params.update(kwargs)
    return _Ridge(**params)


def get_standard_scaler() -> Any:
    """Return a sklearn StandardScaler instance (constructed lazily).

    The typing is Any to avoid importing sklearn in modules that may not need it.
    """
    try:
        from sklearn.preprocessing import StandardScaler

        return StandardScaler()
    except Exception:
        logger.warning("sklearn not available; returning a no-op scaler.")

        class _NoOpScaler:
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X

            def fit_transform(self, X, y=None):
                return X

            def inverse_transform(self, X):
                return X

        return _NoOpScaler()
