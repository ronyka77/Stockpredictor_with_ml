from typing import Any, Dict, Iterable, List, Optional, Tuple
import os
import json
import logging
import tempfile
import shutil
import psutil
import gc
from numpy.lib.format import open_memmap
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from skforecast.direct._forecaster_direct import ForecasterDirect

from src.utils.mlops.mlflow_utils import MLFlowManager

from src.models.skforecast.utils import (
    build_last_window_from_series,
    get_standard_scaler,
    regressor_factory,
)

logger = logging.getLogger(__name__)

# Manifest filename constant for reuse
MANIFEST_FILENAME = "manifest.json"


class EnsembleForecaster:
    """Heterogeneous ensemble of skforecast ForecasterAutoregDirect models with per-step stacking.

    This class keeps a dict of base forecasters and a list of per-step meta models (Ridge/Lasso).
    """

    def __init__(
        self,
        lags: int = 30,
        steps: int = 20,
        base_regressors: Optional[Iterable[str]] = None,
        models_dir: str = "models/skforecast",
        persist_after_fit: bool = False,
    ) -> None:
        self.lags = int(lags)
        self.steps = int(steps)
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

        if base_regressors is None:
            base_regressors = ["lgbm", "xgb", "rf", "ridge", "svr"]

        self.base_names = list(base_regressors)
        self.forecasters: Dict[str, ForecasterDirect] = {}
        for name in self.base_names:
            reg = regressor_factory(name)
            # ForecasterDirect constructor expects (regressor, steps, lags=...)
            self.forecasters[name] = ForecasterDirect(regressor=reg, steps=self.steps, lags=self.lags)

        # Optional persistence behavior: if True we will persist base forecasters to disk
        # after fitting and replace the in-memory object with None to free RAM. The
        # persisted paths are kept in `_forecaster_paths` for on-demand loading.
        self.persist_after_fit = bool(persist_after_fit)
        self._forecaster_paths: Dict[str, str] = {}

        # per-step meta models
        self.meta_models: List[Optional[Any]] = [None] * self.steps
        self.scaler = get_standard_scaler()

    def fit(
        self,
        y_train: pd.Series,
        exog_train: Optional[pd.DataFrame] = None,
        validation: Optional[Tuple[pd.Series, Optional[pd.DataFrame]]] = None,
        save_artifacts: bool = True,
    ) -> None:
        """Fit all base forecasters and (optionally) the per-step meta learners.

        validation: a tuple (y_val, exog_val) used to build meta-features. If not provided,
        stacking will be skipped and predictions will default to a simple mean ensemble.
        """
        # Fit base forecasters
        # Iterate over a static list of keys so we can optionally mutate the dict
        for name in list(self.forecasters.keys()):
            logger.info("Fitting base forecaster: %s", name)
            forecaster = self.forecasters[name]
            forecaster.fit(y=y_train, exog=exog_train)
            # Optionally persist and remove from memory to reduce peak RAM
            if self.persist_after_fit:
                forecasters_dir = os.path.join(self.models_dir, "forecasters")
                os.makedirs(forecasters_dir, exist_ok=True)
                fname = f"forecaster_{name}.joblib"
                path = os.path.join(forecasters_dir, fname)
                try:
                    joblib.dump(forecaster, path, compress=3)
                    self._forecaster_paths[name] = path
                    # replace in-memory object with None to free memory
                    self.forecasters[name] = None
                    del forecaster
                    gc.collect()
                except Exception:
                    logger.exception("Failed to persist forecaster %s; keeping in memory.", name)

        # Fit per-step meta models if validation provided
        if validation is not None:
            y_val, exog_val = validation
            meta_datasets = self._generate_meta_datasets(
                y_train=y_train, y_val=y_val, _exog_val=exog_val
            )
            from sklearn.linear_model import Ridge

            for h in range(1, self.steps + 1):
                x_h, y_h = meta_datasets[h]
                if x_h.shape[0] == 0:
                    logger.warning("No meta-training rows for step %d — skipping meta model." % h)
                    self.meta_models[h - 1] = None
                    continue
                model = Ridge()
                model.fit(x_h, y_h)
                self.meta_models[h - 1] = model

        if save_artifacts:
            self.save(os.path.join(self.models_dir, "ensemble_manifest.json"))

    def _generate_meta_datasets(
        self, *, y_train: pd.Series, y_val: pd.Series, _exog_val: Optional[pd.DataFrame] = None
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Generate per-step meta datasets using walk-forward predictions on the validation series.

        For each validation time index i we form a last_window combining the tail of y_train and already-seen y_val up to i,
        compute each base model's prediction for horizons 1..steps and align them to the true targets.
        Returns a dict mapping step h -> (X_h, y_h) where X_h shape=(n_rows, n_base_models).
        """
        # Streaming implementation: pre-allocate one memmap per horizon and write
        # rows aligned by the target index (target_idx = i + h - 1). This avoids
        # building large in-memory lists/vstack operations.
        n_val = len(y_val)
        n_base = len(self.base_names)

        proc = psutil.Process()
        logger.info(
            "meta generation start n_val=%d bases=%d steps=%d rss=%d",
            n_val,
            n_base,
            self.steps,
            proc.memory_info().rss,
        )

        tmpdir = tempfile.mkdtemp(prefix="skf_meta_")
        memmaps_X: Dict[int, Any] = {}
        memmaps_y: Dict[int, Any] = {}
        try:
            # allocate memmaps for each horizon
            for h in range(1, self.steps + 1):
                path_x = os.path.join(tmpdir, f"meta_X_h{h}.npy")
                path_y = os.path.join(tmpdir, f"meta_y_h{h}.npy")
                memmaps_X[h] = open_memmap(path_x, mode="w+", dtype=np.float32, shape=(n_val, n_base))
                memmaps_y[h] = open_memmap(path_y, mode="w+", dtype=np.float32, shape=(n_val,))
                # initialize with nans
                memmaps_X[h][:, :] = np.nan
                memmaps_y[h][:] = np.nan

            # Precompute last windows for each validation index to avoid repeated concat
            last_windows: List[np.ndarray] = []
            for i in range(n_val):
                history = pd.concat([y_train, y_val.iloc[:i]])
                last_windows.append(build_last_window_from_series(history, self.lags))

            sample_horizons = [1, max(1, self.steps // 2), self.steps]

            # Fill memmaps by processing one base forecaster at a time to limit peak memory
            for j, name in enumerate(self.base_names):
                logger.info("processing base forecaster for meta-features: %s", name)
                forec = self.forecasters.get(name)
                loaded_from_disk = False
                if forec is None:
                    # try to load persisted forecaster if available
                    path = self._forecaster_paths.get(name)
                    if path is not None and os.path.exists(path):
                        try:
                            forec = joblib.load(path)
                            loaded_from_disk = True
                        except Exception:
                            logger.exception("Failed to load persisted forecaster %s; filling with nan", name)
                            forec = None

                for i in range(n_val):
                    try:
                        if forec is None:
                            preds = np.full((self.steps,), np.nan, dtype=np.float32)
                        else:
                            preds = np.asarray(
                                forec.predict(steps=self.steps, last_window=last_windows[i]),
                                dtype=np.float32,
                            )
                    except Exception:
                        preds = np.full((self.steps,), np.nan, dtype=np.float32)

                    # write aligned rows into per-horizon memmaps at target index
                    for h in range(1, self.steps + 1):
                        target_idx = i + h - 1
                        if target_idx >= n_val:
                            continue
                        memmaps_X[h][target_idx, j] = preds[h - 1]
                        memmaps_y[h][target_idx] = float(y_val.iloc[target_idx])

                    # periodic memory logging
                    if i % 1000 == 0:
                        counts = {h: int(np.count_nonzero(~np.isnan(memmaps_y[h]))) for h in sample_horizons}
                        logger.info(
                            "iter=%d rss=%d sample_counts=%s",
                            i,
                            proc.memory_info().rss,
                            counts,
                        )

                # unload any forecaster loaded from disk to free memory
                if loaded_from_disk and forec is not None:
                    try:
                        del forec
                    except Exception:
                        pass
                    gc.collect()

            # Construct meta datasets by selecting non-nan rows per horizon
            meta_datasets: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
            for h in range(1, self.steps + 1):
                y_h = np.array(memmaps_y[h], dtype=np.float32)
                mask = ~np.isnan(y_h)
                if not np.any(mask):
                    meta_datasets[h] = (
                        np.empty((0, n_base), dtype=np.float32),
                        np.empty((0,), dtype=np.float32),
                    )
                else:
                    X_h = np.asarray(memmaps_X[h][mask, :], dtype=np.float32)
                    meta_datasets[h] = (X_h, y_h[mask].astype(np.float32))

                # free memmap objects and remove files to reclaim disk
                try:
                    del memmaps_X[h]
                    del memmaps_y[h]
                except Exception:
                    pass
                try:
                    os.remove(os.path.join(tmpdir, f"meta_X_h{h}.npy"))
                except Exception:
                    pass
                try:
                    os.remove(os.path.join(tmpdir, f"meta_y_h{h}.npy"))
                except Exception:
                    pass
                gc.collect()

        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                logger.exception("Failed to remove temporary meta dir %s", tmpdir)

        logger.info("meta generation complete rss=%d", proc.memory_info().rss)
        return meta_datasets

    def predict(
        self, last_window: Iterable[float], _exog: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Predict `steps` horizons given the last_window (length `lags`).

        Returns a numpy array of shape (steps,).
        """
        last_window = np.asarray(last_window, dtype=np.float32)
        if last_window.shape[0] != self.lags:
            logger.warning(
                "last_window length (%d) != lags (%d). Attempting to adapt.",
                last_window.shape[0],
                self.lags,
            )
            if last_window.shape[0] < self.lags:
                pad = np.full((self.lags - last_window.shape[0],), np.nan, dtype=np.float32)
                last_window = np.concatenate([pad, last_window])
            else:
                last_window = last_window[-self.lags :]

        base_pred_matrix = np.zeros((self.steps, len(self.base_names)), dtype=np.float32)
        for j, name in enumerate(self.base_names):
            try:
                preds = np.asarray(
                    self.forecasters[name].predict(steps=self.steps, last_window=last_window),
                    dtype=np.float32,
                )
            except Exception:
                preds = np.full((self.steps,), np.nan, dtype=np.float32)
            base_pred_matrix[:, j] = preds

        final = np.zeros((self.steps,), dtype=np.float32)
        for h in range(1, self.steps + 1):
            meta = self.meta_models[h - 1]
            row = base_pred_matrix[h - 1 : h, :]
            if meta is not None:
                try:
                    final[h - 1] = float(meta.predict(row)[0])
                except Exception:
                    final[h - 1] = float(np.nanmean(row))
            else:
                # fallback: simple mean of base predictions for the horizon
                final[h - 1] = float(np.nanmean(row))

        return final

    def save(
        self,
        manifest_path: Optional[str] = None,
        use_mlflow: bool = True,
        registered_model_name: Optional[str] = None,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
        run_name: Optional[str] = None,
    ) -> None:
        """Persist base forecasters and meta models.

        By default this will persist the ensemble to MLflow (use_mlflow=True). If
        `use_mlflow` is False the method falls back to the original local save
        behavior writing artifacts under `self.models_dir` (backwards compatible).
        """
        if use_mlflow:
            tmpdir = Path(tempfile.mkdtemp(prefix="skf_ensemble_"))
            try:
                self.save_local_temp(tmpdir)
                mlm = MLFlowManager()
                mlm.log_model_registry(
                    local_model_dir=tmpdir,
                    sk_model=None,
                    registered_model_name=registered_model_name,
                    run_name=run_name,
                    signature=signature,
                    input_example=input_example,
                )
            except Exception:
                # Keep tmpdir for debugging and re-raise
                logger.exception(
                    "Failed to save ensemble to MLflow; preserving tmpdir=%s", str(tmpdir)
                )
                raise
            else:
                try:
                    shutil.rmtree(tmpdir)
                except Exception:
                    logger.exception(
                        "Failed to remove temporary dir %s after MLflow logging", str(tmpdir)
                    )
        else:
            # Backwards-compatible local save
            manifest = {
                "base_forecasters": {},
                "meta_models": [],
                "lags": self.lags,
                "steps": self.steps,
            }

            for name, forecaster in self.forecasters.items():
                fname = f"forecaster_{name}.pkl"
                path = os.path.join(self.models_dir, fname)
                # If the in-memory forecaster was evicted (None) but a persisted
                # path exists, copy the persisted artifact to the destination.
                if forecaster is None:
                    persisted = self._forecaster_paths.get(name)
                    if persisted and os.path.exists(persisted):
                        try:
                            shutil.copy(persisted, path)
                            manifest["base_forecasters"][name] = fname
                            continue
                        except Exception:
                            logger.exception("Failed to copy persisted forecaster for %s", name)
                joblib.dump(forecaster, path)
                manifest["base_forecasters"][name] = fname

            for i, meta in enumerate(self.meta_models, start=1):
                if meta is None:
                    manifest["meta_models"].append(None)
                    continue
                fname = f"meta_step_{i}.pkl"
                path = os.path.join(self.models_dir, fname)
                joblib.dump(meta, path)
                manifest["meta_models"].append(fname)

            if manifest_path is None:
                manifest_path = os.path.join(self.models_dir, MANIFEST_FILENAME)

            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2)

    def save_local_temp(self, tmpdir: Path) -> None:
        """Serialize the ensemble to a temporary directory suitable for MLflow logging.

        Layout written:
        - forecasters/ (joblib .joblib files)
        - meta_models/ (joblib .joblib files)
        - scaler.joblib
        - manifest.json
        """
        tmpdir = Path(tmpdir)
        forecasters_dir = tmpdir / "forecasters"
        meta_models_dir = tmpdir / "meta_models"
        forecasters_dir.mkdir(parents=True, exist_ok=True)
        meta_models_dir.mkdir(parents=True, exist_ok=True)

        # Save each forecaster with compression for smaller artifacts
        for name, f in self.forecasters.items():
            dest = forecasters_dir / f"{name}.joblib"
            # If the in-memory object is None but we have a persisted artifact,
            # reuse that file instead of dumping None.
            if f is None:
                persisted = self._forecaster_paths.get(name)
                if persisted and os.path.exists(persisted):
                    try:
                        shutil.copy(persisted, dest)
                        continue
                    except Exception:
                        logger.exception("Failed to copy persisted forecaster %s to tmpdir", name)
            joblib.dump(f, dest, compress=3)

        # Save meta models
        for i, m in enumerate(self.meta_models, start=1):
            if m is None:
                continue
            dest = meta_models_dir / f"meta_step_{i}.joblib"
            joblib.dump(m, dest, compress=3)

        # Save scaler if present
        if getattr(self, "scaler", None) is not None:
            joblib.dump(self.scaler, tmpdir / "scaler.joblib", compress=3)

        # Manifest for loading
        manifest = {
            "base_forecasters": {
                name: f"forecasters/{name}.joblib" for name in self.forecasters.keys()
            },
            "meta_models": [
                f"meta_models/meta_step_{i}.joblib" if m is not None else None
                for i, m in enumerate(self.meta_models, start=1)
            ],
            "lags": self.lags,
            "steps": self.steps,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(tmpdir / MANIFEST_FILENAME, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

    @classmethod
    def load(cls, models_dir: str = "models/skforecast") -> "EnsembleForecaster":
        """Load an EnsembleForecaster from disk (requires manifest.json in models_dir).

        Note: this returns an instance with forecasters and meta models restored.
        """
        manifest_path = os.path.join(models_dir, MANIFEST_FILENAME)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)

        instance = cls(
            lags=manifest.get("lags", 30), steps=manifest.get("steps", 20), models_dir=models_dir
        )
        # load forecasters
        for name, fname in manifest.get("base_forecasters", {}).items():
            path = os.path.join(models_dir, fname)
            instance.forecasters[name] = joblib.load(path)

        instance.meta_models = []
        for entry in manifest.get("meta_models", []):
            if entry is None:
                instance.meta_models.append(None)
                continue
            instance.meta_models.append(joblib.load(os.path.join(models_dir, entry)))

        return instance

    @classmethod
    def load_from_mlflow(
        cls,
        registered_model_name: str,
        version: Optional[str] = None,
        dst_path: Optional[str] = None,
    ) -> "EnsembleForecaster":
        """Download registered model artifacts from MLflow Model Registry and load the ensemble.

        Returns an EnsembleForecaster loaded from the downloaded artifacts directory.
        """
        mlm = MLFlowManager()
        local_path = mlm.download_registered_model_artifacts(
            registered_model_name, version, dst_path
        )
        # mlm.download_registered_model_artifacts returns a Path
        return cls.load(models_dir=str(local_path))
