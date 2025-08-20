import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessorArtifacts:
    scaler: RobustScaler
    clip_stats: Dict[str, Tuple[float, float]]
    cat_maps: Dict[str, Dict[str, int]]
    feature_names: List[str]


class RealMLPPreprocessor:
    """
    Fit/transform utilities for RealMLP.

    - Numeric: clip to [q1, q2] per feature, then RobustScaler
    - Categorical: `ticker_id` mapped to indices with OOV index 0
    - Persist: scaler.pkl, clip_stats.json, cat_maps.json, feature_names.json
    """

    def __init__(
        self,
        *,
        numeric_clip_q1: float = 0.01,
        numeric_clip_q2: float = 0.99,
        categorical_cols: Optional[List[str]] = None,
        oov_index: int = 0,
    ) -> None:
        self.numeric_clip_q1 = numeric_clip_q1
        self.numeric_clip_q2 = numeric_clip_q2
        self.categorical_cols = categorical_cols or ["ticker_id"]
        self.oov_index = oov_index

        self.scaler: Optional[RobustScaler] = None
        self.clip_stats: Dict[str, Tuple[float, float]] = {}
        self.cat_maps: Dict[str, Dict[str, int]] = {}
        self.feature_names: List[str] = []

    def fit(self, df: pd.DataFrame, *, numeric_cols: List[str]) -> "RealMLPPreprocessor":
        self._compute_clip_stats(df, numeric_cols)
        clipped = self._apply_clipping(df[numeric_cols].copy())
        self.scaler = RobustScaler()
        self.scaler.fit(clipped.values)
        self.feature_names = list(numeric_cols)
        # Build categorical maps (only ticker_id for now)
        for col in self.categorical_cols:
            if col in df.columns:
                self.cat_maps[col] = self._build_cat_map(df[col])
        return self

    def transform(self, df: pd.DataFrame, *, numeric_cols: List[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.scaler is None:
            raise RuntimeError("Preprocessor not fitted")
        # Numeric pipeline
        clipped = self._apply_clipping(df[numeric_cols].copy())
        X_num = self.scaler.transform(clipped.values)
        # Categorical indices (ticker_id)
        cat_idx = None
        for col in self.categorical_cols:
            if col in df.columns and col in self.cat_maps:
                mapping = self.cat_maps[col]
                mapped = df[col].astype(str).map(mapping)
                # Count unseen before fillna
                unseen_mask = ~df[col].astype(str).isin(mapping.keys())
                unseen_count = int(unseen_mask.sum())
                cat_idx = mapped.fillna(self.oov_index).astype(int).to_numpy()
                if unseen_count > 0:
                    logger.warning(f"Unseen {col} values encountered; mapped to OOV index {self.oov_index}. count={unseen_count}")
        return X_num.astype(np.float32), cat_idx

    def _compute_clip_stats(self, df: pd.DataFrame, numeric_cols: List[str]) -> None:
        for col in numeric_cols:
            q1 = df[col].quantile(self.numeric_clip_q1)
            q2 = df[col].quantile(self.numeric_clip_q2)
            self.clip_stats[col] = (float(q1), float(q2))

    def _apply_clipping(self, df_num: pd.DataFrame) -> pd.DataFrame:
        for col, (lo, hi) in self.clip_stats.items():
            if col in df_num.columns:
                df_num[col] = df_num[col].clip(lower=lo, upper=hi)
        return df_num

    def _build_cat_map(self, series: pd.Series) -> Dict[str, int]:
        unique_vals = sorted(series.dropna().astype(str).unique())
        mapping: Dict[str, int] = {val: idx + 1 for idx, val in enumerate(unique_vals)}
        mapping["__OOV__"] = self.oov_index
        return mapping

    def save_artifacts(self, base_dir: Path) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)
        with open(base_dir / "robust_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(base_dir / "clip_stats.json", "w") as f:
            json.dump(self.clip_stats, f)
        with open(base_dir / "cat_maps.json", "w") as f:
            json.dump(self.cat_maps, f)
        with open(base_dir / "feature_names.json", "w") as f:
            json.dump(self.feature_names, f)

    def load_artifacts(self, base_dir: Path) -> "RealMLPPreprocessor":
        with open(base_dir / "robust_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open(base_dir / "clip_stats.json", "r") as f:
            self.clip_stats = json.load(f)
        with open(base_dir / "cat_maps.json", "r") as f:
            self.cat_maps = json.load(f)
        with open(base_dir / "feature_names.json", "r") as f:
            self.feature_names = json.load(f)
        return self


