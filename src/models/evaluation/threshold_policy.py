"""
Threshold Policy Module

Centralizes confidence thresholding logic for prediction filtering.

Provides:
- ThresholdConfig: configuration for thresholding behavior
- ThresholdResult: mask, indices, and stats bundle
- ThresholdPolicy: compute_mask implementation (extensible)
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__, utility="evaluation")


@dataclass
class ThresholdConfig:
    method: str = (
        "ge"  # "ge" (>=), "gt" (>), future: "quantile", "topk", "adaptive", "per_group"
    )
    value: Optional[float] = None
    quantile: Optional[float] = None
    top_k: Optional[int] = None
    group_key: Optional[str] = None
    min_ratio: float = 0.0005
    max_ratio: float = 0.05
    hysteresis: Optional[dict] = None
    adaptive: Optional[dict] = None


@dataclass
class ThresholdResult:
    mask: np.ndarray
    indices: np.ndarray
    stats: Dict[str, float]


class ThresholdPolicy:
    """
    Compute boolean masks for confidence-based filtering given a configuration.

    Notes
    - Sanitizes non-finite confidence values (NaN/Inf): they are excluded from kept set
    - Supports vectorized operations
    - Extensible via `method` dispatch
    """

    def compute_mask(
        self, confidence: np.ndarray, X: Optional[pd.DataFrame], cfg: ThresholdConfig
    ) -> ThresholdResult:
        if confidence is None:
            raise ValueError("confidence must not be None")

        if X is not None and len(confidence) != len(X):
            raise ValueError(
                f"Confidence length {len(confidence)} does not match X length {len(X)}"
            )

        # Ensure 1D numpy array
        conf = np.asarray(confidence).reshape(-1)
        total_samples = conf.shape[0]
        avg_confidence = float(conf.mean())

        # Sanitize non-finite
        finite_mask = np.isfinite(conf)
        non_finite_count = int((~finite_mask).sum())
        if non_finite_count > 0:
            logger.warning(
                f"threshold_policy_non_finite_confidence count={non_finite_count} total={total_samples}"
            )

        # Default dispatch
        method = (cfg.method or "ge").lower()
        if method == "ge":
            if cfg.value is None:
                raise ValueError("ThresholdConfig.value is required for method 'ge'")
            raw_mask = conf >= float(cfg.value)
        elif method == "gt":
            if cfg.value is None:
                raise ValueError("ThresholdConfig.value is required for method 'gt'")
            raw_mask = conf > float(cfg.value)
        else:
            raise ValueError(
                f"Unknown threshold method '{cfg.method}'. Supported: 'ge', 'gt'"
            )

        # Exclude non-finite from raw_mask
        final_mask = np.logical_and(raw_mask, finite_mask)
        indices = np.where(final_mask)[0]

        samples_kept = int(final_mask.sum())
        samples_kept_ratio = (
            float(samples_kept / total_samples) if total_samples else 0.0
        )

        stats = {
            "samples_kept": samples_kept,
            "total_samples": int(total_samples),
            "samples_kept_ratio": samples_kept_ratio,
            "avg_confidence": avg_confidence,
            "non_finite_confidence_count": non_finite_count,
            "policy_method": method,
            "policy_value": float(cfg.value) if cfg.value is not None else None,
        }

        # logger.info(
        #     f"threshold_policy_filter policy_method={method} policy_params={{'value': {cfg.value}}} "
        #     f"stats={{'samples_kept': {samples_kept}, 'total_samples': {total_samples}, "
        #     f"'samples_kept_ratio': {samples_kept_ratio:.4f}, 'avg_confidence': {avg_confidence}, 'max_confidence': {max_confidence}, "
        #     f"'non_finite_confidence_count': {non_finite_count}}}"
        # )

        return ThresholdResult(mask=final_mask, indices=indices, stats=stats)
