"""
Safe numeric helpers for feature engineering.

Provides SafeDivideSpec dataclass and safe_divide_series/compute_bb_position helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SafeDivideSpec:
    """Specification for safe division behavior.

    Attributes:
        denom_epsilon: Treat denominators with abs <= denom_epsilon as zero.
        fill_value: If not None, fill NaNs with this constant (e.g., 0.5). If None, leave NaN.
        sanitize_infinite: If True, replace +/-inf with NaN.
    """

    denom_epsilon: float = 0.0
    fill_value: Optional[float] = None
    sanitize_infinite: bool = True

    def __post_init__(self) -> None:
        if self.denom_epsilon < 0:
            raise ValueError("denom_epsilon must be >= 0")


def safe_divide_series(numer: pd.Series, denom: pd.Series, spec: SafeDivideSpec) -> pd.Series:
    """Compute numer / denom safely according to the specification.

    Preconditions:
        - `numer` and `denom` are broadcastable / index-aligned pandas Series.
    Postconditions:
        - If `spec.sanitize_infinite` is True, result will not contain +/-inf.
        - If `spec.fill_value` is not None, NaNs will be filled with that value.

    Returns:
        pd.Series of float values (NaNs where division is undefined).
    """

    # Align indices
    if not isinstance(numer, pd.Series):
        numer = pd.Series(numer)
    if not isinstance(denom, pd.Series):
        denom = pd.Series(denom)

    # Perform safe division using numpy where to avoid warnings
    # Identify near-zero denominators
    small_denom_mask = denom.abs() <= spec.denom_epsilon

    # Avoid division by zero: when denom is small, set result to NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numer.astype(float) / denom.astype(float)

    # Replace infinities if requested
    if spec.sanitize_infinite:
        result.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Force NaN where denom considered zero
    if small_denom_mask.any():
        result.loc[small_denom_mask.index] = result.where(~small_denom_mask, np.nan)

    # Optionally fill NaNs with provided fill_value
    if spec.fill_value is not None:
        result = result.fillna(spec.fill_value)

    # Ensure float dtype
    return result.astype(float)


def compute_bb_position(
    close: pd.Series, bb_lower: pd.Series, bb_upper: pd.Series, spec: SafeDivideSpec
) -> pd.Series:
    """Compute BB_Position as (close - bb_lower) / (bb_upper - bb_lower) safely.

    Uses SafeDivideSpec to control behavior when band width is zero.
    """
    numer = close - bb_lower
    denom = bb_upper - bb_lower
    return safe_divide_series(numer, denom, spec)
