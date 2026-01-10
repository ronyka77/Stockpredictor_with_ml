"""Volume regime helpers and configuration dataclass.

Provides RollingRegimePolicy and compute_volume_regimes to compute symmetric
volume regime flags using either standard-deviation or quantile-based methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass(frozen=True)
class RollingRegimePolicy:
    """Specifies how to detect high/low volume regimes.

    Attributes:
        lookback: number of points to use for rolling statistics (>=2)
        method: 'std' for symmetric mean +/- k*std, 'quantile' for rolling quantiles
        k: multiplier when using 'std' method (both sides use the same k)
        hi_q: upper quantile when using 'quantile' method
        lo_q: lower quantile when using 'quantile' method
    """

    lookback: int = 50
    method: str = "std"
    k: float = 2.0
    hi_q: float = 0.8
    lo_q: float = 0.2

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("RollingRegimePolicy.lookback must be >= 2")
        if self.method not in {"std", "quantile"}:
            raise ValueError("RollingRegimePolicy.method must be 'std' or 'quantile'")
        if self.method == "quantile":
            if not (0.0 < self.lo_q < self.hi_q < 1.0):
                raise ValueError("For quantile method require 0 < lo_q < hi_q < 1")


def compute_volume_regimes(
    volume: pd.Series, policy: RollingRegimePolicy = RollingRegimePolicy()
) -> Tuple[pd.Series, pd.Series]:
    """Compute symmetric volume regime flags.

    For method 'std':
        high = volume > rolling_mean + k * rolling_std
        low = volume < rolling_mean - k * rolling_std

    For method 'quantile':
        high = volume > rolling_hi_quantile
        low = volume < rolling_lo_quantile

    Returns:
        (high_flags, low_flags) where each is an int Series (0 or 1) aligned to
        the input index. Missing values are treated as 0.
    """
    if policy.method == "std":
        vol_mean = volume.rolling(policy.lookback).mean()
        vol_std = volume.rolling(policy.lookback).std()
        high = (volume > (vol_mean + policy.k * vol_std)).fillna(False).astype(int)
        low = (volume < (vol_mean - policy.k * vol_std)).fillna(False).astype(int)
    else:
        vol_hi = volume.rolling(policy.lookback).quantile(policy.hi_q)
        vol_lo = volume.rolling(policy.lookback).quantile(policy.lo_q)
        high = (volume > vol_hi).fillna(False).astype(int)
        low = (volume < vol_lo).fillna(False).astype(int)

    return high, low
