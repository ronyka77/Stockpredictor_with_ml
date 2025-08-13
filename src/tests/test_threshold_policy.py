import numpy as np
import pandas as pd

from src.models.evaluation.threshold_policy import ThresholdPolicy, ThresholdConfig


def test_threshold_policy_ge_parity_simple():
    conf = np.array([0.1, 0.5, 0.6, 0.62, 0.0, 1.0], dtype=float)
    X = pd.DataFrame({"f": np.arange(len(conf))})
    policy = ThresholdPolicy()
    cfg = ThresholdConfig(method="ge", value=0.62)
    res = policy.compute_mask(conf, X, cfg)
    expected_mask = conf >= 0.62
    assert np.array_equal(res.mask, expected_mask)
    assert np.array_equal(res.indices, np.where(expected_mask)[0])
    assert res.stats["samples_kept"] == int(expected_mask.sum())
    assert res.stats["total_samples"] == len(conf)


def test_threshold_policy_sanitizes_non_finite():
    conf = np.array([0.7, np.nan, np.inf, 0.8], dtype=float)
    X = pd.DataFrame({"f": np.arange(len(conf))})
    policy = ThresholdPolicy()
    cfg = ThresholdConfig(method="ge", value=0.7)
    res = policy.compute_mask(conf, X, cfg)
    assert res.stats["non_finite_confidence_count"] == 2
    # Only finite and >= 0.7 kept -> indices [0, 3]
    assert np.array_equal(res.indices, np.array([0, 3]))


def test_threshold_policy_value_required():
    conf = np.array([0.3, 0.4])
    policy = ThresholdPolicy()
    try:
        policy.compute_mask(conf, None, ThresholdConfig(method="ge", value=None))
    except ValueError as e:
        assert "required" in str(e)
    else:
        assert False, "Expected ValueError for missing value"


def test_threshold_policy_len_mismatch_raises():
    conf = np.array([0.1, 0.2, 0.3])
    X = pd.DataFrame({"f": [1, 2]})
    policy = ThresholdPolicy()
    try:
        policy.compute_mask(conf, X, ThresholdConfig(method="ge", value=0.2))
    except ValueError as e:
        assert "does not match" in str(e)
    else:
        assert False, "Expected ValueError for length mismatch"


