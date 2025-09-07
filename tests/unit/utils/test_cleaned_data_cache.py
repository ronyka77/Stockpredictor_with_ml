import pandas as pd
import json
import pytest
from pathlib import Path
from src.utils.cleaned_data_cache import CleanedDataCache


def test_cleaned_data_cache_basic(tmp_path):
    cache = CleanedDataCache(cache_dir=tmp_path)
    df = pd.DataFrame({"a": [1, 2, 3]})
    cache.set("test_key", df)
    loaded = cache.get("test_key")
    if not loaded.equals(df):
        raise AssertionError("Loaded DataFrame does not equal saved DataFrame")


@pytest.mark.unit
def test_cache_key_and_roundtrip(tmp_path):
    cache_dir = tmp_path / "cache"
    c = CleanedDataCache(cache_dir=str(cache_dir))

    key = c._generate_cache_key(a=1, b="x")
    if not (isinstance(key, str) and len(key) == 32):
        raise AssertionError("Cache key is not a 32-char string")

    X_train = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    y_train = pd.Series([0.1, 0.2])
    X_test = pd.DataFrame({"a": [5], "b": [6.0]})
    y_test = pd.Series([0.3])

    data_result = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "target_column": "ret",
        "feature_count": 2,
        "prediction_horizon": 10,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "removed_features": {},
        "diversity_analysis": {},
    }

    c.save_cleaned_data(data_result, cache_key=key, data_type="training")
    if not c.cache_exists(key, data_type="training"):
        raise AssertionError("Cache entry was not stored as expected")

    loaded = c.load_cleaned_data(cache_key=key, data_type="training")
    if not set(["X_train", "y_train", "X_test", "y_test"]).issubset(loaded.keys()):
        raise AssertionError("Loaded cache missing expected keys")

    age = c.get_cache_age_hours(cache_key=key, data_type="training")
    if age is None or age < 0.0:
        raise AssertionError("Cache age is invalid")


@pytest.mark.unit
def test_clear_cache(tmp_path):
    c = CleanedDataCache(cache_dir=str(tmp_path))
    key = c._generate_cache_key(a=1)
    # create dummy info file to simulate existence
    info_path = (Path(str(tmp_path)) / f"training_{key}").with_suffix(".info.json")
    info_path.write_text(json.dumps({"cache_key": key}))
    if c.cache_exists(key, data_type="training") is not False:  # partial
        raise AssertionError(
            "Expected cache to indicate non-existence for partial info"
        )
    c.clear_cache()  # should not raise
    if any(tmp_path.iterdir()):
        raise AssertionError("Temporary path was not cleared as expected")
