---
title: Test Templates & Examples — StockPredictor_V1
created_by: Ronyka77
created_on: 2025-08-15
confidence: 95/100
---

Purpose
- Ready-to-copy pytest templates and small examples (unit, integration, parametrized, Hypothesis property tests, and golden/snapshot checks) to accelerate consistent test creation.

Test case template (documentation)
- ID: `MOD-<module>-NNN`
- Title: short descriptive title
- Preconditions: fixtures, env vars, golden files
- Steps: concise steps to reproduce
- Inputs: key inputs
- Expected: assertions
- Fixtures: list of required fixtures
- Markers: primary marker (`unit`/`integration`) and any secondary markers

Unit test template (copy-paste)
```python
import pytest

from src.module.example import some_function

@pytest.mark.unit
def test_some_function_basic_case():
    # Arrange
    inp = {"a": 1}

    # Act
    out = some_function(inp)

    # Assert
    assert out["result"] == 2
```

Parametrized example
```python
import pytest

@pytest.mark.unit
@pytest.mark.parametrize("input_val,expected", [(1, 2), (2, 3), (0, 1)])
def test_increment_parametrized(input_val, expected):
    assert increment(input_val) == expected
```

Integration test template (DB / filesystem)
```python
import pytest

from src.data_collector.polygon_news.storage import PolygonNewsStorage

@pytest.mark.integration
def test_store_article_roundtrip(tmp_path, db_session, sample_raw_article_full):
    storage = PolygonNewsStorage(db_session)
    res = storage.store_article(sample_raw_article_full)
    assert res is not None
    # verify persisted row exists
    stored = db_session.query(...).filter_by(id=res).first()
    assert stored is not None
```

Hypothesis (property-based) example
```python
from hypothesis import given, strategies as st
import pandas as pd

@given(df=st.data())
def test_validate_and_clean_data_properties(df):
    # Example: construct small dataframe with floats and NaN
    xs = pd.DataFrame({"a": [1.0, None, float('inf')]})
    cleaned = MLPDataUtils.validate_and_clean_data(xs)
    assert not cleaned.isnull().any().any()
```

Note: include `hypothesis` in dev dependencies when adopting property tests; use limits and settings to keep runs fast.

Snapshot / Parquet roundtrip template
```python
import pandas as pd
import numpy as np

def test_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({"a": np.arange(3)})
    path = tmp_path / "sample.parquet"
    df.to_parquet(path)
    loaded = pd.read_parquet(path)
    pd.testing.assert_frame_equal(df, loaded, check_names=False)
```

Golden Excel schema check (predictors)
```python
def test_save_predictions_to_excel_schema(tmp_path, monkeypatch):
    # prepare Predictor stub that writes an Excel file
    p = DummyPredictor()
    monkeypatch.chdir(tmp_path)
    out = p.save_predictions_to_excel(features_df, metadata_df, predictions)
    assert out.endswith('.xlsx')
    df = pd.read_excel(out)
    expected = {"ticker_id", "date_int", "predicted_return", "predicted_price"}
    assert expected.issubset(set(df.columns))
```

Fixture usage examples
- `mock_http_client` (see `tests/_fixtures/conftest.py`) — set return by `mock_http_client.set_aggregates([...])`.
- `freeze_time` (see `tests/_fixtures/frozen_time.py`) — wrap time-sensitive tests:
```python
def test_rate_limiter(monkeypatch):
    with freeze_time(monkeypatch, start=0.0):
        rl = RateLimiter(requests_per_minute=3)
        # ...
```

Best practices & small checklist
- Use explicit markers: `@pytest.mark.unit` or `@pytest.mark.integration`.
- Keep unit tests small and focused; prefer in-memory data.
- For property-based tests throttle input size and use `@settings(max_examples=50)` in Hypothesis.
- Use `tmp_path` for filesystem isolation and `monkeypatch` for environment control.
- Reference requirement IDs in test docstrings or comments for traceability.

Where to add templates
- Add new templates/examples to `docs/08-technical-reference/tests/TEST_TEMPLATES.md` and reference them from module guides.


