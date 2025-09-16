import json
from pathlib import Path
from datetime import datetime, timedelta


from src.data_collector.polygon_fundamentals.cache_manager import (
    FundamentalCacheManager,
)


def make_cache_file(directory: Path, ticker: str, days_ago: int, payload: dict):
    """
    Create a JSON cache file for a ticker dated days_ago days before today and return its path.
    
    The file is named "{TICKER}_financials_{YYYYMMDD}.json" and is written under the provided directory. The payload is serialized with json.dumps(..., default=str) and written using UTF-8 encoding. If a file with the same name exists it will be overwritten.
    
    Parameters:
        directory (Path): Target directory for the cache file.
        ticker (str): Ticker symbol used as the filename prefix.
        days_ago (int): Number of days to subtract from today to form the date in the filename.
        payload (dict): JSON-serializable data to write to the file.
    
    Returns:
        Path: Path to the created cache file.
    """
    date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y%m%d")
    path = directory / f"{ticker}_financials_{date}.json"
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")
    return path


def test_get_cached_data_returns_none_when_no_files(tmp_path):
    """Return None when no cache files exist for a ticker"""
    cm = FundamentalCacheManager(cache_dir=str(tmp_path))

    # Execution
    result = cm.get_cached_data("ZZZZ")

    # Verification
    assert result is None


def test_get_cached_data_uses_most_recent_valid(tmp_path, fv_factory):
    """Prefer most recent valid cache file over older/expired ones"""
    cm = FundamentalCacheManager(cache_dir=str(tmp_path))

    # older (expired) cache
    make_cache_file(tmp_path, "AAPL", days_ago=10, payload={"results": []})
    try:
        fv = fv_factory.build(value=123.0)
        revenues_field = fv.model_dump() if hasattr(fv, "model_dump") else fv.__dict__
        recent = {"results": [{"revenues": revenues_field}]}
    except Exception:
        recent = {"results": [{"foo": "bar"}]}

    make_cache_file(tmp_path, "AAPL", days_ago=3, payload=recent)

    # Execution
    data = cm.get_cached_data("AAPL")

    # Verification
    assert isinstance(data, dict)
    assert data.get("results") == recent["results"]


def test_save_cache_and_clear_expired(tmp_path):
    """Save cache files and ensure clear_expired_caches removes old files"""
    cm = FundamentalCacheManager(cache_dir=str(tmp_path))

    # Execution: save cache for ticker
    saved = cm.save_cache("msft", {"x": 1}, overwrite=True)

    # Verification: file exists and has correct name
    assert saved is not None
    assert saved.exists()

    # Create an artificially old file and ensure clear_expired_caches removes it
    make_cache_file(tmp_path, "MSFT", days_ago=30, payload={"old": True})
    removed = cm.clear_expired_caches()

    assert removed >= 1
    # The original saved file for today should remain
    assert any(
        p.name.startswith("MSFT_financials_")
        for p in tmp_path.glob("MSFT_financials_*.json")
    )


def test_save_cache_no_overwrite(tmp_path):
    """Saving with overwrite=False does not replace existing cache file"""
    cm = FundamentalCacheManager(cache_dir=str(tmp_path))

    # Save once
    first = cm.save_cache("TEST", {"a": 1}, overwrite=True)
    assert first is not None and first.exists()

    # Save again with overwrite=False should return None and leave file unchanged
    second = cm.save_cache("TEST", {"a": 2}, overwrite=False)
    assert second is None

    # Ensure content is still the original
    data = json.loads(first.read_text(encoding="utf-8"))
    assert data.get("a") == 1


def test_parse_date_from_filename_invalid(tmp_path):
    """Gracefully handle filenames that don't match the cache pattern"""
    cm = FundamentalCacheManager(cache_dir=str(tmp_path))

    # Create a file that doesn't match expected pattern
    bad = tmp_path / "not_a_cache_file.txt"
    bad.write_text("x")

    # Attempt to get cached data returns None and should not raise
    assert cm.get_cached_data("NOPE") is None


def test_clear_expired_caches_empty_dir(tmp_path):
    """clear_expired_caches returns 0 when directory has no cache files"""
    cm = FundamentalCacheManager(cache_dir=str(tmp_path))
    removed = cm.clear_expired_caches()
    assert removed == 0


def test_save_cache_invalid_ticker(tmp_path):
    """Reject saving cache for empty ticker strings"""
    cm = FundamentalCacheManager(cache_dir=str(tmp_path))
    # Empty ticker should be rejected
    res = cm.save_cache("", {"a": 1})
    assert res is None
