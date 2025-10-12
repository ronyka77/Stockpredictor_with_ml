# Concurrent Dividend Processing

This document explains how to use the concurrent dividend processing feature for improved performance when ingesting dividend data from Polygon.io.

## Overview

The dividend pipeline supports both sequential and concurrent processing modes:

- **Sequential**: Process tickers one by one (default, backward compatible)
- **Concurrent**: Process multiple tickers simultaneously using ThreadPoolExecutor

## Benefits of Concurrent Processing

- **Faster processing**: Multiple tickers processed in parallel
- **Better resource utilization**: CPU cores and network I/O used more efficiently
- **Maintained API compliance**: Each thread has its own rate limiter

## Usage

### Programmatic Usage

```python
from src.data_collector.polygon_data.dividend_pipeline import ingest_dividends_for_all_tickers

# Sequential processing (default)
result = ingest_dividends_for_all_tickers(concurrent=False)

# Concurrent processing
result = ingest_dividends_for_all_tickers(
    concurrent=True,
    max_workers=4,           # Number of parallel threads
    requests_per_minute=5,   # API requests per minute per worker
    batch_size=100           # Database batch size
)
```

### Direct Configuration

Configure concurrent processing by modifying the `DIVIDEND_INGESTION_CONFIG` dictionary in `src/data_collector/polygon_data/dividend_pipeline.py`:

```python
# Configuration variables for dividend ingestion
DIVIDEND_INGESTION_CONFIG = {
    "concurrent": True,         # Enable concurrent processing
    "max_workers": 4,          # Number of worker threads
    "requests_per_minute": 5,  # API requests per minute per worker
    "batch_size": 100,         # Database batch size for upserts
}
```

Then run the script:

```bash
# Run with configured settings
uv run python src/data_collector/polygon_data/dividend_pipeline.py
```

### Example Script

See `examples/dividend_concurrent_example.py` for a complete example including performance comparison.

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `concurrent` | `False` | Enable concurrent processing |
| `max_workers` | `4` | Maximum number of parallel threads |
| `requests_per_minute` | `5` | API requests per minute per worker |
| `batch_size` | `100` | Database batch size for upserts |

## Rate Limiting Considerations

Each worker thread creates its own `PolygonDataClient` instance with independent rate limiting. The effective total API rate is:

```
total_requests_per_minute = max_workers × requests_per_minute
```

**Example:**
- 4 workers × 5 rpm each = 20 total API requests per minute
- Adjust based on your Polygon.io plan limits

## Thread Safety

The implementation ensures thread safety through:

- **Per-thread clients**: Each thread gets its own PolygonDataClient instance
- **Thread-safe result collection**: Uses `threading.Lock` for shared statistics
- **Database connection pooling**: Leverages existing ThreadedConnectionPool

## Performance Tuning

### Finding Optimal Settings

1. **Start conservative**: Begin with 2-3 workers and monitor API usage
2. **Monitor API limits**: Check Polygon.io dashboard for rate limit hits
3. **Adjust based on system resources**: More workers need more CPU/memory
4. **Consider network latency**: Concurrent requests may increase network contention

### Recommended Configurations

**Conservative (for free tier):**
```python
max_workers=2, requests_per_minute=3  # 6 total rpm
```

**Balanced (for paid plans):**
```python
max_workers=4, requests_per_minute=5  # 20 total rpm
```

**Aggressive (for high-tier plans):**
```python
max_workers=8, requests_per_minute=10  # 80 total rpm
```

## Error Handling

Concurrent processing includes enhanced error handling:

- **Per-thread error isolation**: One failed ticker doesn't stop others
- **Comprehensive logging**: Thread IDs included in log messages
- **Result aggregation**: Success/failure counts tracked across all threads

## Monitoring

Monitor the following metrics:

- **Processing time**: Compare sequential vs concurrent performance
- **Error rates**: Check for API failures or data quality issues
- **Resource usage**: CPU, memory, and network utilization
- **API usage**: Rate limiting compliance

## Troubleshooting

### Common Issues

1. **Rate limit exceeded**: Reduce `requests_per_minute` or `max_workers`
2. **Database connection errors**: Check ThreadedConnectionPool configuration
3. **Memory issues**: Reduce `max_workers` or `batch_size`
4. **Network timeouts**: Increase timeout values or reduce concurrency

### Debugging

Enable debug logging to see thread-specific processing:

```python
import logging
logging.getLogger("data_collector").setLevel(logging.DEBUG)
```

## Backward Compatibility

The concurrent processing feature is fully backward compatible. Existing code will continue to work unchanged with sequential processing as the default.
