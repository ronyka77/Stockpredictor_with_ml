# Installation Guide - StockPredictor V1

This guide shows you how to install and run the StockPredictor V1 project using `uv`.

## Prerequisites

1. **Python 3.12+** installed
2. **uv** package manager installed
3. **PostgreSQL** database running
4. **Polygon.io API key** (free tier available)

## Installation Methods

### Method 1: Development Installation (Recommended)

```bash
# 1. Navigate to project directory
cd StockPredictor_V1

# 2. Install in development mode with uv
uv pip install -e .

# 3. Install development dependencies (optional)
uv pip install -e ".[dev]"

# 4. Verify environment (API client health)
uv run python -c "from src.data_collector.polygon_data.client import PolygonDataClient; print(PolygonDataClient().health_check())"
```

### Method 2: Regular Installation

```bash
# 1. Navigate to project directory
cd StockPredictor_V1

# 2. Install the package
uv pip install .

# 3. Run an example module (no CLI entrypoints required)
uv run python -m src.data_collector.polygon_data.example_usage
```

### Method 3: Virtual Environment with uv

```bash
# 1. Create a new virtual environment
uv venv stockpredictor-env

# 2. Activate the environment
# Windows (cmd):
stockpredictor-env\Scripts\activate
# Linux/Mac:
source stockpredictor-env/bin/activate

# 3. Install the project
uv pip install -e .

# 4. Run examples (module-based)
uv run python -m src.data_collector.polygon_data.example_usage
```

## Configuration

### 1. Environment Variables (Recommended)

Create a `.env` file in the project root:

```bash
# Polygon.io API
POLYGON_API_KEY=your_api_key_here

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=stock_predictor
DB_USER=postgres
DB_PASSWORD=your_password
```

### 2. Database Setup

Note: PostgreSQL is required for data storage, but installation and setup are out of scope for this guide.
Use your existing PostgreSQL instance and set DB_* variables in `.env` accordingly.

## Usage Examples

### Quick Test
```bash
# Test API connectivity
uv run python -c "from src.data_collector.polygon_data.client import PolygonDataClient; print(PolygonDataClient().health_check())"
```

### Run Data Collection Examples
```bash
# Stock data example
uv run python -m src.data_collector.polygon_data.example_usage
```

### Run Specific Examples
```bash
# Run news pipeline (module)
uv run python -m src.data_collector.polygon_news.news_pipeline
```

### Programmatic Usage
```python
from src.data_collector.polygon_data import DataPipeline
from datetime import date, timedelta

# Quick data fetch
pipeline = DataPipeline()
end_date = date.today()
start_date = end_date - timedelta(days=7)
pipeline.run_grouped_daily_pipeline(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        validate_data=True,
        save_stats=True)
```

## Troubleshooting

### Import Errors
```bash
# Reinstall in development mode
uv pip uninstall stockpredictor-v1
uv pip install -e .
```

### Database Connection Issues
```bash
# Test storage connection (from code)
uv run python -c "from src.data_collector.polygon_data.data_storage import DataStorage; print(DataStorage().health_check())"
```

### API Issues
```bash
# Test API connectivity
uv run python -c "from src.data_collector.polygon_data.client import PolygonDataClient; print(PolygonDataClient().health_check())"
```

## Development Workflow

### 1. Install Development Dependencies
```bash
uv pip install -e ".[dev]"
```

### 2. Code Formatting
```bash
# Format code
black src/
isort src/

# Type checking
mypy src/
```

## Package Structure

After installation, you can import modules like this:

```python
# Main components
from src.data_collector.polygon import (
    DataPipeline,
    PolygonDataClient,
    TickerManager,
    HistoricalDataFetcher,
    DataStorage,
    DataValidator
)

# Configuration
from src.data_collector.polygon_data.config import config

# Data models
from src.data_collector.polygon_data.data_validator import OHLCVRecord
```

## Performance Tips

1. **Use Development Mode**: Install with `-e` flag for faster development
2. **Virtual Environment**: Use `uv venv` for isolated environments
3. **Batch Processing**: Use appropriate batch sizes for your system
4. **Rate Limiting**: Respect API limits (5 req/min for free tier)

## Support

If you encounter issues:

1. Use the centralized logger in `src/utils/core/logger.py` for consistent logging
2. Verify configuration: `uv run python -c "from src.data_collector.polygon_data.config import config; print(config.__dict__)"`
3. Test components individually using the module examples
4. Check API key validity