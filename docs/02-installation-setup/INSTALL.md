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

# 4. Verify installation
python run_polygon_examples.py
```

### Method 2: Regular Installation

```bash
# 1. Navigate to project directory
cd StockPredictor_V1

# 2. Install the package
uv pip install .

# 3. Run the main script
stockpredictor
```

### Method 3: Virtual Environment with uv

```bash
# 1. Create a new virtual environment
uv venv stockpredictor-env

# 2. Activate the environment
# Windows:
stockpredictor-env\Scripts\activate
# Linux/Mac:
source stockpredictor-env/bin/activate

# 3. Install the project
uv pip install -e .

# 4. Run examples
python run_polygon_examples.py
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

```sql
-- Create database
CREATE DATABASE stock_predictor;

-- Grant permissions (if needed)
GRANT ALL PRIVILEGES ON DATABASE stock_predictor TO postgres;
```

## Usage Examples

### Quick Test
```bash
# Test basic functionality
python test_polygon.py

# Or use the installed command
polygon-test
```

### Run Main Application
```bash
# Using the installed script
stockpredictor

# Or directly
python run_polygon_examples.py
```

### Run Specific Examples
```bash
# Run the example module
python -m src.data_collector.polygon.example_usage

# Or navigate to the module
cd src/data_collector/polygon
python example_usage.py
```

### Programmatic Usage
```python
from src.data_collector.polygon import DataPipeline
from datetime import date, timedelta

# Quick data fetch
with DataPipeline() as pipeline:
    stats = pipeline.run_incremental_update(
        days_back=7,
        max_tickers=10
    )
    print(f"Success rate: {stats.success_rate:.1f}%")
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
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Test connection
python -c "from src.data_collector.polygon.data_storage import DataStorage; print(DataStorage().health_check())"
```

### API Issues
```bash
# Test API connectivity
python -c "from src.data_collector.polygon import PolygonDataClient; print(PolygonDataClient().health_check())"
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

### 3. Testing
```bash
# Run basic tests
python test_polygon.py

# Run examples
python run_polygon_examples.py
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
from src.data_collector.polygon.config import config

# Data models
from src.data_collector.polygon.data_validator import OHLCVRecord
```

## Performance Tips

1. **Use Development Mode**: Install with `-e` flag for faster development
2. **Virtual Environment**: Use `uv venv` for isolated environments
3. **Batch Processing**: Use appropriate batch sizes for your system
4. **Rate Limiting**: Respect API limits (5 req/min for free tier)

## Support

If you encounter issues:

1. Check the logs: `tail -f polygon_data_acquisition.log`
2. Verify configuration: `python -c "from src.data_collector.polygon.config import config; print(config.__dict__)"`
3. Test components individually using the example scripts
4. Check database connectivity and API key validity 