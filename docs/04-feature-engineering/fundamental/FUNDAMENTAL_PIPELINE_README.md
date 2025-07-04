# Fundamental Analysis Pipeline

This directory contains a complete pipeline for fundamental analysis of stocks, including data collection, calculation, and storage of fundamental metrics.

## Overview

The fundamental analysis pipeline provides:

- **Data Collection**: Automated collection of financial statements from Polygon API
- **Metric Calculation**: 60+ fundamental metrics across 4 categories
- **Data Validation**: Quality checks and error handling
- **Database Storage**: Structured storage of calculated metrics
- **Monitoring**: Health checks and performance monitoring

## Components

### Core Pipeline (`fundamental_pipeline.py`)
- **FundamentalPipeline**: Main orchestrator class
- **PipelineResult**: Result tracking for individual tickers
- **PipelineStats**: Batch processing statistics

### Pipeline Runner (`run_fundamental_pipeline.py`)
- **FundamentalPipelineRunner**: Command-line interface
- Supports single ticker, batch, and S&P 500 sample processing
- Configurable logging and error handling

### Pipeline Monitor (`fundamental_monitor.py`)
- **FundamentalPipelineMonitor**: Health and performance monitoring
- Database connectivity checks
- Pipeline metrics and reporting

## Fundamental Calculators

### 1. Financial Ratios (`fundamental_indicators/ratios.py`)
- **Valuation**: Book value per share, revenue per share, asset turnover
- **Profitability**: Net/gross/operating margins, ROE, ROA, ROI
- **Liquidity**: Current ratio, quick ratio, cash ratio
- **Leverage**: Debt-to-equity, debt-to-assets, interest coverage

### 2. Growth Metrics (`fundamental_indicators/growth_metrics.py`)
- **Revenue Growth**: QoQ, YoY, 3-year CAGR
- **Earnings Growth**: QoQ and YoY growth rates
- **Asset Growth**: Multi-period growth analysis

### 3. Scoring Systems (`fundamental_indicators/scoring_systems.py`)
- **Altman Z-Score**: Bankruptcy prediction (5-component model)
- **Piotroski F-Score**: Financial health scoring (0-9 scale)

### 4. Sector Analysis (`fundamental_indicators/sector_analysis.py`)
- **GICS Sector Classification**: From SIC codes
- **Sector Ratios**: Key metrics for cross-sectional analysis
- **Percentile Rankings**: Sector-relative performance

## Usage

### Basic Usage

```bash
# Process a single ticker
python src/feature_engineering/run_fundamental_pipeline.py --ticker AAPL

# Process multiple tickers
python src/feature_engineering/run_fundamental_pipeline.py --tickers AAPL GOOGL MSFT

# Process S&P 500 sample
python src/feature_engineering/run_fundamental_pipeline.py --sp500-sample 10
```

### Monitoring

```bash
# Health check
python src/feature_engineering/fundamental_monitor.py --health-check

# Check specific ticker
python src/feature_engineering/fundamental_monitor.py --ticker AAPL

# Generate report
python src/feature_engineering/fundamental_monitor.py --report report.json
```

### Configuration

The pipeline uses `FundamentalConfig` from `config.py` with these key settings:

- **HISTORICAL_YEARS**: Years of historical data (default: 2)
- **OUTLIER_CAPPING**: Enable outlier capping (default: True)
- **MIN_FUNDAMENTAL_DATA_POINTS**: Minimum quarters needed (default: 8)
- **SECTOR_CLASSIFICATION**: Classification system (default: GICS)

## Database Schema

The pipeline stores results in 4 tables:

1. **fundamental_ratios**: Financial ratios and valuation metrics
2. **fundamental_growth_metrics**: Growth rates and trends
3. **fundamental_scores**: Scoring systems (Altman Z, Piotroski F)
4. **fundamental_sector_analysis**: Sector classification and ratios

## Data Flow

1. **Collection**: Fetch financial statements from Polygon API
2. **Validation**: Check data quality and completeness
3. **Calculation**: Run all 4 calculator types
4. **Storage**: Save results to database tables
5. **Monitoring**: Track performance and errors

## Error Handling

- **Data Quality Checks**: Minimum data requirements
- **Outlier Capping**: Prevent extreme values
- **Graceful Degradation**: Continue processing on individual failures
- **Comprehensive Logging**: Detailed error tracking

## Performance

- **Async Processing**: Non-blocking API calls
- **Concurrency Control**: Configurable rate limiting
- **Batch Processing**: Efficient multi-ticker handling
- **Caching**: Reduce redundant API calls (planned)

## Requirements

- **API Access**: Polygon API key required
- **Database**: PostgreSQL with fundamental tables
- **Dependencies**: See `pyproject.toml`

## Environment Variables

```bash
# Required
POLYGON_API_KEY=your_api_key_here

# Optional
DB_HOST=localhost
DB_PORT=5432
DB_NAME=stock_data
DB_USER=postgres
DB_PASSWORD=your_password
```

## Example Output

```
2024-06-08 22:59:45 - INFO - Processing ticker: AAPL
2024-06-08 22:59:47 - INFO - ✓ Successfully processed AAPL
2024-06-08 22:59:47 - INFO -   Metrics calculated: {'ratios': 15, 'growth_metrics': 8, 'scoring_systems': 2, 'sector_analysis': 5}
2024-06-08 22:59:47 - INFO -   Execution time: 2.34s
2024-06-08 22:59:47 - INFO -   Data quality: 0.95
```

## Next Steps

1. Set up Polygon API key
2. Configure database connection
3. Run initial S&P 500 sample
4. Set up monitoring dashboard
5. Schedule regular updates

For more details, see the individual module documentation. 