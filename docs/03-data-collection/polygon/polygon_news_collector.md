# Polygon News Collector - Implementation Documentation

## Overview

The Polygon News Collector is a comprehensive system for collecting, processing, and storing financial news data from the Polygon.io News API. This implementation provides intelligent ticker prioritization, incremental updates, historical backfill capabilities, and seamless integration with existing data infrastructure.

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Database Schema](#database-schema)
7. [API Reference](#api-reference)
8. [Examples](#examples)
9. [Monitoring & Maintenance](#monitoring--maintenance)
10. [Troubleshooting](#troubleshooting)

## Features

### Core Functionality
- **Incremental News Collection**: Automatically collects news from the last stored date to current
- **Historical Backfill**: 2-year historical data collection capability
- **Targeted Collection**: Collect news for specific tickers and date ranges
- **Intelligent Ticker Prioritization**: Market cap and volume-based ticker selection
- **Sentiment Analysis Storage**: Stores Polygon's pre-computed sentiment data
- **Quality Validation**: Content quality assessment and filtering
- **Data Retention Management**: Automatic cleanup of old data (configurable retention period)

### Technical Features
- **Rate Limiting**: Intelligent API rate limiting with adaptive throttling
- **Error Handling**: Comprehensive error handling with retry mechanisms
- **Batch Processing**: Efficient batch operations for database storage
- **Transaction Management**: Safe database operations with rollback support
- **Centralized Configuration**: Shared configuration system with environment variable support
- **Comprehensive Logging**: Detailed logging with centralized logger integration
- **Health Monitoring**: System health checks and status reporting

## Architecture

### Component Overview

```
src/data_collector/polygon_news/
├── __init__.py                 # Package initialization and exports
├── models.py                   # SQLAlchemy database models
├── storage.py                  # Database storage operations
├── news_client.py              # Polygon News API client
├── ticker_integration.py       # Ticker management integration
├── processor.py                # Content processing and cleaning
├── validator.py                # Data quality validation
├── news_collector.py           # Main orchestrator service
└── example_usage.py            # Usage examples and demos
```

### Data Flow

1. **Ticker Selection**: Prioritized ticker list from ticker manager or fallback lists
2. **News Fetching**: API calls to Polygon News API with date filtering
3. **Content Processing**: Text cleaning, metadata extraction, sentiment processing
4. **Quality Validation**: Article quality assessment and filtering
5. **Database Storage**: Batch upsert operations with transaction management
6. **Status Reporting**: Collection statistics and system health monitoring

## Installation & Setup

### Prerequisites

- Python 3.8+
- PostgreSQL database
- Polygon.io API key
- Required Python packages (see `pyproject.toml`)

### Environment Variables

```bash
# Required
export POLYGON_API_KEY='your_polygon_api_key_here'

# Optional (with defaults)
export DATABASE_URL='postgresql://user:password@localhost/stockdb'
export NEWS_MAX_TICKERS='100'
export NEWS_DAYS_LOOKBACK='7'
export NEWS_RETENTION_YEARS='2'
export NEWS_BATCH_SIZE='100'
```

### Database Setup

The system automatically creates required tables on first run:

```python
from src.data_collector.polygon_news.models import create_tables
from sqlalchemy import create_engine

engine = create_engine(database_url)
create_tables(engine)
```

## Configuration

### Centralized Configuration

All configuration is managed through `src/data_collector/config.py`:

```python
@dataclass
class PolygonConfig:
    # API Configuration
    API_KEY: str = os.getenv("POLYGON_API_KEY")
    BASE_URL: str = "https://api.polygon.io"
    REQUESTS_PER_MINUTE: int = 5
    MAX_RETRIES: int = 3
    REQUEST_TIMEOUT: int = 30
    
    # News Collection Configuration
    NEWS_MAX_TICKERS: int = int(os.getenv('NEWS_MAX_TICKERS', '100'))
    NEWS_DAYS_LOOKBACK: int = int(os.getenv('NEWS_DAYS_LOOKBACK', '7'))
    NEWS_RETENTION_YEARS: int = int(os.getenv('NEWS_RETENTION_YEARS', '2'))
    NEWS_BATCH_SIZE: int = int(os.getenv('NEWS_BATCH_SIZE', '100'))
    
    # Database Configuration
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "stockdb")
    DB_USER: str = os.getenv("DB_USER", "user")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "password")
```

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `POLYGON_API_KEY` | Required | Your Polygon.io API key |
| `DATABASE_URL` | Auto-generated | PostgreSQL connection string |
| `NEWS_MAX_TICKERS` | 100 | Maximum tickers to process per collection |
| `NEWS_DAYS_LOOKBACK` | 7 | Days to look back for new tickers |
| `NEWS_RETENTION_YEARS` | 2 | Years to retain news data |
| `NEWS_BATCH_SIZE` | 100 | Batch size for database operations |
| `REQUESTS_PER_MINUTE` | 5 | API rate limit (adjust based on plan) |

## Usage

### Quick Start

```python
from src.data_collector.polygon_news.news_collector import main

# Run incremental collection with main function
success = main()
```

### Command Line Usage

```bash
# Run incremental collection
python -m src.data_collector.polygon_news.news_collector

# Or from the news directory
cd src/data_collector/polygon_news
python news_collector.py
```

### Programmatic Usage

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.data_collector.polygon_news import PolygonNewsCollector
from src.data_collector.config import config

# Setup database
engine = create_engine(config.database_url)
Session = sessionmaker(bind=engine)
session = Session()

# Initialize collector
collector = PolygonNewsCollector(
    db_session=session,
    polygon_api_key=config.API_KEY,
    requests_per_minute=config.REQUESTS_PER_MINUTE
)

# Run incremental collection
stats = collector.collect_incremental_news(
    max_tickers=config.NEWS_MAX_TICKERS,
    days_lookback=config.NEWS_DAYS_LOOKBACK
)

print(f"Collected {stats['total_articles_stored']} new articles")
```

## Database Schema

### Tables

#### `polygon_news_articles`
Main news article storage with comprehensive metadata.

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key |
| `polygon_id` | String(100) | Unique Polygon article ID |
| `title` | String(1000) | Article title |
| `description` | Text | Article description |
| `article_url` | String(2000) | Original article URL |
| `amp_url` | String(2000) | AMP version URL |
| `image_url` | String(2000) | Article image URL |
| `author` | String(200) | Article author |
| `published_utc` | DateTime | Publication timestamp |
| `publisher_name` | String(200) | Publisher name |
| `publisher_homepage_url` | String(500) | Publisher homepage |
| `publisher_logo_url` | String(500) | Publisher logo |
| `publisher_favicon_url` | String(500) | Publisher favicon |
| `keywords` | ARRAY(String) | Article keywords |
| `quality_score` | Float | Content quality score |
| `relevance_score` | Float | Relevance score |
| `created_at` | DateTime | Record creation time |
| `updated_at` | DateTime | Last update time |
| `is_processed` | Boolean | Processing status |
| `processing_errors` | Text | Processing error details |

#### `polygon_news_tickers`
Association between articles and stock tickers.

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key |
| `article_id` | Integer | Foreign key to articles |
| `ticker` | String(10) | Stock ticker symbol |
| `created_at` | DateTime | Record creation time |

#### `polygon_news_insights`
Sentiment analysis and insights from Polygon API.

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key |
| `article_id` | Integer | Foreign key to articles |
| `sentiment` | String(20) | Sentiment (positive/negative/neutral) |
| `sentiment_reasoning` | Text | Sentiment reasoning |
| `insight_type` | String(50) | Type of insight |
| `insight_value` | Text | Insight value |
| `confidence_score` | Float | Confidence score |
| `created_at` | DateTime | Record creation time |

### Indexes

- `polygon_news_articles.polygon_id` (unique)
- `polygon_news_articles.published_utc`
- `polygon_news_articles.publisher_name`
- `polygon_news_articles.is_processed`
- `polygon_news_tickers.ticker`
- `polygon_news_insights.sentiment`
- `polygon_news_insights.insight_type`

## API Reference

### PolygonNewsCollector

Main orchestrator class for news collection operations.

#### Methods

##### `collect_incremental_news(max_tickers, days_lookback)`
Collect news incrementally from the last stored date.

**Parameters:**
- `max_tickers` (int): Maximum number of tickers to process
- `days_lookback` (int): Days to look back for new tickers

**Returns:**
- Dictionary with collection statistics

##### `collect_historical_news(start_date, end_date, max_tickers)`
Collect historical news for a specific date range.

**Parameters:**
- `start_date` (datetime): Start date for collection
- `end_date` (datetime): End date for collection
- `max_tickers` (int): Maximum number of tickers to process

**Returns:**
- Dictionary with collection statistics

##### `collect_targeted_news(tickers, start_date, end_date, limit_per_ticker)`
Collect news for specific tickers and date range.

**Parameters:**
- `tickers` (List[str]): List of ticker symbols
- `start_date` (datetime): Start date for collection
- `end_date` (datetime): End date for collection
- `limit_per_ticker` (int): Maximum articles per ticker

**Returns:**
- Dictionary with collection statistics

##### `get_collection_status()`
Get current system status and health information.

**Returns:**
- Dictionary with system status, latest dates, and statistics

##### `cleanup_old_data(retention_days)`
Clean up old news data beyond retention period.

**Parameters:**
- `retention_days` (int, optional): Retention period in days

**Returns:**
- Number of articles deleted

### PolygonNewsStorage

Database storage operations for news data.

#### Methods

##### `store_article(article_data)`
Store a single article with upsert logic.

##### `store_articles_batch(articles_data)`
Store multiple articles in batch with transaction management.

##### `get_latest_date_for_ticker(ticker)`
Get the latest article date for a specific ticker.

##### `get_articles_for_ticker(ticker, start_date, end_date, limit)`
Get articles for a specific ticker within date range.

##### `cleanup_old_articles(retention_days)`
Remove articles older than retention period.

### PolygonNewsClient

Specialized client for Polygon.io News API.

#### Methods

##### `get_news_for_ticker(ticker, published_utc_gte, published_utc_lte, order, limit, sort)`
Get news articles for a specific ticker.

##### `get_news_for_multiple_tickers(tickers, published_utc_gte, published_utc_lte, order, limit_per_ticker)`
Get news articles for multiple tickers.

##### `get_recent_market_news(days_back, major_tickers, limit)`
Get recent market news across major tickers.

## Examples

### Example 1: Basic Incremental Collection

```python
from src.data_collector.polygon_news.news_collector import main

# Simple incremental collection
success = main()
if success:
    print("News collection completed successfully")
else:
    print("News collection failed")
```

### Example 2: Custom Collection Parameters

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.data_collector.polygon_news import PolygonNewsCollector
from src.data_collector.config import config

# Setup
engine = create_engine(config.database_url)
Session = sessionmaker(bind=engine)
session = Session()

# Initialize with custom parameters
collector = PolygonNewsCollector(
    db_session=session,
    polygon_api_key=config.API_KEY,
    requests_per_minute=10  # Higher rate for paid plans
)

# Custom incremental collection
stats = collector.collect_incremental_news(
    max_tickers=50,
    days_lookback=14
)

print(f"Collection Results:")
print(f"- API calls: {stats['total_api_calls']}")
print(f"- Articles stored: {stats['total_articles_stored']}")
print(f"- Failed tickers: {len(stats['failed_tickers'])}")
```

### Example 3: Historical Backfill

```python
from datetime import datetime, timedelta

# Historical backfill for last 6 months
end_date = datetime.now()
start_date = end_date - timedelta(days=180)

stats = collector.collect_historical_news(
    start_date=start_date,
    end_date=end_date,
    max_tickers=100
)

print(f"Historical collection completed: {stats}")
```

### Example 4: Targeted Collection

```python
# Collect news for specific tickers
target_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

stats = collector.collect_targeted_news(
    tickers=target_tickers,
    start_date=start_date,
    end_date=end_date,
    limit_per_ticker=100
)

print(f"Targeted collection for {target_tickers}: {stats}")
```

### Example 5: Data Analysis

```python
from src.data_collector.polygon_news.storage import PolygonNewsStorage

# Get storage instance
storage = PolygonNewsStorage(session)

# Get articles for a specific ticker
articles = storage.get_articles_for_ticker(
    ticker='AAPL',
    start_date=datetime.now() - timedelta(days=30),
    limit=50
)

print(f"Found {len(articles)} articles for AAPL")

# Get statistics
stats = storage.get_article_statistics(
    start_date=datetime.now() - timedelta(days=30)
)

print(f"Statistics: {stats}")
```

## Monitoring & Maintenance

### Health Checks

```python
# Check system health
status = collector.get_collection_status()

print(f"System Status: {status['status']}")
print(f"Latest Article: {status.get('latest_article_date')}")
print(f"Database Health: {status.get('database_health')}")
```

### Data Cleanup

```python
# Clean up old data (automatic with retention period)
deleted_count = collector.cleanup_old_data()
print(f"Cleaned up {deleted_count} old articles")

# Custom retention period
deleted_count = collector.cleanup_old_data(retention_days=365)
print(f"Cleaned up {deleted_count} articles older than 1 year")
```

### Performance Monitoring

```python
# Monitor collection performance
stats = collector.collect_incremental_news(max_tickers=100)

print(f"Performance Metrics:")
print(f"- Duration: {stats.get('duration')}")
print(f"- Articles per minute: {stats['total_articles_stored'] / (stats['duration'].total_seconds() / 60)}")
print(f"- Success rate: {(stats['total_articles_stored'] / stats['total_articles_fetched']) * 100:.1f}%")
```

## Troubleshooting

### Common Issues

#### 1. API Rate Limiting
**Problem**: Getting rate limit errors from Polygon API.

**Solution**:
```python
# Reduce requests per minute
collector = PolygonNewsCollector(
    db_session=session,
    polygon_api_key=config.API_KEY,
    requests_per_minute=3  # Reduce from default 5
)
```

#### 2. Database Connection Issues
**Problem**: Database connection failures.

**Solution**:
```bash
# Check database URL
echo $DATABASE_URL

# Test connection
psql $DATABASE_URL -c "SELECT 1;"
```

#### 3. Missing Articles
**Problem**: Expected articles not being collected.

**Solution**:
```python
# Check ticker prioritization
from src.data_collector.polygon_news.ticker_integration import NewsTickerIntegration

ticker_integration = NewsTickerIntegration()
prioritized = ticker_integration.get_prioritized_tickers(max_tickers=10)
print(f"Top priority tickers: {[t['ticker'] for t in prioritized[:10]]}")
```

#### 4. Memory Issues
**Problem**: High memory usage during collection.

**Solution**:
```python
# Reduce batch size
collector.collect_incremental_news(
    max_tickers=25,  # Reduce from default
    days_lookback=3   # Reduce lookback period
)
```

### Logging

All components use centralized logging. Check logs for detailed error information:

```python
from src.logger import get_polygon_logger

logger = get_polygon_logger(__name__)
logger.info("Custom log message")
```

### Error Recovery

The system includes automatic error recovery:

- **Failed API calls**: Automatically retried with exponential backoff
- **Database errors**: Transactions rolled back, errors logged
- **Processing errors**: Individual articles skipped, collection continues
- **Network issues**: Adaptive rate limiting and retry mechanisms

## Performance Considerations

### Optimization Tips

1. **Batch Size**: Adjust `NEWS_BATCH_SIZE` based on available memory
2. **Rate Limiting**: Increase `REQUESTS_PER_MINUTE` for paid Polygon plans
3. **Ticker Selection**: Use `max_tickers` parameter to control scope
4. **Date Ranges**: Limit date ranges for large historical collections
5. **Database Indexing**: Ensure proper indexes for query performance

### Scaling

For high-volume deployments:

1. **Database**: Use connection pooling and read replicas
2. **API**: Distribute across multiple API keys
3. **Processing**: Implement parallel processing for multiple tickers
4. **Storage**: Consider partitioning large tables by date

## Integration

### With Existing Systems

The news collector integrates seamlessly with:

- **Ticker Manager**: Automatic ticker prioritization
- **Data Pipeline**: Shared configuration and logging
- **Database**: Uses existing database infrastructure
- **Monitoring**: Centralized logging and health checks

### Custom Extensions

Extend functionality by:

1. **Custom Processors**: Add new content processing logic
2. **Additional Validators**: Implement custom quality checks
3. **Enhanced Storage**: Add custom storage backends
4. **Monitoring Integration**: Connect to external monitoring systems

## Conclusion

The Polygon News Collector provides a robust, scalable solution for financial news data collection. With its comprehensive feature set, intelligent design, and seamless integration capabilities, it serves as a reliable foundation for news-driven financial analysis and research.

For additional support or feature requests, refer to the codebase documentation or contact the development team.