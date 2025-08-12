# 🚀 Installation & Setup

This section covers everything needed to install and configure StockPredictor V1 for development or production use.

## 📄 Documents

### [Installation Guide](./INSTALL.md)
Complete setup instructions including:
- **Prerequisites**: Python 3.12+, PostgreSQL, Polygon.io API key
- **Installation Methods**: Development, regular, and virtual environment options
- **Configuration**: Environment variables and database setup
- **Usage Examples**: Quick tests and basic functionality
- **Troubleshooting**: Common issues and solutions
- **Development Workflow**: Code formatting, testing, and package structure

## 🛠️ Quick Setup

```bash
# 1. Navigate to project directory
cd StockPredictor_V1

# 2. Install in development mode
uv pip install -e .

# 3. Set up environment variables (Windows cmd)
copy .environment.example .env
# Then edit .env with your API keys and settings

# 4. Quick health check (Polygon API client)
uv run python -c "from src.data_collector.polygon_data.client import PolygonDataClient; print(PolygonDataClient().health_check())"
```

## ⚙️ Requirements

### System Requirements
- **Python**: 3.12 or higher
- **Package Manager**: `uv` (recommended)
- **Database**: PostgreSQL 12+ (required for data storage; setup not covered here)
- **API Access**: Polygon.io API key
 - **Optional (GPU)**: PyTorch with CUDA 12.8 for neural network models on Windows

### Environment Variables
```bash
POLYGON_API_KEY=your_api_key_here
DB_HOST=localhost
DB_PORT=5432
DB_NAME=stock_predictor
DB_USER=postgres
DB_PASSWORD=your_password
```

## 🔧 Development Setup

For active development:
1. Install development dependencies: `uv pip install -e ".[dev]"`
2. Configure IDE with project structure
3. Run tests to verify setup: `uv run pytest -q`
4. Use the centralized logger in `src/utils/logger.py` for any logging needs

## 🔄 Next Steps

After installation:
1. Review [Data Collection](../03-data-collection/) to set up data sources
2. Configure [Feature Engineering](../04-feature-engineering/) parameters
3. Explore [Model Training](../05-model-training/) capabilities 