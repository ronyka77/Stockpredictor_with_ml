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

# 3. Set up environment variables
cp .environment.example .env
# Edit .env with your API keys

# 4. Test installation
python run_polygon_examples.py
```

## ⚙️ Requirements

### System Requirements
- **Python**: 3.12 or higher
- **Package Manager**: `uv` (recommended)
- **Database**: PostgreSQL 12+
- **API Access**: Polygon.io API key

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
2. Set up pre-commit hooks for code quality
3. Configure IDE with project structure
4. Run tests to verify setup

## 🔄 Next Steps

After installation:
1. Review [Data Collection](../03-data-collection/) to set up data sources
2. Configure [Feature Engineering](../04-feature-engineering/) parameters
3. Explore [Model Training](../05-model-training/) capabilities 