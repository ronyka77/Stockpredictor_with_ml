# 📋 Project Overview

This section provides high-level information about the StockPredictor V1 project, including business objectives, value proposition, and strategic approach.

## 📄 Documents

### [Business Overview](./business_overview.md)
Comprehensive overview of the project including:
- **Project Purpose**: Value proposition and strategic goals
- **Methodology**: Three-tier prediction framework (10, 30, 90 days)
- **Data Integration**: Multi-dimensional market view combining:
  - Historical price patterns
  - Technical indicators  
  - Fundamental company metrics
  - Real-time news sentiment
- **Implementation Architecture**: PostgreSQL + Parquet storage
- **Risk Management**: Validation and backtesting approach
- **Success Metrics**: Custom 60% accuracy target

## 🎯 Key Takeaways

- **Target Accuracy**: 60% using custom metric (directional + magnitude)
- **Portfolio**: 50-100 carefully selected US stocks
- **Prediction Horizons**: 10, 30, and 90-day forecasts
- **Competitive Advantage**: Multi-dimensional data integration
- **Architecture**: Modular, scalable design for continuous improvement

## 🔄 Next Steps

After reading this section:
1. Proceed to [Installation & Setup](../02-installation-setup/) to get started
2. Review [Data Collection](../03-data-collection/) to understand data sources
3. Explore [Feature Engineering](../04-feature-engineering/) for analysis capabilities 