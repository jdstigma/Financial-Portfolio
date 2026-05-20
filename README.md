# Financial Portfolio 📊

A comprehensive stock market analysis and forecasting portfolio featuring S&P 500 data stored in GitHub LFS, with advanced regression analysis and multi-method forecasting.

## 🎯 Features

### Data Management
- ✅ **10 years of S&P 500 data** - Historical price and volume data
- ✅ **50 major S&P 500 stocks** - Individual company performance
- ✅ **200-500 MB datasets** - Stored in Git LFS for efficiency
- ✅ **Market aggregates** - Daily market-wide statistics
- ✅ **Correlation analysis** - Stock correlations matrices

### Regression Analysis
- 📈 **Linear Regression** - Trend analysis with R² and RMSE metrics
- 📉 **Log Transformation (Log₁₀ & Natural Log)** - For exponential growth patterns
- 🔗 **Multiple Regression** - Multi-feature analysis
- 📊 **Statistical Diagnostics** - Residuals, Q-Q plots, distribution analysis

### Forecasting Methods
1. **Linear Trend** - Simple trend extrapolation
2. **Exponential Smoothing** - Captures level & trend components
3. **ARIMA** - AutoRegressive Integrated Moving Average
4. **Random Forest** - ML ensemble method
5. **Gradient Boosting** - Advanced ML ensemble

**Metrics for all models:** MAE, RMSE, MAPE

## 📁 Project Structure

```
Financial-Portfolio/
├── download_data.py              # Download S&P 500 data from Yahoo Finance
├── regression_analysis.py        # Regression analysis module
├── forecasting.py               # Multi-method forecasting module
├── requirements.txt             # Python dependencies
├── .gitattributes              # Git LFS configuration
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
│
├── data/                       # Data directory (Git LFS tracked)
│   ├── sp500_index.csv        # S&P 500 index data
│   ├── sp500_stocks.csv       # Individual stock data
│   ├── market_daily_aggregate.csv
│   └── stock_correlations.csv
│
└── results/                    # Analysis results and plots
    ├── linear_regression_analysis.png
    ├── log_regression_analysis.png
    ├── multiple_regression_analysis.png
    ├── forecast_*.png
    └── forecasting_comparison.png
```

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install Git LFS
git lfs install

# Clone repository
git clone https://github.com/jdstigma/Financial-Portfolio.git
cd Financial-Portfolio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

```bash
python download_data.py
```

This downloads 10 years of data for S&P 500 companies and creates CSV files in `data/` directory.

**Expected output size:** 200-500 MB

### 3. Run Analysis

```bash
# Regression Analysis
python regression_analysis.py

# Forecasting
python forecasting.py
```

### 4. Push to GitHub with Git LFS

```bash
# Add and commit data
git add data/
git commit -m "Add S&P 500 stock market dataset"

# Push to GitHub
git push origin main
```

## 📊 Data Specifications

| Aspect | Details |
|--------|----------|
| **Time Period** | 10 years of historical data |
| **Companies** | Top 50 S&P 500 companies |
| **Data Points** | OHLCV + Adjusted Close |
| **Frequency** | Daily |
| **Total Records** | ~125,000+ stock records + index data |
| **Format** | CSV (Git LFS tracked) |
| **Update Frequency** | As needed (manual) |

## 💡 Usage Examples

### Load and Analyze a Stock

```python
import pandas as pd
from regression_analysis import RegressionAnalyzer

# Load data
data = pd.read_csv('data/sp500_stocks.csv')
stock_data = data[data['Ticker'] == 'AAPL'][['Adj Close', 'Volume']]

# Perform regression analysis
analyzer = RegressionAnalyzer(stock_data, ticker='AAPL')
analyzer.linear_regression()
analyzer.log_regression(base=10)
analyzer.multiple_regression()
```

### Forecast Stock Prices

```python
from forecasting import StockForecaster

# Load data
prices = data[data['Ticker'] == 'AAPL']['Adj Close']

# Create forecasts
forecaster = StockForecaster(prices, ticker='AAPL')
forecaster.linear_trend_forecast(periods=30)
forecaster.arima_forecast(order=(5,1,2), periods=30)
forecaster.gradient_boosting_forecast(lags=5, periods=30)
forecaster.forecast_comparison()
```

### Calculate Correlations

```python
# Load correlation matrix
corr = pd.read_csv('data/stock_correlations.csv', index_col=0)

# Find most correlated pairs
print(corr.unstack().nlargest(10))
```

## 📈 Analysis Output

### Regression Analysis Generates:
- **Linear Regression Plot** - Price trend with fitted line
- **Residuals Analysis** - Diagnostic plots for model validation
- **Q-Q Plot** - Normality of residuals
- **Log Regression Results** - Exponential growth patterns
- **Multiple Regression** - Feature importance visualization

### Forecasting Generates:
- **Individual Forecast Plots** - For each method
- **Comparison Charts** - MAE, RMSE, MAPE comparison
- **Prediction Metrics** - Accuracy statistics

## 🔧 Technologies Used

| Component | Technology |
|-----------|------------|
| **Data Collection** | yfinance |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn, plotly |
| **Regression** | scikit-learn |
| **Time Series** | statsmodels |
| **ML Forecasting** | scikit-learn (RF, XGBoost) |
| **Storage** | Git LFS |

## ⚙️ Configuration

### Git LFS Setup

Git LFS is pre-configured in `.gitattributes`. CSV files are automatically tracked when pushed.

```bash
# Check LFS status
git lfs ls-files

# Verify LFS installation
git lfs version
```

### Storage Limits

- **Free Tier:** 1 GB storage, 1 GB/month bandwidth
- **Pro Tier:** Increased storage/bandwidth (see GitHub Pricing)

## 📚 Key Insights & Metrics

### Linear Regression
- **Slope**: Daily price change
- **R² Score**: Goodness of fit (0-1)
- **RMSE**: Prediction error in dollars
- **Annual Rate**: Projected yearly growth

### Log Regression
- **Log₁₀ Transformation**: For percentage changes
- **Natural Log**: For continuous compounding
- **Daily % Change**: Compound growth rate
- **Exponential Model**: Better for stocks with exponential growth

### Multiple Regression
- **Days**: Time trend coefficient
- **Price_Lag1**: Momentum indicator
- **Log_Price**: Current price level
- **Adjusted R²**: Penalized R² for multiple features

### Forecasting Accuracy

| Method | Best For | Speed |
|--------|----------|-------|
| Linear Trend | Stable trends | ⚡⚡⚡ Fast |
| Exponential Smoothing | Trending data | ⚡⚡ Medium |
| ARIMA | Complex patterns | ⚡ Slow |
| Random Forest | Non-linear | ⚡ Slow |
| Gradient Boosting | High accuracy | ⚡ Slow |

## 🎓 Portfolio Ideas & Applications

1. **Stock Performance Dashboard** - Real-time analysis visualization
2. **Portfolio Optimization** - Efficient frontier analysis
3. **Risk Assessment** - Volatility and correlation analysis
4. **Trading Signals** - Based on technical indicators
5. **Sector Analysis** - Compare sector performance
6. **Machine Learning** - Train predictive models
7. **Backtesting** - Test trading strategies
8. **Risk Metrics** - Sharpe ratio, beta, VaR

## ⚠️ Important Notes

- **First Run**: Data download takes 5-10 minutes
- **Internet Required**: For initial data download
- **Yahoo Finance Data**: Historical (no real-time updates)
- **LFS Limits**: Monitor storage usage on free tier
- **Data Freshness**: Manually update data as needed

## 🔗 Resources

- [yfinance Documentation](https://github.com/ranaroussi/yfinance)
- [Git LFS Guide](https://git-lfs.github.com/)
- [S&P 500 Index Info](https://en.wikipedia.org/wiki/S%26P_500)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [scikit-learn Guide](https://scikit-learn.org/)
- [statsmodels](https://www.statsmodels.org/)

## 📞 Support & Contributions

For questions or issues:
1. Check existing documentation
2. Review generated plots and results
3. Verify data files exist in `data/` directory
4. Ensure all dependencies are installed

## 📝 License

This project is open source and available under the MIT License.

---

**Last Updated:** May 2026  
**Author:** jdstigma  
**Repository:** [Financial-Portfolio](https://github.com/jdstigma/Financial-Portfolio)
