"""
Financial Data Download Module

Downloads S&P 500 stock data from Yahoo Finance and creates
large CSV datasets for portfolio analysis.

Fixes applied:
- yfinance no longer returns 'Adj Close' as a separate column; 'Close' is
  already adjusted. We now download 'Close' and rename it to 'Adj Close'
    for downstream compatibility.
    - Removed the duplicate 'AVGO' ticker from TOP_SP500_STOCKS.
    - Updated the required-column guard from < 7 to < 6 to match the new schema.
    """

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Create data directory
Data_DIR = Path('data')
Data_DIR.mkdir(exist_ok=True)

# Top 50 S&P 500 Companies (duplicates removed)
TOP_SP500_STOCKS = [
        'AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',
        'JNJ', 'V', 'WMT', 'JPM', 'PG', 'MA', 'HD', 'MCD',
        'NFLX', 'INTC', 'AMD', 'IBM', 'CSCO', 'PEP', 'KO', 'COST',
        'ABBV', 'LLY', 'MRK', 'PFE', 'TMO', 'ABT', 'ADBE', 'CRM', 'ORCL',
        'DIS', 'QCOM', 'AVGO', 'ACN', 'ASML', 'NKE', 'CMCSA', 'TXN', 'WDAY',
        'NOW', 'BKNG', 'MSCI', 'FISV', 'ROP', 'BA', 'PYPL'
]


class FinancialDataDownloader:
        """Download and process financial data from Yahoo Finance."""

    def __init__(self, start_date=None, end_date=None, lookback_years=10):
                self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')

        if start_date is None:
                        start_datetime = datetime.now() - timedelta(days=365 * lookback_years)
                        self.start_date = start_datetime.strftime('%Y-%m-%d')
else:
                self.start_date = start_date

        print(f"Date Range: {self.start_date} to {self.end_date}")

    def _process_downloaded_data(self, data, ticker):
                """
                        Process downloaded data and extract required columns.

                                In modern yfinance (>=0.2.x) 'Adj Close' is no longer returned as a
                                        separate column — 'Close' already contains the adjusted price.  We
                                                select the six available OHLCV columns and rename 'Close' to
                                                        'Adj Close' so the rest of the pipeline stays consistent.
                                                                """
                if data is None or len(data) == 0:
                                return None

                try:
                                # Flatten MultiIndex columns produced when group_by='ticker' is used
                                if isinstance(data.columns, pd.MultiIndex):
                                                    data.columns = data.columns.get_level_values(0)

                                # Reset index so Date becomes a regular column
                                df = data.reset_index()

                    # Modern yfinance: Open, High, Low, Close, Volume (no 'Adj Close')
                                required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

                    available_cols = [col for col in required_cols if col in df.columns]

            if len(available_cols) < 6:
                                return None

            df = df[required_cols].copy()

            # Rename 'Close' to 'Adj Close' for downstream compatibility
            df.rename(columns={'Close': 'Adj Close'}, inplace=True)

            df['Ticker'] = ticker

            return df

except Exception as e:
            print(f"  Processing error for {ticker}: {e}")
            return None

    def download_sp500_index(self):
                """Download S&P 500 index data."""
        print("\n" + "=" * 60)
        print("Downloading S&P 500 Index Data...")
        print("=" * 60)

        try:
                        sp500 = yf.download('^GSPC', start=self.start_date, end=self.end_date,
                                                                            progress=False, timeout=30)

            if sp500 is None or len(sp500) == 0:
                                print("x No data returned")
                                return None

            result = self._process_downloaded_data(sp500, '^GSPC')

            if result is not None:
                                print(f"v Downloaded {len(result)} records")
                                print(f"  Date Range: {result['Date'].min().date()} to {result['Date'].max().date()}")
                                print(f"  Price Range: ${result['Adj Close'].min():.2f} - ${result['Adj Close'].max():.2f}")
                                return result
else:
                print("x Error processing data")
                return None

except Exception as e:
            print(f"x Error: {str(e)[:100]}")
            return None

    def download_stock_data(self, tickers=None):
                """Download historical data for multiple stocks."""
        if tickers is None:
                        tickers = TOP_SP500_STOCKS

        print("\n" + "=" * 60)
        print(f"Downloading {len(tickers)} Stock Data...")
        print("=" * 60)

        all_data = []
        failed_tickers = []

        for ticker in tqdm(tickers, desc="Download Progress"):
                        try:
                                            ticker_data = yf.download(ticker, start=self.start_date, end=self.end_date,
                                                                                                                progress=False, timeout=30)

                            result = self._process_downloaded_data(ticker_data, ticker)

                if result is not None and len(result) > 0:
                                        all_data.append(result)
else:
                    failed_tickers.append(ticker)

except Exception as e:
                failed_tickers.append(ticker)

        if all_data:
                        combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\nv Downloaded {len(combined_df)} records for {combined_df['Ticker'].nunique()} stocks")
            if failed_tickers:
                                print(f"x Failed: {len(failed_tickers)} tickers: {failed_tickers}")
                            return combined_df
else:
            print("\nx No data downloaded")
            return None

    def calculate_correlations(self, stock_data):
                """Calculate correlation matrix between stocks."""
        print("\n" + "=" * 60)
        print("Calculating Stock Correlations...")
        print("=" * 60)

        try:
                        pivot_data = stock_data.pivot(index='Date', columns='Ticker', values='Adj Close')
            correlations = pivot_data.corr()

            print(f"v Calculated correlations for {len(correlations)} stocks")
            return correlations

except Exception as e:
            print(f"x Error: {e}")
            return None

    def calculate_market_aggregate(self, stock_data):
                """Calculate daily market aggregate statistics."""
        print("\n" + "=" * 60)
        print("Calculating Market Aggregates...")
        print("=" * 60)

        try:
                        daily_agg = stock_data.groupby('Date').agg({
                            'Adj Close': ['mean', 'median', 'std', 'min', 'max'],
                            'Volume': ['sum', 'mean'],
                            'Ticker': 'count'
        }).reset_index()

            daily_agg.columns = ['Date', 'Avg_Price', 'Median_Price', 'Std_Price',
                                                                   'Min_Price', 'Max_Price', 'Total_Volume', 'Avg_Volume',
                                                                   'Stock_Count']

            print(f"v Calculated aggregates for {len(daily_agg)} days")
            return daily_agg

except Exception as e:
            print(f"x Error: {e}")
            return None

    def save_data(self, sp500_data, stock_data, correlations, aggregates):
                """Save all data to CSV files."""
        print("\n" + "=" * 60)
        print("Saving Data to Files...")
        print("=" * 60)

        files_saved = []

        if sp500_data is not None and len(sp500_data) > 0:
                        index_file = Data_DIR / 'sp500_index.csv'
            sp500_data.to_csv(index_file, index=False)
            size_mb = sp500_data.memory_usage(deep=True).sum() / 1024 ** 2
            files_saved.append(('sp500_index.csv', size_mb))

        if stock_data is not None and len(stock_data) > 0:
                        stocks_file = Data_DIR / 'sp500_stocks.csv'
            stock_data.to_csv(stocks_file, index=False)
            size_mb = stock_data.memory_usage(deep=True).sum() / 1024 ** 2
            files_saved.append(('sp500_stocks.csv', size_mb))

        if correlations is not None and len(correlations) > 0:
                        corr_file = Data_DIR / 'stock_correlations.csv'
            correlations.to_csv(corr_file)
            size_mb = correlations.memory_usage(deep=True).sum() / 1024 ** 2
            files_saved.append(('stock_correlations.csv', size_mb))

        if aggregates is not None and len(aggregates) > 0:
                        agg_file = Data_DIR / 'market_daily_aggregate.csv'
            aggregates.to_csv(agg_file, index=False)
            size_mb = aggregates.memory_usage(deep=True).sum() / 1024 ** 2
            files_saved.append(('market_daily_aggregate.csv', size_mb))

        total_size = 0
        for filename, size_mb in files_saved:
                        print(f"v Saved {filename}: {size_mb:.2f} MB")
            total_size += size_mb

        print(f"\n{'=' * 60}")
        print(f"Total data size: {total_size:.2f} MB")
        print(f"Location: {Data_DIR.absolute()}")
        print(f"{'=' * 60}")

    def run_full_pipeline(self):
                """Run the complete download and processing pipeline."""
        print("\n" + "#" * 60)
        print("# FINANCIAL DATA DOWNLOAD PIPELINE")
        print("#" * 60)

        sp500_data = self.download_sp500_index()
        stock_data = self.download_stock_data()

        if stock_data is not None and len(stock_data) > 0:
                        correlations = self.calculate_correlations(stock_data)
            aggregates = self.calculate_market_aggregate(stock_data)

            self.save_data(sp500_data, stock_data, correlations, aggregates)

            print("\n" + "#" * 60)
            print("# PIPELINE COMPLETE!")
            print("#" * 60)
            print("\nNext steps:")
            print("1. python regression_analysis.py")
            print("2. python forecasting.py")
            return True
else:
            print("\nx Pipeline failed: No stock data downloaded")
            return False


if __name__ == "__main__":
        downloader = FinancialDataDownloader(lookback_years=10)
    success = downloader.run_full_pipeline()

    if success:
                print("\nv All data successfully downloaded and saved!")
else:
        print("\nx Download failed. Check your internet connection and try again.")
