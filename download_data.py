"""
Financial Data Download Module

Downloads S&P 500 stock data from Yahoo Finance and creates
large CSV datasets for portfolio analysis.

Uses yf.Ticker().history() instead of yf.download() to avoid
MultiIndex / column-rename bugs present in all yfinance versions.
history() always returns a plain, flat DataFrame with consistent
column names regardless of yfinance version.
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

# Top 49 S&P 500 Companies (duplicate AVGO removed)
TOP_SP500_STOCKS = [
      'AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',
      'JNJ', 'V', 'WMT', 'JPM', 'PG', 'MA', 'HD', 'MCD',
      'NFLX', 'INTC', 'AMD', 'IBM', 'CSCO', 'PEP', 'KO', 'COST',
      'ABBV', 'LLY', 'MRK', 'PFE', 'TMO', 'ABT', 'ADBE', 'CRM', 'ORCL',
      'DIS', 'QCOM', 'AVGO', 'ACN', 'ASML', 'NKE', 'CMCSA', 'TXN', 'WDAY',
      'NOW', 'BKNG', 'MSCI', 'FISV', 'ROP', 'BA', 'PYPL'
]


def fetch_ticker(symbol, start_date, end_date):
      """
          Fetch OHLCV data for a single symbol using yf.Ticker.history().

              history() returns a flat DataFrame with columns:
                      Open, High, Low, Close, Volume, Dividends, Stock Splits
                          The 'Close' column is already split/dividend adjusted.
                              We rename it to 'Adj Close' for downstream compatibility.

                                  Returns a tidy DataFrame with columns:
                                          Date, Open, High, Low, Adj Close, Volume, Ticker
                                              or None on failure.
                                                  """
      try:
                ticker_obj = yf.Ticker(symbol)
                df = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)

          if df is None or len(df) == 0:
                        return None

        # history() uses a DatetimeIndex named 'Date'
        df = df.reset_index()

        # Normalise date column name
        date_col = 'Date' if 'Date' in df.columns else df.columns[0]
        df = df.rename(columns={date_col: 'Date'})

        # Rename Close -> Adj Close
        if 'Close' in df.columns:
                      df = df.rename(columns={'Close': 'Adj Close'})

        # Keep only the columns we need
        keep = ['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume']
        missing = [c for c in keep if c not in df.columns]
        if missing:
                      return None

        df = df[keep].copy()
        df['Ticker'] = symbol
        df = df.dropna(subset=['Adj Close'])

        # Strip timezone from Date so CSVs stay clean
        if hasattr(df['Date'], 'dt') and df['Date'].dt.tz is not None:
                      df['Date'] = df['Date'].dt.tz_localize(None)

        return df if len(df) > 0 else None

except Exception as e:
        return None


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

    def download_sp500_index(self):
              """Download S&P 500 index data."""
              print("\n" + "=" * 60)
              print("Downloading S&P 500 Index Data...")
              print("=" * 60)

        result = fetch_ticker('^GSPC', self.start_date, self.end_date)

        if result is not None:
                      print(f"v Downloaded {len(result)} records")
                      print(f"  Date Range: {result['Date'].min()} to {result['Date'].max()}")
                      print(f"  Price Range: ${result['Adj Close'].min():.2f} - ${result['Adj Close'].max():.2f}")
                      return result
else:
            print("x Failed to download S&P 500 index data")
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
                      result = fetch_ticker(ticker, self.start_date, self.end_date)
                      if result is not None:
                                        all_data.append(result)
else:
                failed_tickers.append(ticker)

        if all_data:
                      combined_df = pd.concat(all_data, ignore_index=True)
                      print(f"\nv Downloaded {len(combined_df)} records for {combined_df['Ticker'].nunique()} stocks")
                      if failed_tickers:
                                        print(f"x Failed ({len(failed_tickers)}): {failed_tickers}")
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
                      pivot_data = stock_data.pivot_table(
                          index='Date', columns='Ticker', values='Adj Close'
        )
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
                      daily_agg = stock_data.groupby('Date').agg(
                          Avg_Price=('Adj Close', 'mean'),
                          Median_Price=('Adj Close', 'median'),
                          Std_Price=('Adj Close', 'std'),
                          Min_Price=('Adj Close', 'min'),
                          Max_Price=('Adj Close', 'max'),
                          Total_Volume=('Volume', 'sum'),
                          Avg_Volume=('Volume', 'mean'),
                          Stock_Count=('Ticker', 'count')
        ).reset_index()

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
                      path = Data_DIR / 'sp500_index.csv'
            sp500_data.to_csv(path, index=False)
            files_saved.append(('sp500_index.csv', sp500_data.memory_usage(deep=True).sum() / 1024 ** 2))

        if stock_data is not None and len(stock_data) > 0:
                      path = Data_DIR / 'sp500_stocks.csv'
            stock_data.to_csv(path, index=False)
            files_saved.append(('sp500_stocks.csv', stock_data.memory_usage(deep=True).sum() / 1024 ** 2))

        if correlations is not None and len(correlations) > 0:
                      path = Data_DIR / 'stock_correlations.csv'
            correlations.to_csv(path)
            files_saved.append(('stock_correlations.csv', correlations.memory_usage(deep=True).sum() / 1024 ** 2))

        if aggregates is not None and len(aggregates) > 0:
                      path = Data_DIR / 'market_daily_aggregate.csv'
            aggregates.to_csv(path, index=False)
            files_saved.append(('market_daily_aggregate.csv', aggregates.memory_usage(deep=True).sum() / 1024 ** 2))

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
