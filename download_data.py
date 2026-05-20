"""
Financial Data Download Module

Downloads S&P 500 stock data from Yahoo Finance and creates
large CSV datasets for portfolio analysis.

Uses yf.Ticker().history() instead of yf.download() to avoid
MultiIndex / column-rename bugs present in all yfinance versions.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

Data_DIR = Path('data')
Data_DIR.mkdir(exist_ok=True)

TOP_SP500_STOCKS = [
    'AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',
    'JNJ', 'V', 'WMT', 'JPM', 'PG', 'MA', 'HD', 'MCD',
    'NFLX', 'INTC', 'AMD', 'IBM', 'CSCO', 'PEP', 'KO', 'COST',
    'ABBV', 'LLY', 'MRK', 'PFE', 'TMO', 'ABT', 'ADBE', 'CRM', 'ORCL',
    'DIS', 'QCOM', 'AVGO', 'ACN', 'ASML', 'NKE', 'CMCSA', 'TXN', 'WDAY',
    'NOW', 'BKNG', 'MSCI', 'FISV', 'ROP', 'BA', 'PYPL'
]


def fetch_ticker(symbol, start_date, end_date):
    try:
        df = yf.Ticker(symbol).history(start=start_date, end=end_date, auto_adjust=True)
        if df is None or len(df) == 0:
            return None
        df = df.reset_index()
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
        if 'Close' in df.columns:
            df = df.rename(columns={'Close': 'Adj Close'})
        keep = ['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume']
        if any(c not in df.columns for c in keep):
            return None
        df = df[keep].copy()
        df['Ticker'] = symbol
        df = df.dropna(subset=['Adj Close'])
        if hasattr(df['Date'], 'dt') and df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
        return df if len(df) > 0 else None
    except Exception:
        return None


class FinancialDataDownloader:

    def __init__(self, start_date=None, end_date=None, lookback_years=10):
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            self.start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime('%Y-%m-%d')
        else:
            self.start_date = start_date
        print(f"Date Range: {self.start_date} to {self.end_date}")

    def download_sp500_index(self):
        print("\n" + "=" * 60)
        print("Downloading S&P 500 Index Data...")
        print("=" * 60)
        result = fetch_ticker('^GSPC', self.start_date, self.end_date)
        if result is not None:
            print(f"Downloaded {len(result)} records")
            print(f"  Date Range: {result['Date'].min()} to {result['Date'].max()}")
            print(f"  Price Range: ${result['Adj Close'].min():.2f} - ${result['Adj Close'].max():.2f}")
            return result
        print("Failed to download S&P 500 index data")
        return None

    def download_stock_data(self, tickers=None):
        if tickers is None:
            tickers = TOP_SP500_STOCKS
        print("\n" + "=" * 60)
        print(f"Downloading {len(tickers)} Stock Data...")
        print("=" * 60)
        all_data, failed = [], []
        for ticker in tqdm(tickers, desc="Download Progress"):
            result = fetch_ticker(ticker, self.start_date, self.end_date)
            if result is not None:
                all_data.append(result)
            else:
                failed.append(ticker)
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            print(f"\nDownloaded {len(combined)} records for {combined['Ticker'].nunique()} stocks")
            if failed:
                print(f"Failed ({len(failed)}): {failed}")
            return combined
        print("\nNo data downloaded")
        return None

    def calculate_correlations(self, stock_data):
        print("\n" + "=" * 60)
        print("Calculating Stock Correlations...")
        print("=" * 60)
        try:
            pivot = stock_data.pivot_table(index='Date', columns='Ticker', values='Adj Close')
            corr = pivot.corr()
            print(f"Calculated correlations for {len(corr)} stocks")
            return corr
        except Exception as e:
            print(f"Error: {e}")
            return None

    def calculate_market_aggregate(self, stock_data):
        print("\n" + "=" * 60)
        print("Calculating Market Aggregates...")
        print("=" * 60)
        try:
            agg = stock_data.groupby('Date').agg(
                Avg_Price=('Adj Close', 'mean'),
                Median_Price=('Adj Close', 'median'),
                Std_Price=('Adj Close', 'std'),
                Min_Price=('Adj Close', 'min'),
                Max_Price=('Adj Close', 'max'),
                Total_Volume=('Volume', 'sum'),
                Avg_Volume=('Volume', 'mean'),
                Stock_Count=('Ticker', 'count')
            ).reset_index()
            print(f"Calculated aggregates for {len(agg)} days")
            return agg
        except Exception as e:
            print(f"Error: {e}")
            return None

    def save_data(self, sp500_data, stock_data, correlations, aggregates):
        print("\n" + "=" * 60)
        print("Saving Data to Files...")
        print("=" * 60)
        total = 0
        for df, name in [
            (sp500_data, 'sp500_index.csv'),
            (stock_data, 'sp500_stocks.csv'),
            (aggregates, 'market_daily_aggregate.csv'),
        ]:
            if df is not None and len(df) > 0:
                df.to_csv(Data_DIR / name, index=False)
                mb = df.memory_usage(deep=True).sum() / 1024 ** 2
                print(f"Saved {name}: {mb:.2f} MB")
                total += mb
        if correlations is not None and len(correlations) > 0:
            correlations.to_csv(Data_DIR / 'stock_correlations.csv')
            mb = correlations.memory_usage(deep=True).sum() / 1024 ** 2
            print(f"Saved stock_correlations.csv: {mb:.2f} MB")
            total += mb
        print(f"\nTotal: {total:.2f} MB  |  Location: {Data_DIR.absolute()}")

    def run_full_pipeline(self):
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
            print("\nNext steps:\n1. python regression_analysis.py\n2. python forecasting.py")
            return True
        print("\nPipeline failed: No stock data downloaded")
        return False


if __name__ == "__main__":
    downloader = FinancialDataDownloader(lookback_years=10)
    success = downloader.run_full_pipeline()
    print("\nAll data successfully downloaded and saved!" if success else "\nDownload failed.")