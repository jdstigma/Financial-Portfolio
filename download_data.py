"""
Financial Data Download Module - FIXED VERSION

Downloads S&P 500 stock data from Yahoo Finance and creates
large CSV datasets for portfolio analysis.

This version handles yfinance returning 7 columns (includes Dividends/Stock Splits).
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

# Top 50 S&P 500 Companies
TOP_SP500_STOCKS = [
    'AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',
    'JNJ', 'V', 'WMT', 'JPM', 'PG', 'MA', 'HD', 'MCD',
    'NFLX', 'INTC', 'AMD', 'IBM', 'CSCO', 'PEP', 'KO', 'COST',
    'ABBV', 'LLY', 'MRK', 'PFE', 'TMO', 'ABT', 'ADBE', 'CRM', 'ORCL',
    'DIS', 'QCOM', 'AVGO', 'ACN', 'ASML', 'NKE', 'CMCSA', 'TXN', 'WDAY',
    'NOW', 'BKNG', 'MSCI', 'FISV', 'ROP', 'BA', 'PYPL', 'AVGO'
]


class FinancialDataDownloader:
    """Download and process financial data from Yahoo Finance."""
    
    def __init__(self, start_date=None, end_date=None, lookback_years=10):
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            start_datetime = datetime.now() - timedelta(days=365*lookback_years)
            self.start_date = start_datetime.strftime('%Y-%m-%d')
        else:
            self.start_date = start_date
        
        print(f"Date Range: {self.start_date} to {self.end_date}")
    
    def _process_downloaded_data(self, data, ticker):
        """
        Process downloaded data and extract required columns.
        Handles varying column structures from yfinance.
        """
        if data is None or len(data) == 0:
            return None
        
        try:
            # Reset index to make Date a column
            df = data.reset_index()
            
            # Select only the columns we need
            # yfinance can return: Date, Open, High, Low, Close, Adj Close, Volume, [Dividends, Stock Splits]
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            
            # Filter to only columns that exist
            available_cols = [col for col in required_cols if col in df.columns]
            
            if len(available_cols) < 7:
                return None
            
            # Take only the required columns
            df = df[required_cols]
            df['Ticker'] = ticker
            
            return df
        
        except Exception as e:
            return None
    
    def download_sp500_index(self):
        """Download S&P 500 index data."""
        print("\n" + "="*60)
        print("Downloading S&P 500 Index Data...")
        print("="*60)
        
        try:
            sp500 = yf.download('^GSPC', start=self.start_date, end=self.end_date, 
                               progress=False, timeout=30)
            
            if sp500 is None or len(sp500) == 0:
                print("✗ No data returned")
                return None
            
            # Process the data
            result = self._process_downloaded_data(sp500, '^GSPC')
            
            if result is not None:
                print(f"✓ Downloaded {len(result)} records")
                print(f"  Date Range: {result['Date'].min().date()} to {result['Date'].max().date()}")
                print(f"  Price Range: ${result['Adj Close'].min():.2f} - ${result['Adj Close'].max():.2f}")
                return result
            else:
                print("✗ Error processing data")
                return None
        
        except Exception as e:
            print(f"✗ Error: {str(e)[:100]}")
            return None
    
    def download_stock_data(self, tickers=None):
        """Download historical data for multiple stocks."""
        if tickers is None:
            tickers = TOP_SP500_STOCKS
        
        print("\n" + "="*60)
        print(f"Downloading {len(tickers)} Stock Data...")
        print("="*60)
        
        all_data = []
        failed_tickers = []
        
        for ticker in tqdm(tickers, desc="Download Progress"):
            try:
                # Download single ticker
                ticker_data = yf.download(ticker, start=self.start_date, end=self.end_date,
                                         progress=False, timeout=30)
                
                # Process the data
                result = self._process_downloaded_data(ticker_data, ticker)
                
                if result is not None and len(result) > 0:
                    all_data.append(result)
                else:
                    failed_tickers.append(ticker)
            
            except Exception as e:
                failed_tickers.append(ticker)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\n✓ Downloaded {len(combined_df)} records for {combined_df['Ticker'].nunique()} stocks")
            if failed_tickers:
                print(f"✗ Failed: {len(failed_tickers)} tickers")
            return combined_df
        else:
            print("\n✗ No data downloaded")
            return None
    
    def calculate_correlations(self, stock_data):
        """Calculate correlation matrix between stocks."""
        print("\n" + "="*60)
        print("Calculating Stock Correlations...")
        print("="*60)
        
        try:
            pivot_data = stock_data.pivot(index='Date', columns='Ticker', values='Adj Close')
            correlations = pivot_data.corr()
            
            print(f"✓ Calculated correlations for {len(correlations)} stocks")
            return correlations
        
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def calculate_market_aggregate(self, stock_data):
        """Calculate daily market aggregate statistics."""
        print("\n" + "="*60)
        print("Calculating Market Aggregates...")
        print("="*60)
        
        try:
            daily_agg = stock_data.groupby('Date').agg({
                'Adj Close': ['mean', 'median', 'std', 'min', 'max'],
                'Volume': ['sum', 'mean'],
                'Ticker': 'count'
            }).reset_index()
            
            daily_agg.columns = ['Date', 'Avg_Price', 'Median_Price', 'Std_Price', 
                                'Min_Price', 'Max_Price', 'Total_Volume', 'Avg_Volume', 'Stock_Count']
            
            print(f"✓ Calculated aggregates for {len(daily_agg)} days")
            return daily_agg
        
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def save_data(self, sp500_data, stock_data, correlations, aggregates):
        """Save all data to CSV files."""
        print("\n" + "="*60)
        print("Saving Data to Files...")
        print("="*60)
        
        files_saved = []
        
        # Save S&P 500 index
        if sp500_data is not None and len(sp500_data) > 0:
            index_file = Data_DIR / 'sp500_index.csv'
            sp500_data.to_csv(index_file, index=False)
            size_mb = sp500_data.memory_usage(deep=True).sum() / 1024**2
            files_saved.append(('sp500_index.csv', size_mb))
        
        # Save stock data
        if stock_data is not None and len(stock_data) > 0:
            stocks_file = Data_DIR / 'sp500_stocks.csv'
            stock_data.to_csv(stocks_file, index=False)
            size_mb = stock_data.memory_usage(deep=True).sum() / 1024**2
            files_saved.append(('sp500_stocks.csv', size_mb))
        
        # Save correlations
        if correlations is not None and len(correlations) > 0:
            corr_file = Data_DIR / 'stock_correlations.csv'
            correlations.to_csv(corr_file)
            size_mb = correlations.memory_usage(deep=True).sum() / 1024**2
            files_saved.append(('stock_correlations.csv', size_mb))
        
        # Save aggregates
        if aggregates is not None and len(aggregates) > 0:
            agg_file = Data_DIR / 'market_daily_aggregate.csv'
            aggregates.to_csv(agg_file, index=False)
            size_mb = aggregates.memory_usage(deep=True).sum() / 1024**2
            files_saved.append(('market_daily_aggregate.csv', size_mb))
        
        # Print summary
        total_size = 0
        for filename, size_mb in files_saved:
            print(f"✓ Saved {filename}: {size_mb:.2f} MB")
            total_size += size_mb
        
        print(f"\n{'='*60}")
        print(f"Total data size: {total_size:.2f} MB")
        print(f"Location: {Data_DIR.absolute()}")
        print(f"{'='*60}")
    
    def run_full_pipeline(self):
        """Run the complete download and processing pipeline."""
        print("\n" + "#"*60)
        print("# FINANCIAL DATA DOWNLOAD PIPELINE")
        print("#"*60)
        
        # Download data
        sp500_data = self.download_sp500_index()
        stock_data = self.download_stock_data()
        
        if stock_data is not None and len(stock_data) > 0:
            # Process data
            correlations = self.calculate_correlations(stock_data)
            aggregates = self.calculate_market_aggregate(stock_data)
            
            # Save data
            self.save_data(sp500_data, stock_data, correlations, aggregates)
            
            print("\n" + "#"*60)
            print("# PIPELINE COMPLETE!")
            print("#"*60)
            print("\nNext steps:")
            print("1. python regression_analysis.py")
            print("2. python forecasting.py")
            return True
        else:
            print("\n✗ Pipeline failed: No stock data downloaded")
            return False


if __name__ == "__main__":
    downloader = FinancialDataDownloader(lookback_years=10)
    success = downloader.run_full_pipeline()
    
    if success:
        print("\n✓ All data successfully downloaded and saved!")
    else:
        print("\n✗ Download failed. Check your internet connection and try again.")
