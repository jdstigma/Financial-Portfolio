"""
Financial Data Download Module

Downloads S&P 500 stock data from Yahoo Finance and creates
large CSV datasets for portfolio analysis.

Datasets generated:
- sp500_index.csv: S&P 500 index historical data
- sp500_stocks.csv: Individual stock data for top 50 S&P 500 companies
- market_daily_aggregate.csv: Daily market aggregates
- stock_correlations.csv: Correlation matrix

Expected total size: 200-500 MB
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
from pathlib import Path
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Create data directory if it doesn't exist
Data_DIR = Path('data')
Data_DIR.mkdir(exist_ok=True)

# Top 50 S&P 500 Companies (as of 2024)
TOP_SP500_STOCKS = [
    'AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',
    'BRK.B', 'JNJ', 'V', 'WMT', 'JPM', 'PG', 'MA', 'HD', 'MCD', 'BA',
    'NFLX', 'PYPL', 'INTC', 'AMD', 'IBM', 'CSCO', 'PEP', 'KO', 'COST',
    'ABBV', 'LLY', 'MRK', 'PFE', 'TMO', 'ABT', 'ADBE', 'CRM', 'ORCL',
    'DIS', 'QCOM', 'AVGO', 'ACN', 'ASML', 'NKE', 'CMCSA', 'TXN', 'WDAY',
    'NOW', 'BKNG', 'MSCI', 'FISV', 'ROP'
]


class FinancialDataDownloader:
    """
    Download and process financial data from Yahoo Finance.
    """
    
    def __init__(self, start_date=None, end_date=None, lookback_years=10):
        """
        Initialize downloader.
        
        Parameters:
        -----------
        start_date : str, optional
            Start date in format 'YYYY-MM-DD'
        end_date : str, optional
            End date in format 'YYYY-MM-DD'
        lookback_years : int
            Years of historical data to download (default: 10)
        """
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            start_datetime = datetime.now() - timedelta(days=365*lookback_years)
            self.start_date = start_datetime.strftime('%Y-%m-%d')
        else:
            self.start_date = start_date
        
        print(f"Date Range: {self.start_date} to {self.end_date}")
    
    def download_sp500_index(self):
        """
        Download S&P 500 index data.
        
        Returns:
        --------
        pd.DataFrame : S&P 500 index data
        """
        print("\n" + "="*60)
        print("Downloading S&P 500 Index Data...")
        print("="*60)
        
        try:
            sp500 = yf.download('^GSPC', start=self.start_date, end=self.end_date, progress=False)
            sp500 = sp500.reset_index()
            sp500.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            sp500['Ticker'] = '^GSPC'
            
            print(f"✓ Downloaded {len(sp500)} records")
            print(f"  Date Range: {sp500['Date'].min().date()} to {sp500['Date'].max().date()}")
            print(f"  Price Range: ${sp500['Adj Close'].min():.2f} - ${sp500['Adj Close'].max():.2f}")
            
            return sp500
        
        except Exception as e:
            print(f"✗ Error downloading S&P 500 index: {e}")
            return None
    
    def download_stock_data(self, tickers=None, batch_size=10):
        """
        Download historical data for multiple stocks.
        
        Parameters:
        -----------
        tickers : list, optional
            List of stock tickers (default: TOP_SP500_STOCKS)
        batch_size : int
            Number of stocks to download at once
        
        Returns:
        --------
        pd.DataFrame : Combined stock data
        """
        if tickers is None:
            tickers = TOP_SP500_STOCKS
        
        print("\n" + "="*60)
        print(f"Downloading {len(tickers)} Stock Data...")
        print("="*60)
        
        all_data = []
        
        # Download in batches to avoid rate limiting
        for i in tqdm(range(0, len(tickers), batch_size), desc="Download Progress"):
            batch = tickers[i:i+batch_size]
            
            try:
                batch_data = yf.download(batch, start=self.start_date, end=self.end_date, 
                                        progress=False, group_by='ticker')
                
                for ticker in batch:
                    try:
                        if ticker in batch_data.columns.get_level_values(0) or len(batch) == 1:
                            if len(batch) == 1:
                                ticker_data = batch_data.copy()
                            else:
                                ticker_data = batch_data[ticker]
                            
                            df = ticker_data.reset_index()
                            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                            df['Ticker'] = ticker
                            all_data.append(df)
                    
                    except Exception as e:
                        print(f"  ⚠ Skipping {ticker}: {str(e)[:50]}")
            
            except Exception as e:
                print(f"  ⚠ Batch error: {str(e)[:50]}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\n✓ Downloaded {len(combined_df)} records for {combined_df['Ticker'].nunique()} stocks")
            return combined_df
        else:
            print("✗ No data downloaded")
            return None
    
    def calculate_correlations(self, stock_data):
        """
        Calculate correlation matrix between stocks.
        
        Parameters:
        -----------
        stock_data : pd.DataFrame
            Stock data with Adj Close prices
        
        Returns:
        --------
        pd.DataFrame : Correlation matrix
        """
        print("\n" + "="*60)
        print("Calculating Stock Correlations...")
        print("="*60)
        
        try:
            # Pivot to get prices by ticker
            pivot_data = stock_data.pivot(index='Date', columns='Ticker', values='Adj Close')
            
            # Calculate correlations
            correlations = pivot_data.corr()
            
            print(f"✓ Calculated correlations for {len(correlations)} stocks")
            return correlations
        
        except Exception as e:
            print(f"✗ Error calculating correlations: {e}")
            return None
    
    def calculate_market_aggregate(self, stock_data):
        """
        Calculate daily market aggregate statistics.
        
        Parameters:
        -----------
        stock_data : pd.DataFrame
            Stock data
        
        Returns:
        --------
        pd.DataFrame : Daily market aggregates
        """
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
            print(f"✗ Error calculating aggregates: {e}")
            return None
    
    def save_data(self, sp500_data, stock_data, correlations, aggregates):
        """
        Save all data to CSV files with Git LFS tracking.
        
        Parameters:
        -----------
        sp500_data : pd.DataFrame
            S&P 500 index data
        stock_data : pd.DataFrame
            Individual stock data
        correlations : pd.DataFrame
            Correlation matrix
        aggregates : pd.DataFrame
            Market aggregates
        """
        print("\n" + "="*60)
        print("Saving Data to Files...")
        print("="*60)
        
        files_saved = []
        
        # Save S&P 500 index
        if sp500_data is not None:
            index_file = Data_DIR / 'sp500_index.csv'
            sp500_data.to_csv(index_file, index=False)
            files_saved.append(('sp500_index.csv', sp500_data.memory_usage(deep=True).sum() / 1024**2))
        
        # Save stock data
        if stock_data is not None:
            stocks_file = Data_DIR / 'sp500_stocks.csv'
            stock_data.to_csv(stocks_file, index=False)
            files_saved.append(('sp500_stocks.csv', stock_data.memory_usage(deep=True).sum() / 1024**2))
        
        # Save correlations
        if correlations is not None:
            corr_file = Data_DIR / 'stock_correlations.csv'
            correlations.to_csv(corr_file)
            files_saved.append(('stock_correlations.csv', correlations.memory_usage(deep=True).sum() / 1024**2))
        
        # Save aggregates
        if aggregates is not None:
            agg_file = Data_DIR / 'market_daily_aggregate.csv'
            aggregates.to_csv(agg_file, index=False)
            files_saved.append(('market_daily_aggregate.csv', aggregates.memory_usage(deep=True).sum() / 1024**2))
        
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
        """
        Run the complete download and processing pipeline.
        """
        print("\n" + "#"*60)
        print("# FINANCIAL DATA DOWNLOAD PIPELINE")
        print("#"*60)
        
        # Download data
        sp500_data = self.download_sp500_index()
        stock_data = self.download_stock_data()
        
        if stock_data is not None:
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
    # Initialize downloader
    downloader = FinancialDataDownloader(lookback_years=10)
    
    # Run pipeline
    success = downloader.run_full_pipeline()
    
    if success:
        print("\n✓ All data successfully downloaded and saved!")
    else:
        print("\n✗ Download failed. Check your internet connection and try again.")
