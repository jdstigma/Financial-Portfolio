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
    'JNJ', 'V', 'WMT', 'JPM', 'PG', 'MA', 'HD', 'MCD',
    'NFLX', 'INTC', 'AMD', 'IBM', 'CSCO', 'PEP', 'KO', 'COST',
    'ABBV', 'LLY', 'MRK', 'PFE', 'TMO', 'ABT', 'ADBE', 'CRM', 'ORCL',
    'DIS', 'QCOM', 'AVGO', 'ACN', 'ASML', 'NKE', 'CMCSA', 'TXN', 'WDAY',
    'NOW', 'BKNG', 'MSCI', 'FISV', 'ROP', 'BA', 'PYPL', 'AVGO'\n]


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
            sp500 = yf.download('^GSPC', start=self.start_date, end=self.end_date, 
                               progress=False, timeout=30)
            
            if sp500 is None or len(sp500) == 0:
                print("✗ No data returned from Yahoo Finance")
                return None
            
            # Reset index to make Date a column
            sp500 = sp500.reset_index()
            
            # Get the actual column names from the downloaded data
            actual_cols = list(sp500.columns)
            
            # Map to standard names (handle case variations)
            col_mapping = {}
            for col in actual_cols:
                col_lower = col.lower()
                if col_lower == 'date':
                    col_mapping[col] = 'Date'
                elif col_lower == 'open':
                    col_mapping[col] = 'Open'
                elif col_lower == 'high':
                    col_mapping[col] = 'High'
                elif col_lower == 'low':
                    col_mapping[col] = 'Low'
                elif col_lower == 'close':
                    col_mapping[col] = 'Close'
                elif 'adj close' in col_lower or 'adjclose' in col_lower:
                    col_mapping[col] = 'Adj Close'
                elif col_lower == 'volume':
                    col_mapping[col] = 'Volume'
            
            sp500 = sp500.rename(columns=col_mapping)
            
            # Keep only essential columns
            essential_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            available_cols = [col for col in essential_cols if col in sp500.columns]
            
            if len(available_cols) < 6:
                print(f"✗ Not enough columns. Available: {available_cols}")
                return None
            
            sp500 = sp500[available_cols]
            sp500['Ticker'] = '^GSPC'
            
            print(f"✓ Downloaded {len(sp500)} records")
            print(f"  Date Range: {sp500['Date'].min().date()} to {sp500['Date'].max().date()}")
            print(f"  Price Range: ${sp500['Adj Close'].min():.2f} - ${sp500['Adj Close'].max():.2f}")
            
            return sp500
        
        except Exception as e:
            print(f"✗ Error downloading S&P 500 index: {str(e)[:100]}")
            return None
    
    def download_stock_data(self, tickers=None):
        """
        Download historical data for multiple stocks individually.
        
        Parameters:
        -----------
        tickers : list, optional
            List of stock tickers (default: TOP_SP500_STOCKS)
        
        Returns:
        --------
        pd.DataFrame : Combined stock data
        """
        if tickers is None:
            tickers = TOP_SP500_STOCKS
        
        print("\n" + "="*60)
        print(f"Downloading {len(tickers)} Stock Data...\")\n        print(\"=\"*60)\n        \n        all_data = []\n        failed_tickers = []\n        \n        # Download individually to avoid yfinance compatibility issues\n        for ticker in tqdm(tickers, desc=\"Download Progress\"):\n            try:\n                # Download single ticker\n                ticker_data = yf.download(ticker, start=self.start_date, end=self.end_date,\n                                         progress=False, timeout=30)\n                \n                if ticker_data is None or len(ticker_data) == 0:\n                    failed_tickers.append(ticker)\n                    continue\n                \n                # Reset index to make Date a column\n                ticker_data = ticker_data.reset_index()\n                \n                # Get actual column names\n                actual_cols = list(ticker_data.columns)\n                \n                # Map to standard names\n                col_mapping = {}\n                for col in actual_cols:\n                    col_lower = col.lower()\n                    if col_lower == 'date':\n                        col_mapping[col] = 'Date'\n                    elif col_lower == 'open':\n                        col_mapping[col] = 'Open'\n                    elif col_lower == 'high':\n                        col_mapping[col] = 'High'\n                    elif col_lower == 'low':\n                        col_mapping[col] = 'Low'\n                    elif col_lower == 'close':\n                        col_mapping[col] = 'Close'\n                    elif 'adj close' in col_lower or 'adjclose' in col_lower:\n                        col_mapping[col] = 'Adj Close'\n                    elif col_lower == 'volume':\n                        col_mapping[col] = 'Volume'\n                \n                ticker_data = ticker_data.rename(columns=col_mapping)\n                \n                # Keep essential columns only\n                essential_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']\n                available_cols = [col for col in essential_cols if col in ticker_data.columns]\n                \n                if len(available_cols) >= 6:  # Must have at least 6 columns\n                    ticker_data = ticker_data[available_cols]\n                    ticker_data['Ticker'] = ticker\n                    all_data.append(ticker_data)\n                else:\n                    failed_tickers.append(ticker)\n            \n            except Exception as e:\n                failed_tickers.append(ticker)\n        \n        if all_data:\n            combined_df = pd.concat(all_data, ignore_index=True)\n            print(f\"\\n✓ Downloaded {len(combined_df)} records for {combined_df['Ticker'].nunique()} stocks\")\n            if failed_tickers:\n                print(f\"✗ Failed: {len(failed_tickers)} tickers ({', '.join(failed_tickers[:10])}...)\")\n            return combined_df\n        else:\n            print(\"\\n✗ No data downloaded\")\n            return None\n    \n    def calculate_correlations(self, stock_data):\n        \"\"\"\n        Calculate correlation matrix between stocks.\n        \n        Parameters:\n        -----------\n        stock_data : pd.DataFrame\n            Stock data with Adj Close prices\n        \n        Returns:\n        --------\n        pd.DataFrame : Correlation matrix\n        \"\"\"\n        print(\"\\n\" + \"=\"*60)\n        print(\"Calculating Stock Correlations...\")\n        print(\"=\"*60)\n        \n        try:\n            # Pivot to get prices by ticker\n            pivot_data = stock_data.pivot(index='Date', columns='Ticker', values='Adj Close')\n            \n            # Calculate correlations\n            correlations = pivot_data.corr()\n            \n            print(f\"✓ Calculated correlations for {len(correlations)} stocks\")\n            return correlations\n        \n        except Exception as e:\n            print(f\"✗ Error calculating correlations: {e}\")\n            return None\n    \n    def calculate_market_aggregate(self, stock_data):\n        \"\"\"\n        Calculate daily market aggregate statistics.\n        \n        Parameters:\n        -----------\n        stock_data : pd.DataFrame\n            Stock data\n        \n        Returns:\n        --------\n        pd.DataFrame : Daily market aggregates\n        \"\"\"\n        print(\"\\n\" + \"=\"*60)\n        print(\"Calculating Market Aggregates...\")\n        print(\"=\"*60)\n        \n        try:\n            daily_agg = stock_data.groupby('Date').agg({\n                'Adj Close': ['mean', 'median', 'std', 'min', 'max'],\n                'Volume': ['sum', 'mean'],\n                'Ticker': 'count'\n            }).reset_index()\n            \n            daily_agg.columns = ['Date', 'Avg_Price', 'Median_Price', 'Std_Price', \n                                'Min_Price', 'Max_Price', 'Total_Volume', 'Avg_Volume', 'Stock_Count']\n            \n            print(f\"✓ Calculated aggregates for {len(daily_agg)} days\")\n            return daily_agg\n        \n        except Exception as e:\n            print(f\"✗ Error calculating aggregates: {e}\")\n            return None\n    \n    def save_data(self, sp500_data, stock_data, correlations, aggregates):\n        \"\"\"\n        Save all data to CSV files with Git LFS tracking.\n        \n        Parameters:\n        -----------\n        sp500_data : pd.DataFrame\n            S&P 500 index data\n        stock_data : pd.DataFrame\n            Individual stock data\n        correlations : pd.DataFrame\n            Correlation matrix\n        aggregates : pd.DataFrame\n            Market aggregates\n        \"\"\"\n        print(\"\\n\" + \"=\"*60)\n        print(\"Saving Data to Files...\")\n        print(\"=\"*60)\n        \n        files_saved = []\n        \n        # Save S&P 500 index\n        if sp500_data is not None and len(sp500_data) > 0:\n            index_file = Data_DIR / 'sp500_index.csv'\n            sp500_data.to_csv(index_file, index=False)\n            size_mb = sp500_data.memory_usage(deep=True).sum() / 1024**2\n            files_saved.append(('sp500_index.csv', size_mb))\n        \n        # Save stock data\n        if stock_data is not None and len(stock_data) > 0:\n            stocks_file = Data_DIR / 'sp500_stocks.csv'\n            stock_data.to_csv(stocks_file, index=False)\n            size_mb = stock_data.memory_usage(deep=True).sum() / 1024**2\n            files_saved.append(('sp500_stocks.csv', size_mb))\n        \n        # Save correlations\n        if correlations is not None and len(correlations) > 0:\n            corr_file = Data_DIR / 'stock_correlations.csv'\n            correlations.to_csv(corr_file)\n            size_mb = correlations.memory_usage(deep=True).sum() / 1024**2\n            files_saved.append(('stock_correlations.csv', size_mb))\n        \n        # Save aggregates\n        if aggregates is not None and len(aggregates) > 0:\n            agg_file = Data_DIR / 'market_daily_aggregate.csv'\n            aggregates.to_csv(agg_file, index=False)\n            size_mb = aggregates.memory_usage(deep=True).sum() / 1024**2\n            files_saved.append(('market_daily_aggregate.csv', size_mb))\n        \n        # Print summary\n        total_size = 0\n        for filename, size_mb in files_saved:\n            print(f\"✓ Saved {filename}: {size_mb:.2f} MB\")\n            total_size += size_mb\n        \n        print(f\"\\n{'='*60}\")\n        print(f\"Total data size: {total_size:.2f} MB\")\n        print(f\"Location: {Data_DIR.absolute()}\")\n        print(f\"{'='*60}\")\n    \n    def run_full_pipeline(self):\n        \"\"\"\n        Run the complete download and processing pipeline.\n        \"\"\"\n        print(\"\\n\" + \"#\"*60)\n        print(\"# FINANCIAL DATA DOWNLOAD PIPELINE\")\n        print(\"#\"*60)\n        \n        # Download data\n        sp500_data = self.download_sp500_index()\n        stock_data = self.download_stock_data()\n        \n        if stock_data is not None and len(stock_data) > 0:\n            # Process data\n            correlations = self.calculate_correlations(stock_data)\n            aggregates = self.calculate_market_aggregate(stock_data)\n            \n            # Save data\n            self.save_data(sp500_data, stock_data, correlations, aggregates)\n            \n            print(\"\\n\" + \"#\"*60)\n            print(\"# PIPELINE COMPLETE!\")\n            print(\"#\"*60)\n            print(\"\\nNext steps:\")\n            print(\"1. python regression_analysis.py\")\n            print(\"2. python forecasting.py\")\n            return True\n        else:\n            print(\"\\n✗ Pipeline failed: No stock data downloaded\")\n            return False\n\n\nif __name__ == \"__main__\":\n    # Initialize downloader\n    downloader = FinancialDataDownloader(lookback_years=10)\n    \n    # Run pipeline\n    success = downloader.run_full_pipeline()\n    \n    if success:\n        print(\"\\n✓ All data successfully downloaded and saved!\")\n    else:\n        print(\"\\n✗ Download failed. Check your internet connection and try again.\")\n