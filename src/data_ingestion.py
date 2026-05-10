"""
DATA INGESTION MODULE
Purpose: Fetch financial data from Yahoo Finance and prepare it for processing

Key Functions:
  - Fetch OHLCV data (Open, High, Low, Close, Volume) for specified stocks
  - Handle errors gracefully
  - Save data to CSV file for further processing

What is OHLCV?
  O (Open):   Stock opening price on a given day
  H (High):   Highest price during the day
  L (Low):    Lowest price during the day
  C (Close):  Closing price at end of day
  V (Volume): Number of shares traded
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import SYMBOLS, DATA_PERIOD


class DataIngestion:
    """
    Class to handle downloading financial data from Yahoo Finance
    
    Attributes:
        symbols (list): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"])
        period (str): Time period for data ("1y", "5y", etc.)
    """
    
    def __init__(self):
        """Initialize DataIngestion with symbols and period from config"""
        self.symbols = SYMBOLS
        self.period = DATA_PERIOD
    
    def fetch_data(self, symbol):
        """
        Fetch historical OHLCV data for a single stock from Yahoo Finance
        
        Args:
            symbol (str): Stock ticker symbol (e.g., "AAPL", "GOOGL", "MSFT")
        
        Returns:
            pd.DataFrame: DataFrame containing OHLCV data with columns:
                         [Date, Open, High, Low, Close, Adj Close, Volume, symbol]
                         Returns None if fetch fails
        
        Example:
            >>> ingestion = DataIngestion()
            >>> df = ingestion.fetch_data("AAPL")
            >>> print(df.head())
        """
        try:
            print(f"Fetching data for {symbol}...")
            
            # Download data from Yahoo Finance
            data = yf.download(symbol, period=self.period, progress=False)
            
            # Reset index to make Date a regular column (not index)
            data.reset_index(inplace=True)
            
            # Add stock symbol column for identification
            data['symbol'] = symbol
            
            print(f"{symbol}: Successfully fetched {len(data)} rows")
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def fetch_all_data(self):
        """
        Fetch data for all symbols specified in config
        
        Returns:
            pd.DataFrame: Combined DataFrame with data from all symbols
                         Returns None if all fetches fail
        """
        all_data = []
        
        print(f"\nStarting data collection for {len(self.symbols)} symbols...\n")
        
        for symbol in self.symbols:
            data = self.fetch_data(symbol)
            if data is not None:
                all_data.append(data)
        
        if all_data:
            # Combine all DataFrames into one
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"\nTotal rows collected: {len(combined_data)}")
            return combined_data
        else:
            print("No data was collected!")
            return None
    
    def save_to_csv(self, df, filepath="data/raw_data.csv"):
        """
        Save DataFrame to CSV file
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filepath (str): Where to save the file (default: "data/raw_data.csv")
        
        Returns:
            bool: True if successful, False otherwise
        """
        if df is not None:
            # Create directory if it doesn't exist
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            df.to_csv(filepath, index=False)
            print(f"Data saved to: {filepath}")
            return True
        else:
            print("Cannot save: DataFrame is empty")
            return False


if __name__ == "__main__":
    # Test et
    ingestion = DataIngestion()
    raw_data = ingestion.fetch_all_data()
    
    if raw_data is not None:
        print("\nFirst 5 rows:")
        print(raw_data.head())
        ingestion.save_to_csv(raw_data)
    else:
        print("data cannot be saved!")
