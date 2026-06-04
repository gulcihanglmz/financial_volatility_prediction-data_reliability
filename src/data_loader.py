"""
DATA LOADER MODULE
Purpose: Load financial data from CSV file and perform basic preprocessing
Functions:
  - Load NASDAQ dataset from CSV
  - Handle missing values
  - Data validation
  - Basic statistics
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

class DataLoader:
    """
    Class to load and preprocess financial data from CSV files
    
    Attributes:
        filepath (str): Path to CSV file
        data (pd.DataFrame): Loaded data
    """
    
    def __init__(self, filepath=None):
        """
        Initialize DataLoader
        
        Args:
            filepath (str): Path to CSV file (optional)
        """
        self.filepath = filepath
        self.data = None
    
    def load_csv(self, filepath=None):
        """
        Load CSV file into DataFrame
        
        Args:
            filepath (str): Path to CSV file
        
        Returns:
            pd.DataFrame: Loaded data
        
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if filepath is None:
            filepath = self.filepath
        
        if filepath is None:
            raise ValueError("File path must be provided!")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        print(f"Loading data from: {filepath}")
        
        try:
            self.data = pd.read_csv(filepath)
            print(f" Data loaded successfully!")
            print(f"   Shape: {self.data.shape} (rows, columns)")
            return self.data
        
        except Exception as e:
            print(f" Error loading file: {str(e)}")
            raise
    
    def preprocess(self):
        """
        Perform basic data preprocessing
        Returns:
            pd.DataFrame: Preprocessed data
        """
        if self.data is None:
            raise ValueError("No data loaded! Call load_csv() first.")
        
        print("\nStarting data preprocessing...")
        
        # 1. Convert Date column to datetime
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data.set_index('Date', inplace=True)
            print("✓ Date column converted and set as index")
        
        # 2. Remove leading/trailing spaces from column names
        self.data.columns = self.data.columns.str.strip()
        print("✓ Column names cleaned")
        
        # 3. Handle missing values
        initial_missing = self.data.isnull().sum().sum()
        if initial_missing > 0:
            print(f"  Found {initial_missing} missing values")
            self.data.ffill(inplace=True)  # Forward fill
            print(f"✓ Missing values filled using forward fill")
        else:
            print("✓ No missing values found")
        
        # 4. Remove duplicates
        initial_rows = len(self.data)
        self.data.drop_duplicates(inplace=True)
        removed_rows = initial_rows - len(self.data)
        if removed_rows > 0:
            print(f"✓ Removed {removed_rows} duplicate rows")
        
        # 5. Sort by date (ascending)
        self.data.sort_index(inplace=True)
        print("✓ Data sorted by date")
        
        print(f"Preprocessing completed!")
        print(f"   Final shape: {self.data.shape}")
        
        return self.data
    
    def get_summary_statistics(self):
        """
        Display summary statistics
        
        Returns:
            pd.DataFrame: Statistical summary
        """
        if self.data is None:
            raise ValueError("No data loaded! Call load_csv() first.")
        
        print("\nSummary Statistics:\n")
        summary = self.data.describe()
        print(summary)
        
        return summary
    
    def get_data_info(self):
        """
        Display data information
        
        Returns:
            None
        """
        if self.data is None:
            raise ValueError("No data loaded! Call load_csv() first.")
        
        print("\nData Information:\n")
        print(self.data.info())
        print(f"\nDate range: {self.data.index.min()} to {self.data.index.max()}")
        print(f"Total trading days: {len(self.data)}")
        print(f"Missing values:\n{self.data.isnull().sum()}")
    
    def get_data(self):
        """
        Get the loaded and preprocessed data
        
        Returns:
            pd.DataFrame: Data
        """
        if self.data is None:
            raise ValueError("No data loaded! Call load_csv() first.")
        
        return self.data
    
    def save_processed_data(self, output_path="data/processed_data.csv"):
        """
        Save processed data to CSV file
        
        Args:
            output_path (str): Path to save file
        
        Returns:
            bool: True if successful
        """
        if self.data is None:
            raise ValueError("No data to save! Call load_csv() first.")
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        try:
            self.data.to_csv(output_path)
            print(f"Data saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # This is for testing the DataLoader module
    # Replace with your actual CSV file path
    
    csv_path = "data/nasdq.csv"  # CSV file path
    
    loader = DataLoader(csv_path)
    loader.load_csv()
    loader.preprocess()
    loader.get_data_info()
    loader.get_summary_statistics()
    loader.save_processed_data()
