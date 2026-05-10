"""
FEATURE ENGINEERING MODULE
Purpose: Create technical indicators for price prediction
Technical Indicators:
  1. Moving Averages (MA) - Trend identification
  2. RSI (Relative Strength Index) - Momentum
  3. MACD (Moving Average Convergence Divergence) - Trend strength
  4. ATR (Average True Range) - Volatility
  5. Price changes - Momentum features
  6. Volume features - Trading activity
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import TECHNICAL_INDICATORS


class FeatureEngineer:
    """
    Class to create and manage technical indicators
    
    Attributes:
        data (pd.DataFrame): Input data with OHLCV columns
        features (pd.DataFrame): DataFrame with all features
    """
    
    def __init__(self, data):
        """
        Initialize FeatureEngineer
        
        Args:
            data (pd.DataFrame): Data with columns [Open, High, Low, Close, Volume]
        """
        self.data = data.copy()
        self.features = data.copy()
    
    def add_moving_averages(self):
        """
        Add simple moving averages (SMA)
        
        Moving Average = Average of last N prices
        - Short MA (5-day): Quick trend
        - Long MA (20-day): Longer trend
        
        Returns:
            pd.DataFrame: Data with MA columns
        """
        print(" Adding Moving Averages...")
        
        ma_short = TECHNICAL_INDICATORS['ma_short']
        ma_long = TECHNICAL_INDICATORS['ma_long']
        
        self.features[f'MA_{ma_short}'] = self.data['Close'].rolling(window=ma_short).mean()
        self.features[f'MA_{ma_long}'] = self.data['Close'].rolling(window=ma_long).mean()
        
        # MA ratio (short/long) - trend strength indicator
        self.features['MA_Ratio'] = self.features[f'MA_{ma_short}'] / self.features[f'MA_{ma_long}']
        
        print(f"   ✓ MA_{ma_short} (short-term)")
        print(f"   ✓ MA_{ma_long} (long-term)")
        print(f"   ✓ MA_Ratio (trend strength)")
        
        return self.features
    
    def add_rsi(self):
        """
        Add RSI (Relative Strength Index)
        
        RSI measures overbought/oversold conditions
        Formula: RSI = 100 - (100 / (1 + RS))
                where RS = Average Gain / Average Loss
        
        Interpretation:
        - RSI > 70: Overbought (might fall)
        - RSI < 30: Oversold (might rise)
        - 30-70: Neutral
        
        Returns:
            pd.DataFrame: Data with RSI column
        """
        print(" Adding RSI (Relative Strength Index)...")
        
        rsi_period = TECHNICAL_INDICATORS['rsi_period']
        
        # Calculate price changes
        delta = self.data['Close'].diff()
        
        # Separate gains and losses
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=rsi_period).mean()
        avg_losses = losses.rolling(window=rsi_period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        self.features['RSI'] = rsi
        
        print(f"   ✓ RSI_{rsi_period}period)")
        
        return self.features
    
    def add_macd(self):
        """
        Add MACD (Moving Average Convergence Divergence)
        
        MACD = Fast EMA - Slow EMA
        Signal = 9-period EMA of MACD
        Histogram = MACD - Signal
        
        Used for trend and momentum identification
        
        Returns:
            pd.DataFrame: Data with MACD columns
        """
        print(" Adding MACD...")
        
        fast = TECHNICAL_INDICATORS['macd_fast']
        slow = TECHNICAL_INDICATORS['macd_slow']
        signal_period = 9
        
        # Calculate exponential moving averages
        ema_fast = self.data['Close'].ewm(span=fast).mean()
        ema_slow = self.data['Close'].ewm(span=slow).mean()
        
        # MACD line
        macd = ema_fast - ema_slow
        
        # Signal line
        signal = macd.ewm(span=signal_period).mean()
        
        # MACD histogram
        histogram = macd - signal
        
        self.features['MACD'] = macd
        self.features['MACD_Signal'] = signal
        self.features['MACD_Histogram'] = histogram
        
        print(f"   ✓ MACD ({fast} vs {slow} period)")
        print(f"   ✓ MACD_Signal ({signal_period} period)")
        print(f"   ✓ MACD_Histogram")
        
        return self.features
    
    def add_atr(self):
        """
        Add ATR (Average True Range)
        
        ATR measures volatility
        True Range = max(High-Low, |High-Close_prev|, |Low-Close_prev|)
        
        Returns:
            pd.DataFrame: Data with ATR column
        """
        print(" Adding ATR (Average True Range)...")
        
        atr_period = TECHNICAL_INDICATORS['atr_period']
        
        # Calculate true range
        high_low = self.data['High'] - self.data['Low']
        high_close = abs(self.data['High'] - self.data['Close'].shift())
        low_close = abs(self.data['Low'] - self.data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # ATR is the average of true range
        atr = true_range.rolling(window=atr_period).mean()
        
        self.features['ATR'] = atr
        
        print(f"   ✓ ATR_{atr_period}period)")
        
        return self.features
    
    def add_momentum_features(self):
        """
        Add simple momentum features
        
        Features:
        - Price change % (1-day, 5-day, 20-day)
        - Volume change
        - Volume weighted price
        
        Returns:
            pd.DataFrame: Data with momentum columns
        """
        print(" Adding Momentum Features...")
        
        # Percentage change
        self.features['Price_Change_1d'] = self.data['Close'].pct_change(1)
        self.features['Price_Change_5d'] = self.data['Close'].pct_change(5)
        self.features['Price_Change_20d'] = self.data['Close'].pct_change(20)
        
        # Volume change
        self.features['Volume_Change'] = self.data['Volume'].pct_change(1)
        
        # Volume weighted price
        self.features['VWAP'] = (self.data['Close'] * self.data['Volume']).rolling(window=20).sum() / \
                                self.data['Volume'].rolling(window=20).sum()
        
        print(f"   ✓ Price_Change (1d, 5d, 20d)")
        print(f"   ✓ Volume_Change")
        print(f"   ✓ VWAP (Volume Weighted Average Price)")
        
        return self.features
    
    def add_external_indicators(self):
        """
        Add external economic indicators if available
        
        Features: VIX, InterestRate, Oil, Gold prices
        
        Returns:
            pd.DataFrame: Data with external indicator features
        """
        print(" Adding External Indicators...")
        
        external_cols = ['VIX', 'InterestRate', 'Oil', 'Gold']
        
        for col in external_cols:
            if col in self.data.columns:
                # Normalize indicators
                self.features[f'{col}_Change'] = self.data[col].pct_change(1)
                self.features[f'{col}_MA_5'] = self.data[col].rolling(window=5).mean()
                print(f"   ✓ {col} features")
        
        return self.features
    
    def create_target_variable(self):
        """
        Create target variable for supervised learning
        
        Target: Next day's Close price movement (binary)
            1 = Price will increase (Close_t+1 > Close_t)
            0 = Price will stay same or decrease
        
        Returns:
            pd.DataFrame: Data with target variable
        """
        print(" Creating Target Variable...")
        
        # Calculate next day's closing price
        self.features['Target'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)
        
        print(f"   ✓ Target variable created (Binary classification)")
        print(f"     1 = Price increases, 0 = Price stays/decreases")
        
        return self.features
    
    def remove_missing_values(self):
        """
        Remove rows with missing values (created by rolling window operations)
        
        Returns:
            pd.DataFrame: Clean features
        """
        initial_rows = len(self.features)
        self.features.dropna(inplace=True)
        removed_rows = initial_rows - len(self.features)
        
        print(f"\n🧹 Cleaning data:")
        print(f"   Removed {removed_rows} rows with missing values")
        print(f"   Final dataset: {len(self.features)} rows")
        
        return self.features
    
    def get_features(self):
        """Get feature dataframe"""
        return self.features
    
    def apply_all_features(self):
        """Apply all feature engineering steps"""
        print("\n" + "="*70)
        print(" FEATURE ENGINEERING - APPLYING ALL FEATURES")
        print("="*70 + "\n")
        
        self.add_moving_averages()
        self.add_rsi()
        self.add_macd()
        self.add_atr()
        self.add_momentum_features()
        self.add_external_indicators()
        self.create_target_variable()
        self.remove_missing_values()
        
        print("\n" + "="*70)
        print(f" Feature Engineering Completed!")
        print(f"   Total features created: {len(self.features.columns) - len(self.data.columns)}")
        print(f"   Final shape: {self.features.shape}")
        print("="*70 + "\n")
        
        return self.features


# Example usage
if __name__ == "__main__":
    print("Feature Engineering Module - Example Usage")
    print("This module requires data loaded from DataLoader")
