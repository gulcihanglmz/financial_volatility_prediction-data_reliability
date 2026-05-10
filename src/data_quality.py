"""
DATA QUALITY MODULE
Purpose: Detect anomalies in financial data using Isolation Forest
         Calculate Data Reliability Score (0-100 scale)

Key Concepts:
  - Isolation Forest: Unsupervised anomaly detection algorithm
  - Anomaly: Data points that deviate significantly from normal patterns
  - Data Reliability Score: Measure of data quality (0=bad, 100=perfect)

Anomaly Types Detected:
  1. Extreme price values (flash crashes, spikes)
  2. Unusual volume patterns
  3. Missing or inconsistent data
  4. Statistical outliers
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import (
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_RANDOM_STATE,
    RELIABILITY_THRESHOLD
)


class DataQuality:
    """
    Class to assess data quality and detect anomalies
    
    Attributes:
        data (pd.DataFrame): Financial data
        model (IsolationForest): Trained anomaly detection model
        anomaly_scores (pd.Series): Anomaly score for each row
        reliability_scores (pd.Series): Reliability score (0-100) for each row
    """
    
    def __init__(self, data):
        """
        Initialize DataQuality
        
        Args:
            data (pd.DataFrame): Financial data with columns [Open, High, Low, Close, Volume]
        """
        self.data = data.copy()
        self.model = None
        self.scaler = StandardScaler()
        self.anomaly_scores = None
        self.reliability_scores = None
        self.features_for_detection = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    def detect_anomalies(self):
        """
        Use Isolation Forest to detect anomalies in financial data
        
        Returns:
            pd.DataFrame: Original data with anomaly indicators
        
        Process:
            1. Select relevant features (OHLCV)
            2. Standardize data (important for distance-based algorithms)
            3. Train Isolation Forest
            4. Generate anomaly predictions (-1 = anomaly, 1 = normal)
        """
        print("\n Starting anomaly detection...")
        print(f"   Contamination rate: {ISOLATION_FOREST_CONTAMINATION*100:.1f}%")
        
        # 1. Prepare features
        try:
            X = self.data[self.features_for_detection].copy()
        except KeyError as e:
            raise ValueError(f"Missing required column: {str(e)}")
        
        # 2. Standardize features (crucial for Isolation Forest)
        X_scaled = self.scaler.fit_transform(X)
        
        # 3. Train Isolation Forest
        self.model = IsolationForest(
            contamination=ISOLATION_FOREST_CONTAMINATION,
            random_state=ISOLATION_FOREST_RANDOM_STATE,
            n_estimators=100
        )
        
        predictions = self.model.fit_predict(X_scaled)
        anomaly_scores = self.model.score_samples(X_scaled)
        
        # 4. Store results
        self.data['is_anomaly'] = predictions  # -1 = anomaly, 1 = normal
        self.data['anomaly_score'] = anomaly_scores  # Lower = more anomalous
        
        num_anomalies = (predictions == -1).sum()
        num_normal = (predictions == 1).sum()
        
        print(f" Anomaly detection completed!")
        print(f"   Normal data points: {num_normal}")
        print(f"   Anomalies detected: {num_anomalies}")
        
        return self.data
    
    def calculate_reliability_score(self):
        """
        Calculate Data Reliability Score (0-100 scale) for each data point
        
        Returns:
            pd.Series: Reliability scores
        
        Calculation:
            - Base Score: From anomaly score (-1 to 1)
            - Penalties: For missing values, price gaps, etc.
            - Final Score: 0-100 scale
            
        Score Interpretation:
            0-40:   UNRELIABLE    (DO NOT use for prediction)
            40-70:  CAUTION      (Use with care)
            70-100: RELIABLE     (Safe to use for prediction)
        """
        print("\n Calculating reliability scores...")
        
        # Start with anomaly scores (normalize to 0-1)
        # Anomaly score ranges from -1 (most anomalous) to 0 (least anomalous)
        # We'll convert this to 0-100 scale (min-max scaling)
        
        anomaly_normalized = (self.data['anomaly_score'] - self.data['anomaly_score'].min()) / \
                            (self.data['anomaly_score'].max() - self.data['anomaly_score'].min())
        
        # Convert to 0-100 scale (higher = better quality)
        self.reliability_scores = anomaly_normalized * 100
        
        # Apply penalties for specific issues 
        penalties = np.zeros(len(self.data))
        
        # 1. Penalty for missing values
        missing_count = self.data[self.features_for_detection].isnull().sum(axis=1)
        penalties += missing_count * 10  # 10 points per missing value (her eksik veri = -10 puan)
        
        # 2. Penalty for extreme price changes (potential data errors)
        if 'Close' in self.data.columns:
            price_changes = self.data['Close'].pct_change().abs()
            extreme_threshold = price_changes.quantile(0.95)
            extreme_changes = price_changes > extreme_threshold * 2
            penalties[extreme_changes.values] += 15
        
        # 3. Penalty for zero or negative volumes
        if 'Volume' in self.data.columns:
            zero_volume = self.data['Volume'] <= 0
            penalties[zero_volume.values] += 20
        
        # 4. Penalty if marked as anomaly
        penalties[(self.data['is_anomaly'] == -1).values] += 25
        
        # Apply penalties
        self.reliability_scores -= penalties
        
        # Clip to 0-100 range
        self.reliability_scores = np.clip(self.reliability_scores, 0, 100)
        
        self.data['reliability_score'] = self.reliability_scores
        
        # Statistics
        mean_score = self.reliability_scores.mean()
        std_score = self.reliability_scores.std()
        reliable_count = (self.reliability_scores >= RELIABILITY_THRESHOLD).sum()
        
        print(f" Reliability scores calculated!")
        print(f"   Mean score: {mean_score:.2f}")
        print(f"   Std dev: {std_score:.2f}")
        print(f"   Data points above threshold ({RELIABILITY_THRESHOLD}): {reliable_count} / {len(self.data)}")
        
        return self.reliability_scores
    
    def get_quality_report(self):
        """
        Generate quality report summary
        
        Returns:
            dict: Quality metrics
        """
        if self.reliability_scores is None:
            raise ValueError("Call calculate_reliability_score() first!")
        
        report = {
            'total_records': len(self.data),
            'anomalies_detected': (self.data['is_anomaly'] == -1).sum(),
            'normal_records': (self.data['is_anomaly'] == 1).sum(),
            'mean_reliability': self.reliability_scores.mean(),
            'min_reliability': self.reliability_scores.min(),
            'max_reliability': self.reliability_scores.max(),
            'reliable_records': (self.reliability_scores >= RELIABILITY_THRESHOLD).sum(),
            'unreliable_records': (self.reliability_scores < 40).sum(),
        }
        
        return report
    
    def print_quality_report(self):
        """Print formatted quality report"""
        report = self.get_quality_report()
        
        print("\n" + "="*70)
        print(" DATA QUALITY REPORT")
        print("="*70)
        print(f"Total records:              {report['total_records']}")
        print(f"Anomalies detected:         {report['anomalies_detected']}")
        print(f"Normal records:             {report['normal_records']}")
        print(f"\nReliability Scores:")
        print(f"  Mean:                     {report['mean_reliability']:.2f}")
        print(f"  Min:                      {report['min_reliability']:.2f}")
        print(f"  Max:                      {report['max_reliability']:.2f}")
        print(f"\nData Quality Distribution:")
        print(f"  Reliable (≥70):           {report['reliable_records']} ({report['reliable_records']/report['total_records']*100:.1f}%) ✅")
        print(f"  Caution (40-70):          {report['total_records'] - report['reliable_records'] - report['unreliable_records']} ({(report['total_records'] - report['reliable_records'] - report['unreliable_records'])/report['total_records']*100:.1f}%) ⚠️")
        print(f"  Unreliable (<40):         {report['unreliable_records']} ({report['unreliable_records']/report['total_records']*100:.1f}%) ❌")
        print("="*70 + "\n")
        
        return report
    
    def get_data(self):
        """Get processed data with quality metrics"""
        if self.data is None:
            raise ValueError("No data available!")
        return self.data


# Example usage
if __name__ == "__main__":
    print("Data Quality Module - Example Usage")
    print("This module requires data loaded from DataLoader")
