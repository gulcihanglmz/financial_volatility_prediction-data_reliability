"""
================================================================================
PROJECT CONFIGURATION FILE
================================================================================
This file contains all configuration parameters for the financial forecasting 
system. Settings are organized by category.

CATEGORIES:
  1. Database Configuration - PostgreSQL connection settings
  2. Financial Data Settings - Which stocks to analyze
  3. Machine Learning Parameters - Model hyperparameters  
  4. Airflow Scheduler - Automation schedule
================================================================================
"""
# ❕Tüm önemli ayarları BİR YERDEN kontrol edebiliriz. Kod içinde değiştirmemize gerek yok! 


# 1. DATABASE CONFIGURATION
# These are the connection parameters for PostgreSQL Database
# The system will store raw data, processed data, predictions, and anomaly scores in this database.

DB_HOST = "localhost"                    # Database server address
DB_PORT = 5432                          # PostgreSQL default port
DB_NAME = "financial_db"                # Name of our database
DB_USER = "postgres"                    # Database username
DB_PASSWORD = "your_password"           # TODO: Change this to your actual password!
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# Tüm bulduğumuz tahminler ve kalite skorları PostgreSQL veritabanında saklanıyor. Bu şekilde daha sonradan analiz edebiliyoruz

# Example connection string created from above:
# postgresql://postgres:your_password@localhost:5432/financial_db

# 2. FINANCIAL DATA SETTINGS
# Configuration for which stocks to fetch and how much historical data

SYMBOLS = ["AAPL", "GOOGL", "MSFT"]     # Stock ticker symbols to analyze
                                        # You can add more: ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
                                        
DATA_PERIOD = "1y"                      # How much historical data to fetch
                                        # Options: "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"
                                        
DATA_FREQUENCY = "daily"                # Data frequency (daily, weekly, monthly)


# 3. MACHINE LEARNING PARAMETERS
# Hyperparameters for Isolation Forest (anomaly detection) and Random Forest (prediction)

# --- ISOLATION FOREST (Anomaly Detection) ---
ISOLATION_FOREST_CONTAMINATION = 0.1    # Expected proportion of anomalies
                                        # 0.1 means: expect ~10% of data to be anomalies
                                        
ISOLATION_FOREST_RANDOM_STATE = 42      # Random seed for reproducibility

# --- DATA RELIABILITY THRESHOLD ---
RELIABILITY_THRESHOLD = 70               # Quality gate threshold (0-100 scale)
                                        # If reliability score < 70: DO NOT predict
                                        # If reliability score >= 70: Safe to predict
                                        
# --- RANDOM FOREST (Price Prediction) ---
RANDOM_FOREST_N_ESTIMATORS = 100        # Number of decision trees in the forest
RANDOM_FOREST_MAX_DEPTH = 10            # Maximum depth of each tree
RANDOM_FOREST_RANDOM_STATE = 42         # Random seed for reproducibility

# --- TRAIN-TEST SPLIT ---
TRAIN_TEST_SPLIT = 0.8                  # 80% for training, 20% for testing historical data

# 4. FEATURE ENGINEERING
# Technical indicators to calculate from OHLCV data
'''
OHLCV data represents five critical data points—Open, High, Low, Close, and Volume
used to analyze a financial instrument's market activity over a specific time frame (e.g., 1 minute, 1 day).
'''

TECHNICAL_INDICATORS = {
    "ma_short": 5,          # Short-term moving average (5 days)
    "ma_long": 20,          # Long-term moving average (20 days)
    "rsi_period": 14,       # RSI (Relative Strength Index) period
    "macd_fast": 12,        # MACD fast exponential moving average
    "macd_slow": 26,        # MACD slow exponential moving average
    "atr_period": 14,       # Average True Range period
}

# Data reliability thresholds
# - RELIABILITY_THRESHOLD: minimum score considered 'reliable' for automatic inference
# - CRITICAL_RELIABILITY: below this score we suspend automatic predictions
RELIABILITY_THRESHOLD = 70
CRITICAL_RELIABILITY = 50


# 5. APACHE AIRFLOW SCHEDULER
# Configuration for automated pipeline execution

AIRFLOW_DAG_ID = "financial_forecasting_pipeline"
AIRFLOW_DAG_INTERVAL = "0 9 * * 1-5"    # Cron format: Monday-Friday at 9:00 AM
                                        # "0 9 * * 1-5" explanation:
                                        #   0 = minute (0)
                                        #   9 = hour (9 AM)
                                        #   * = day of month (any)
                                        #   * = month (any)
                                        #   1-5 = day of week (Monday=1 to Friday=5)
                                        
AIRFLOW_CATCHUP = False                 # Don't run missed schedules


# 6. FILE PATHS
# Paths where data and models will be stored

DATA_RAW_PATH = "data/raw_data.csv"
DATA_PROCESSED_PATH = "data/processed_data.csv"
MODEL_ISOLATION_FOREST_PATH = "models/isolation_forest_model.pkl"
MODEL_RANDOM_FOREST_PATH = "models/random_forest_model.pkl"


# 7. LOGGING & MONITORING
# Configuration for system logs and monitoring

LOG_LEVEL = "INFO"                      # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE_PATH = "logs/pipeline.log"
