"""
NASDAQ Volatility Prediction - Volatility Only Pipeline
Simplified version focusing on market volatility forecasting
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.data_quality import DataQuality
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

print("=" * 100)
print(" " * 25 + "NASDAQ VOLATILITY PREDICTION - MAIN PIPELINE")
print("=" * 100)

# ============= DATA LOADING AND PREPROCESSING =============
print("\n[1/5] DATA LOADING & PREPROCESSING")
print("-" * 100)

loader = DataLoader(filepath='data/nasdq.csv')
data = loader.load_csv()
data = loader.preprocess()

print(f"   Data loaded: {data.shape[0]} rows x {data.shape[1]} columns")
print(f"   Date range: {data.index.min().date()} to {data.index.max().date()}")

# ============= FEATURE ENGINEERING =============
print("\n[2/5] FEATURE ENGINEERING")
print("-" * 100)

engineer = FeatureEngineer(data)
data_engineered = engineer.apply_all_features()
print(f"   Features created: {data_engineered.shape[1]} (original 12 + engineered {data_engineered.shape[1]-12})")

# ============= DATA QUALITY ASSESSMENT =============
print("\n[3/5] DATA QUALITY ASSESSMENT")
print("-" * 100)

quality = DataQuality(data_engineered)
quality.detect_anomalies()

# ============= LAG FEATURES & TRAIN/TEST SPLIT =============
print("\n[4/5] LAG FEATURES & TRAIN-TEST SPLIT")
print("-" * 100)

def add_lag_features(df, lags=[1, 2, 3]):
    lag_df = df.copy()
    features_to_lag = ['Close', 'Volume', 'MA_5', 'MA_20', 'RSI', 'MACD']
    
    for feature in features_to_lag:
        if feature in lag_df.columns:
            for lag in lags:
                lag_df[f'{feature}_lag{lag}'] = df[feature].shift(lag)
    
    lag_df = lag_df.dropna()
    return lag_df

data_with_lags = add_lag_features(data_engineered.copy())

X = data_with_lags.drop(["Target"], axis=1)
y_volatility = data_with_lags["Target"]

# Chronological train/test split
split_idx = int(len(X) * 0.8)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y_volatility.iloc[:split_idx]
y_test = y_volatility.iloc[split_idx:]

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   Train set: {X_train.shape[0]} days ({data_with_lags.index[0]} to {data_with_lags.index[split_idx-1]})")
print(f"   Test set:  {X_test.shape[0]} days ({data_with_lags.index[split_idx]} to {data_with_lags.index[-1]})")
print(f"   Features normalized. Total features: {X_train.shape[1]}")

# ============= MODEL TRAINING & EVALUATION =============
print("\n[5/5] MODEL TRAINING & EVALUATION - VOLATILITY PREDICTION")
print("-" * 100)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    'SVM': SVC(kernel='rbf', C=10, probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

results_list = []

for model_name, model in models.items():
    print(f"\n   Training {model_name}...")
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    results_list.append({
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })
    
    print(f"   {model_name:20s} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

# ============= RESULTS =============
print("\n" + "=" * 100)
print("VOLATILITY PREDICTION RESULTS")
print("=" * 100)

df_results = pd.DataFrame(results_list).sort_values('Accuracy', ascending=False)
print("\n" + df_results.to_string(index=False))

best_model_name = df_results.iloc[0]['Model']
best_accuracy = df_results.iloc[0]['Accuracy']

print(f"\n   Best Model: {best_model_name} ({best_accuracy:.2%} accuracy)")
print(f"   Training improved over baseline: {(best_accuracy - 0.5) * 100:.2f}%")

# ============= SAVE RESULTS & MODELS =============
print("\n" + "=" * 100)
print("SAVING RESULTS AND MODELS")
print("=" * 100)

df_results.to_csv('data/volatility_prediction_results.csv', index=False)
print(f"\n   ✓ Results saved to: data/volatility_prediction_results.csv")

# Save best model + scaler
model_obj = models[best_model_name]
joblib.dump(model_obj, 'models/volatility_best_model.pkl')
joblib.dump(scaler, 'models/volatility_scaler.pkl')
joblib.dump(list(X_train.columns), 'models/feature_columns.pkl')

print(f"   ✓ Model saved to: models/volatility_best_model.pkl")
print(f"   ✓ Scaler saved to: models/volatility_scaler.pkl")
print(f"   ✓ Feature columns saved to: models/feature_columns.pkl")
print(f"   ✓ Best model saved: models/volatility_best_model.pkl")
print(f"   ✓ Scaler saved: models/volatility_scaler.pkl")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 100)
