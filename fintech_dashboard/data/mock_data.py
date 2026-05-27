"""
Mock data generator for the Financial Volatility Prediction Dashboard.
Simulates realistic financial time series, model outputs, and data quality metrics.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


def generate_price_series(n=500, start_price=100.0, seed=42):
    """Generate realistic OHLCV financial price data."""
    np.random.seed(seed)
    dates = pd.date_range(end=datetime.today(), periods=n, freq="B")

    returns = np.random.normal(0.0005, 0.015, n)
    # Add volatility clustering (GARCH-like effect)
    for i in range(1, n):
        if abs(returns[i - 1]) > 0.02:
            returns[i] *= 1.5

    prices = [start_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    prices = np.array(prices)

    # Generate OHLCV
    daily_range = prices * np.abs(np.random.normal(0.01, 0.005, n))
    opens = prices * (1 + np.random.normal(0, 0.003, n))
    highs = np.maximum(prices, opens) + daily_range * np.abs(np.random.normal(0.5, 0.2, n))
    lows = np.minimum(prices, opens) - daily_range * np.abs(np.random.normal(0.5, 0.2, n))
    volumes = np.abs(np.random.normal(2_500_000, 800_000, n)).astype(int)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        }
    )
    return df


def generate_volatility_series(price_df, window=20):
    """Calculate realized volatility and generate predicted volatility."""
    np.random.seed(99)
    close = price_df["close"].values
    log_returns = np.diff(np.log(close))
    n = len(log_returns)

    realized_vol = np.array(
        [np.std(log_returns[max(0, i - window) : i]) * np.sqrt(252) for i in range(1, n + 1)]
    )
    realized_vol = np.append(realized_vol[0], realized_vol)

    # LSTM predictions with slight lag and noise
    lstm_pred = np.roll(realized_vol, 1) * (1 + np.random.normal(0, 0.05, len(realized_vol)))
    lstm_pred[0] = realized_vol[0]

    # ARIMA predictions - smoother, less accurate
    arima_pred = pd.Series(realized_vol).rolling(5).mean().fillna(method="bfill").values
    arima_pred *= 1 + np.random.normal(0, 0.08, len(arima_pred))

    # GBM predictions
    gbm_pred = realized_vol * (1 + np.random.normal(0, 0.06, len(realized_vol)))

    # Data Reliability Score (mock - asset-independent for mock data)
    data_reliability = 85 + np.random.normal(0, 5, len(realized_vol))
    data_reliability = np.clip(data_reliability, 0, 100)

    df = price_df.copy()
    df["realized_vol"] = realized_vol
    df["lstm_pred"] = np.clip(lstm_pred, 0.01, 1.0)
    df["arima_pred"] = np.clip(arima_pred, 0.01, 1.0)
    df["gbm_pred"] = np.clip(gbm_pred, 0.01, 1.0)
    df["data_reliability"] = data_reliability
    return df


def generate_future_forecast(n_days=30, last_vol=0.18, seed=77):
    """Generate future volatility forecast with confidence intervals."""
    np.random.seed(seed)
    dates = pd.date_range(start=datetime.today() + timedelta(days=1), periods=n_days, freq="B")

    # Mean-reverting forecast
    long_run_vol = 0.20
    forecast = [last_vol]
    for _ in range(n_days - 1):
        prev = forecast[-1]
        shock = np.random.normal(0, 0.005)
        new_val = prev + 0.1 * (long_run_vol - prev) + shock
        forecast.append(max(0.01, new_val))

    forecast = np.array(forecast)
    uncertainty = np.linspace(0.01, 0.06, n_days)

    df = pd.DataFrame(
        {
            "date": dates,
            "forecast": forecast,
            "upper_80": forecast + 1.28 * uncertainty,
            "lower_80": forecast - 1.28 * uncertainty,
            "upper_95": forecast + 1.96 * uncertainty,
            "lower_95": forecast - 1.96 * uncertainty,
        }
    )
    return df


def generate_model_metrics():
    """Generate model comparison metrics.

    Prefer loading a CSV `data/volatility_model_comparison.csv` if available
    (columns: Model,Accuracy,Precision,Recall,F1-Score with values in 0-1).
    Returns a dict keyed by model name with percentage metrics and a color.
    Falls back to the embedded summary if the file is not present or parsing fails.
    """
    csv_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "volatility_model_comparison.csv")
    )
    palette = ["#00d4aa", "#4fc3f7", "#ffd54f", "#ff8a65", "#9b59b6", "#e67e22", "#2ecc71", "#e74c3c"]
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            out = {}
            for i, row in df.iterrows():
                name = str(row["Model"]).strip()
                # Some CSVs store metrics in 0-1 range; convert to percentages if so
                acc = float(row.get("Accuracy", 0.0))
                prec = float(row.get("Precision", 0.0))
                rec = float(row.get("Recall", 0.0))
                f1 = float(row.get("F1-Score", row.get("F1", 0.0)))
                if acc <= 1.0 and acc > 0.0:
                    acc *= 100.0
                if prec <= 1.0 and prec > 0.0:
                    prec *= 100.0
                if rec <= 1.0 and rec > 0.0:
                    rec *= 100.0
                if f1 <= 1.0 and f1 > 0.0:
                    f1 *= 100.0

                color = palette[i % len(palette)]
                out[name] = {
                    "Accuracy": round(acc, 2),
                    "Precision": round(prec, 2),
                    "Recall": round(rec, 2),
                    "F1": round(f1, 2),
                    "color": color,
                }
            if out:
                return out
    except Exception:
        # fall through to default
        pass

    # Fallback (kept for backward compatibility)
    models = {
        "Logistic Regression": {
            "Accuracy": 71.63,
            "Precision": 71.76,
            "Recall": 87.77,
            "F1": 78.96,
            "color": "#00d4aa",
        },
        "Gradient Boosting": {
            "Accuracy": 71.30,
            "Precision": 76.04,
            "Recall": 70.02,
            "F1": 72.91,
            "color": "#4fc3f7",
        },
        "Random Forest": {
            "Accuracy": 67.46,
            "Precision": 66.73,
            "Recall": 81.77,
            "F1": 73.49,
            "color": "#ffd54f",
        },
        "Support Vector Machine": {
            "Accuracy": 53.84,
            "Precision": 54.74,
            "Recall": 94.24,
            "F1": 69.25,
            "color": "#ff8a65",
        },
    }
    return models


def generate_training_curves(epochs=100):
    """Generate training/validation loss curves."""
    np.random.seed(55)
    x = np.arange(1, epochs + 1)

    # LSTM loss curves
    lstm_train = 0.08 * np.exp(-0.04 * x) + 0.012 + np.random.normal(0, 0.001, epochs)
    lstm_val = 0.09 * np.exp(-0.035 * x) + 0.015 + np.random.normal(0, 0.0015, epochs)

    df = pd.DataFrame(
        {
            "epoch": x,
            "lstm_train_loss": np.clip(lstm_train, 0.005, 0.1),
            "lstm_val_loss": np.clip(lstm_val, 0.008, 0.11),
        }
    )
    return df


def generate_data_quality(asset_seed=33):
    """Generate data quality metrics over time (asset-specific)."""
    np.random.seed(asset_seed)
    n = 200
    dates = pd.date_range(end=datetime.today(), periods=n, freq="B")

    completeness = np.clip(0.98 - 0.005 * np.random.exponential(1, n), 0.7, 1.0)
    consistency = np.clip(0.95 + np.random.normal(0, 0.02, n), 0.75, 1.0)
    timeliness = np.clip(0.97 + np.random.normal(0, 0.015, n), 0.8, 1.0)

    # Introduce some bad periods
    bad_idx = np.random.choice(n, 15, replace=False)
    completeness[bad_idx] -= np.random.uniform(0.05, 0.15, 15)
    consistency[bad_idx] -= np.random.uniform(0.03, 0.10, 15)

    overall = (completeness * 0.4 + consistency * 0.35 + timeliness * 0.25) * 100

    df = pd.DataFrame(
        {
            "date": dates,
            "completeness": np.clip(completeness, 0, 1),
            "consistency": np.clip(consistency, 0, 1),
            "timeliness": np.clip(timeliness, 0, 1),
            "overall_score": np.clip(overall, 0, 100),
            "missing_pct": np.clip(1 - completeness, 0, 0.3) * 100,
        }
    )
    return df


def generate_outliers(price_df):
    """Generate outlier detection results."""
    np.random.seed(11)
    returns = price_df["close"].pct_change().dropna()
    mean_r = returns.mean()
    std_r = returns.std()
    z_scores = (returns - mean_r) / std_r
    outliers = np.abs(z_scores) > 2.5

    df = pd.DataFrame(
        {
            "date": returns.index,
            "return": returns.values,
            "z_score": z_scores.values,
            "is_outlier": outliers.values,
        }
    )
    return df


def get_overview_stats(vol_df, quality_df):
    """Compute summary statistics for the overview panel."""
    current_vol = vol_df["realized_vol"].iloc[-1]
    vol_30d_ago = vol_df["realized_vol"].iloc[-21]
    vol_change = (current_vol - vol_30d_ago) / vol_30d_ago * 100

    quality_score = quality_df["overall_score"].iloc[-1]
    total_records = len(vol_df)

    # Risk level
    if current_vol < 0.15:
        risk_level = "LOW"
        risk_color = "#00d4aa"
    elif current_vol < 0.30:
        risk_level = "MEDIUM"
        risk_color = "#ffd54f"
    else:
        risk_level = "HIGH"
        risk_color = "#ff6b6b"

    return {
        "current_vol": current_vol,
        "vol_change_pct": vol_change,
        "quality_score": quality_score,
        "total_records": total_records,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "lstm_rmse": 0.0182,
        "lstm_r2": 0.912,
        "missing_pct": quality_df["missing_pct"].mean(),
    }


AI_INSIGHTS = [
    {
        "title": "Volatility Clustering Detected",
        "body": "The LSTM model identifies significant volatility clustering over the past 15 trading days. Periods of elevated variance tend to persist for 8-12 sessions before mean reversion occurs.",
        "severity": "warning",
        "icon": "chart-line",
    },
    {
        "title": "Data Quality Degradation",
        "body": "Completeness scores dropped below 90% on 3 occasions in the last month. These gaps correlate with a 6.2% increase in LSTM prediction error, suggesting pre-processing improvements are warranted.",
        "severity": "error",
        "icon": "database",
    },
    {
        "title": "Model Convergence Stable",
        "body": "LSTM validation loss has plateaued at 0.015, indicating the model has reached optimal convergence. GBM shows competitive accuracy with significantly lower training time.",
        "severity": "success",
        "icon": "cpu",
    },
    {
        "title": "Elevated Tail Risk",
        "body": "Z-score analysis flags 23 outlier return events in the dataset, concentrated during high-volume sessions. The 95th percentile daily move exceeds 3.1%, indicating fat-tailed distribution.",
        "severity": "warning",
        "icon": "alert-triangle",
    },
]
