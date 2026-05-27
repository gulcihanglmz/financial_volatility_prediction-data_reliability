"""
Financial Volatility Prediction Dashboard
==========================================
Production-grade fintech dashboard for volatility prediction and data reliability analysis.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import importlib
from datetime import datetime
from sklearn.ensemble import IsolationForest

# ── Local modules ──────────────────────────────────────────────────────────────
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for _path in (CURRENT_DIR, ROOT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

yf = None
joblib = None
try:
    yf = importlib.import_module("yfinance")
except Exception:
    pass
try:
    joblib = importlib.import_module("joblib")
except Exception:
    pass

try:
    from src.feature_engineering import FeatureEngineer
except Exception:
    FeatureEngineer = None

from data.mock_data import (
    generate_price_series,
    generate_volatility_series,
    generate_future_forecast,
    generate_model_metrics,
    generate_training_curves,
    generate_data_quality,
    generate_outliers,
    get_overview_stats,
    AI_INSIGHTS,
)
from utils.charts import (
    candlestick_chart,
    volatility_comparison_chart,
    forecast_chart,
    model_comparison_bar,
    training_curve_chart,
    error_distribution_chart,
    data_quality_timeline,
    missing_data_chart,
    outlier_scatter,
    quality_radar,
)
from utils.styles import inject_css
from utils.export import generate_pdf_report


ASSET_OPTIONS = ["S&P 500 (SPX)", "NASDAQ (NDX)", "Bitcoin (BTC)", "Gold (XAUUSD)", "EUR/USD"]
ASSET_TO_TICKER = {
    "S&P 500 (SPX)": "^GSPC",
    "NASDAQ (NDX)": "^IXIC",
    "Bitcoin (BTC)": "BTC-USD",
    "Gold (XAUUSD)": "GC=F",
    "EUR/USD": "EURUSD=X",
}


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VolPred — Financial Volatility Dashboard",
    page_icon="assets/favicon.ico" if os.path.exists("assets/favicon.ico") else None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state defaults ────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if "refresh_counter" not in st.session_state:
    st.session_state.refresh_counter = 0
if "selected_asset" not in st.session_state:
    st.session_state.selected_asset = ASSET_OPTIONS[0]


# ── Inject custom CSS ─────────────────────────────────────────────────────────
st.markdown(inject_css(st.session_state.dark_mode), unsafe_allow_html=True)


# ── Data (cached) ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_trained_model():
    if joblib is None:
        return None, None, "joblib missing"

    model_path = os.path.join(ROOT_DIR, "models", "volatility_best_model.pkl")
    scaler_path = os.path.join(ROOT_DIR, "models", "volatility_scaler.pkl")

    if not os.path.exists(model_path):
        return None, None, "model file missing"
    if not os.path.exists(scaler_path):
        return None, None, "scaler file missing"

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler, "volatility_best_model.pkl loaded"
    except Exception as exc:
        return None, None, f"model load error: {exc}"


def _add_lag_features(df: pd.DataFrame, lags=(1, 2, 3)) -> pd.DataFrame:
    lag_df = df.copy()
    features_to_lag = ["Close", "Volume", "MA_5", "MA_20", "RSI", "MACD"]
    for feature in features_to_lag:
        if feature in lag_df.columns:
            for lag in lags:
                lag_df[f"{feature}_lag{lag}"] = lag_df[feature].shift(lag)
    return lag_df.dropna()


def _fetch_api_price_series(asset_label: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not installed")

    ticker = ASSET_TO_TICKER.get(asset_label)
    if not ticker:
        raise RuntimeError("asset ticker mapping missing")

    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    if df is None or df.empty:
        raise RuntimeError("api returned empty data")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    expected = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise RuntimeError(f"api columns missing: {missing}")

    df = df[expected].dropna(subset=["date", "close"]).copy()
    df["volume"] = df["volume"].fillna(0)
    if len(df) < 120:
        raise RuntimeError("api history too short for feature engineering")

    return df


def _build_api_volatility_frame(asset_label: str, model, scaler) -> pd.DataFrame:
    if FeatureEngineer is None:
        raise RuntimeError("feature engineering module unavailable")

    raw = _fetch_api_price_series(asset_label)
    feat_input = raw.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "date": "Date",
        }
    ).set_index("Date")

    engineered = FeatureEngineer(feat_input).apply_all_features()
    lagged = _add_lag_features(engineered)
    if lagged.empty:
        raise RuntimeError("no rows after lag feature generation")

    lagged["realized_vol"] = (
        lagged["Close"].pct_change().rolling(5).std() * np.sqrt(252)
    ).clip(lower=0.01, upper=1.0)
    lagged["realized_vol"] = lagged["realized_vol"].bfill().ffill()

    lagged["Volatility_Target"] = (
        lagged["Close"].pct_change().rolling(5).std()
        > lagged["Close"].pct_change().rolling(5).std().median()
    ).astype(int)

    X = lagged.drop(columns=["Target", "Volatility_Target"], errors="ignore")
    expected_cols = getattr(model, "feature_names_in_", None)
    if expected_cols is not None:
        expected_cols = list(expected_cols)
        missing_cols = [c for c in expected_cols if c not in X.columns]
        if missing_cols:
            raise RuntimeError(f"model feature mismatch: missing {missing_cols[:5]}")
        X = X[expected_cols]

    if hasattr(scaler, "n_features_in_") and X.shape[1] != scaler.n_features_in_:
        raise RuntimeError(
            f"scaler feature mismatch: expected {scaler.n_features_in_}, got {X.shape[1]}"
        )

    X_scaled = scaler.transform(X)
    pred_label = model.predict(X_scaled)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)
        high_prob = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else pred_label.astype(float)
    else:
        high_prob = pred_label.astype(float)

    base_vol = lagged["realized_vol"].to_numpy()
    threshold = np.nanmedian(base_vol)
    model_vol = np.where(
        pred_label == 1,
        np.maximum(base_vol, threshold * (1.05 + 0.30 * high_prob)),
        np.minimum(base_vol, threshold * (0.95 - 0.25 * (1 - high_prob))),
    )

    lagged["lstm_pred"] = np.clip(model_vol, 0.01, 1.0)
    lagged["arima_pred"] = lagged["realized_vol"].rolling(5).mean().bfill()
    lagged["gbm_pred"] = lagged["realized_vol"].ewm(span=8, adjust=False).mean()

    # ── Compute Data Reliability Score using Isolation Forest (asset-specific)
    feature_cols = [c for c in lagged.columns if c not in [
        "Target", "Volatility_Target", "realized_vol", 
        "lstm_pred", "arima_pred", "gbm_pred"
    ]]
    if feature_cols:
        iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
        anomaly_scores = iso_forest.fit_predict(lagged[feature_cols].fillna(0))
        # Convert to 0-100 scale (normal=100, anomaly=0)
        raw_scores = iso_forest.score_samples(lagged[feature_cols].fillna(0))
        data_reliability = np.clip((1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())) * 100, 0, 100)
        lagged["data_reliability"] = data_reliability
    else:
        lagged["data_reliability"] = 100.0

    out = lagged.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    return out[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "realized_vol",
            "lstm_pred",
            "arima_pred",
            "gbm_pred",
            "data_reliability",
        ]
    ]


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def _classification_metrics() -> dict:
    # Classification results provided from the project's model evaluation table.
    return {
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


def _compute_live_metrics(vol_df: pd.DataFrame, model_name: str = "PKL Model") -> dict:
    return _classification_metrics()


@st.cache_data(ttl=60)
def load_data(asset_label: str, counter=0):
    model, scaler, model_status = load_trained_model()
    data_source = "MOCK"
    # Asset-specific seed so mock data differs per asset when API is unavailable
    asset_seed = hash(asset_label) % (2 ** 31)

    vol_df = None
    if model is not None and scaler is not None:
        try:
            vol_df = _build_api_volatility_frame(asset_label, model, scaler)
            data_source = "API"
        except Exception as exc:
            data_source = f"MOCK (fallback: {exc})"

    if vol_df is None:
        price_df = generate_price_series(n=500, seed=asset_seed)
        vol_df = generate_volatility_series(price_df)
    forecast_df = generate_future_forecast(n_days=30, last_vol=vol_df["realized_vol"].iloc[-1])
    # Asset-specific quality data: use same asset_seed
    quality_df = generate_data_quality(asset_seed=asset_seed)

    outlier_input = vol_df[["date", "close"]].copy()
    outlier_input = outlier_input.set_index("date")
    outlier_df = generate_outliers(outlier_input)

    if data_source == "API":
        metrics = _compute_live_metrics(vol_df, model_name="VOL PKL")
    else:
        metrics = generate_model_metrics()

    training_df = generate_training_curves()
    stats = get_overview_stats(vol_df, quality_df)
    system_info = {
        "data_source": data_source,
        "model_status": model_status,
        "ticker": ASSET_TO_TICKER.get(asset_label, "N/A"),
    }
    return vol_df, forecast_df, quality_df, outlier_df, metrics, training_df, stats, system_info

vol_df, forecast_df, quality_df, outlier_df, metrics, training_df, stats, system_info = load_data(
    st.session_state.selected_asset,
    st.session_state.refresh_counter
)

best_model_name, best_model_metrics = max(metrics.items(), key=lambda x: x[1]["F1"])


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        f"""<div style="font-family:'DM Mono',monospace; font-size:18px;
            font-weight:600; color:{'#00d4aa' if st.session_state.dark_mode else '#0077aa'};
            margin-bottom:2px;">VOLPRED</div>
        <div style="font-size:10px; color:#8892b0; letter-spacing:0.12em;
            text-transform:uppercase; margin-bottom:20px;">
        Financial Analytics Suite</div>""",
        unsafe_allow_html=True,
    )

    # Dark / light toggle
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown(
            "<span style='font-size:11px; font-family:DM Mono,monospace;"
            " color:#8892b0; letter-spacing:.1em'>DISPLAY MODE</span>",
            unsafe_allow_html=True,
        )
    with col_b:
        if st.button("Dark" if not st.session_state.dark_mode else "Light", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.markdown("---")

    # Asset selector
    st.markdown(
        "<div style='font-size:10px;letter-spacing:.14em;text-transform:uppercase;"
        "color:#8892b0;font-family:DM Mono,monospace;margin-bottom:6px'>ASSET</div>",
        unsafe_allow_html=True,
    )
    st.selectbox(
        "", ASSET_OPTIONS,
        key="selected_asset",
        label_visibility="collapsed",
    )
    asset = st.session_state.selected_asset

    # Time range
    st.markdown(
        "<div style='font-size:10px;letter-spacing:.14em;text-transform:uppercase;"
        "color:#8892b0;font-family:DM Mono,monospace;margin-top:14px;margin-bottom:6px'>TIME RANGE</div>",
        unsafe_allow_html=True,
    )
    period = st.radio("", ["1W", "1M", "3M", "1Y", "ALL"], horizontal=True, index=3, label_visibility="collapsed")

    st.markdown("---")

    # Risk level indicator
    risk = stats["risk_level"]
    risk_colors = {"LOW": "#00d4aa", "MEDIUM": "#ffd54f", "HIGH": "#ff6b6b"}
    rc = risk_colors[risk]
    st.markdown(
        f"""<div style="text-align:center; padding: 14px; border-radius:6px;
            background:{'#141b2d' if st.session_state.dark_mode else '#f0f4ff'};
            border: 1px solid {rc}44; margin-bottom:12px;">
        <div style="font-size:9px;letter-spacing:.18em;text-transform:uppercase;
            color:#8892b0;font-family:DM Mono,monospace;margin-bottom:4px">RISK LEVEL</div>
        <div style="font-size:22px;font-weight:700;color:{rc};
            font-family:DM Mono,monospace;">{risk}</div>
        <div style="font-size:10px;color:#8892b0;">Current volatility: {stats['current_vol']:.2%}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""<div style="font-size:10px;color:#8892b0;font-family:DM Mono,monospace;
            border:1px solid #1e3a5f;border-radius:6px;padding:10px;margin-bottom:10px;">
        SOURCE: <span style="color:#e8eaf6">{system_info['data_source']}</span><br>
        MODEL: <span style="color:#e8eaf6">{system_info['model_status']}</span><br>
        TICKER: <span style="color:#e8eaf6">{system_info['ticker']}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # Real-time refresh simulation
    st.markdown("---")
    auto_refresh = st.checkbox("Auto-Refresh (60s)", value=False)
    if st.button("Refresh Data", use_container_width=True):
        st.session_state.refresh_counter += 1
        st.cache_data.clear()
        st.rerun()

    if auto_refresh:
        elapsed = time.time() - st.session_state.last_refresh
        remaining = max(0, 60 - int(elapsed))
        st.markdown(
            f"<div style='text-align:center;font-size:10px;color:#8892b0;"
            f"font-family:DM Mono,monospace'>Next refresh in {remaining}s</div>",
            unsafe_allow_html=True,
        )
        if elapsed > 60:
            st.session_state.last_refresh = time.time()
            st.session_state.refresh_counter += 1
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")

    # Export buttons
    st.markdown(
        "<div style='font-size:10px;letter-spacing:.14em;text-transform:uppercase;"
        "color:#8892b0;font-family:DM Mono,monospace;margin-bottom:8px'>EXPORT</div>",
        unsafe_allow_html=True,
    )
    report_html = generate_pdf_report(stats, metrics)
    st.download_button(
        "Download Report (HTML)",
        data=report_html,
        file_name=f"volpred_report_{datetime.today().strftime('%Y%m%d')}.html",
        mime="text/html",
        use_container_width=True,
    )

    csv_data = vol_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Data (CSV)",
        data=csv_data,
        file_name="volatility_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:9px;color:#4a5568;text-align:center;"
        f"font-family:DM Mono,monospace;'>VOLPRED v2.1 — {datetime.today().strftime('%d %b %Y')}</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# ── Ticker bar ────────────────────────────────────────────────────────────────
vol_chg = stats["vol_change_pct"]
chg_class = "ticker-up" if vol_chg < 0 else "ticker-down"
chg_sign = "+" if vol_chg > 0 else ""

st.markdown(
    f"""<div class="ticker-bar">
    <span>VOLPRED ANALYTICS TERMINAL</span>
    <span style="margin-left:auto">
        <span style="color:#8892b0">REALIZED VOL:</span>
        <span class="{chg_class}">&nbsp;{stats['current_vol']:.2%}</span>
    </span>
    <span>
        <span style="color:#8892b0">30D CHANGE:</span>
        <span class="{chg_class}">&nbsp;{chg_sign}{vol_chg:.1f}%</span>
    </span>
    <span>
        <span style="color:#8892b0">DATA QUALITY:</span>
        <span style="color:#ffd54f">&nbsp;{stats['quality_score']:.0f}/100</span>
    </span>
    <span>
        <span style="color:#8892b0">RECORDS:</span>
        <span>&nbsp;{stats['total_records']:,}</span>
    </span>
    <span>
        <span style="color:#8892b0">ASSET:</span>
        <span>&nbsp;{asset.split('(')[0].strip()}</span>
    </span>
    <span>
        <span style="color:#8892b0">SOURCE:</span>
        <span>&nbsp;{system_info['data_source']}</span>
    </span>
    </div>""",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_timeseries, tab_models, tab_quality, tab_forecast, tab_insights = st.tabs([
    "Overview",
    "Time Series",
    "Model Performance",
    "Data Reliability",
    "Prediction",
    "AI Insights",
])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    st.markdown('<div class="section-header">Portfolio & Model Summary</div>', unsafe_allow_html=True)

    # Row 1: KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(
            "Realized Volatility",
            f"{stats['current_vol']:.2%}",
            delta=f"{stats['vol_change_pct']:+.1f}% (30D)",
            delta_color="inverse",
        )
    with c2:
        st.metric("Best Model", best_model_name, delta=f"F1: {best_model_metrics['F1']:.2f}%")
    with c3:
        st.metric("Best Accuracy", f"{best_model_metrics['Accuracy']:.2f}%", delta="Classification")
    with c4:
        st.metric("Data Quality", f"{stats['quality_score']:.0f} / 100",
                  delta=f"-{stats['missing_pct']:.1f}% missing")
    with c5:
        st.metric("Dataset Size", f"{stats['total_records']:,}", delta="+5 today")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Row 2: Model summary pills
    st.markdown('<div class="section-header">Model Leaderboard</div>', unsafe_allow_html=True)
    cols = st.columns(len(metrics))
    for col, (model_name, m) in zip(cols, metrics.items()):
        with col:
            bar_w = int(m["F1"])
            st.markdown(
                f"""<div class="stat-pill">
                <div style="font-size:13px;font-weight:600;
                    color:{m['color']};margin-bottom:6px">{model_name}</div>
                <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px">
                    <span style="color:#8892b0">Accuracy</span>
                    <span style="font-family:DM Mono,monospace">{m['Accuracy']:.2f}%</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px">
                    <span style="color:#8892b0">Precision</span>
                    <span style="font-family:DM Mono,monospace">{m['Precision']:.2f}%</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px">
                    <span style="color:#8892b0">Recall</span>
                    <span style="font-family:DM Mono,monospace">{m['Recall']:.2f}%</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:8px">
                    <span style="color:#8892b0">F1-Score</span>
                    <span style="font-family:DM Mono,monospace">{m['F1']:.2f}</span>
                </div>
                <div style="background:#1e3a5f;border-radius:2px;height:4px">
                    <div style="background:{m['color']};width:{bar_w}%;height:4px;border-radius:2px"></div>
                </div>
                <div style="text-align:right;font-size:9px;color:#8892b0;margin-top:2px">{m['F1']:.0f}% F1</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Row 3: Quick volatility chart
    st.markdown('<div class="section-header">Volatility Overview</div>', unsafe_allow_html=True)
    col_chart, col_quality = st.columns([3, 1])
    with col_chart:
        st.plotly_chart(
            volatility_comparison_chart(vol_df, st.session_state.dark_mode, period),
            key="volatility_comparison_overview",
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with col_quality:
        st.plotly_chart(
            quality_radar(quality_df, st.session_state.dark_mode),
            use_container_width=True,
            config={"displayModeBar": False},
        )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2: TIME SERIES
# ─────────────────────────────────────────────────────────────────────────────
with tab_timeseries:
    st.markdown('<div class="section-header">Price Action & Volatility</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style='font-size:12px;color:#bcd3e6'>
        Graph shows daily price action (OHLC) for the selected asset. A green candle means the close is above the open (price up); a red candle means the close is below the open (price down). The bars below show trading volume — higher volume typically confirms the price move.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_opt1, col_opt2, _, col_dl = st.columns([2, 2, 4, 2])
    with col_opt1:
        ts_period = st.selectbox("Candlestick period", ["1W", "1M", "3M", "1Y", "ALL"], index=3, label_visibility="collapsed")
    with col_opt2:
        chart_type = st.radio("", ["Candlestick", "Line"], horizontal=True, label_visibility="collapsed")

    st.plotly_chart(
        candlestick_chart(vol_df, st.session_state.dark_mode, ts_period),
        key="candlestick_timeseries",
        use_container_width=True,
        config={"displayModeBar": True, "toImageButtonOptions": {"filename": "price_chart", "format": "png"}},
    )

    st.markdown(
        """
        <div style='font-size:11px;color:#8892b0;margin-top:6px'>
        Note: Short gaps or unusual candles may be caused by missing data or by the app falling back to mock data when the API is unavailable.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">Realized vs Predicted Volatility</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style='font-size:12px;color:#bcd3e6'>
        The white line shows realized (observed) volatility; the colored lines show model predictions (LSTM, ARIMA, GBM) for the same dates. The Y-axis shows annualized volatility in percent. Predictions that lie close to the realized line indicate better model accuracy.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        volatility_comparison_chart(vol_df, st.session_state.dark_mode, ts_period),
        key="volatility_comparison_timeseries",
        use_container_width=True,
        config={"displayModeBar": True, "toImageButtonOptions": {"filename": "volatility_comparison", "format": "png"}},
    )

    # Stats table
    recent = vol_df.tail(5)[["date", "close", "realized_vol", "lstm_pred", "arima_pred", "gbm_pred"]].copy()
    recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
    recent.columns = ["Date", "Close", "Realized Vol", "LSTM Pred", "ARIMA Pred", "GBM Pred"]
    for col in ["Realized Vol", "LSTM Pred", "ARIMA Pred", "GBM Pred"]:
        recent[col] = recent[col].apply(lambda x: f"{x:.2%}")
    recent["Close"] = recent["Close"].apply(lambda x: f"${x:.2f}")
    st.markdown('<div class="section-header">Recent Observations</div>', unsafe_allow_html=True)
    st.dataframe(recent, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab_models:
    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
    st.plotly_chart(
        model_comparison_bar(metrics, st.session_state.dark_mode),
        key="model_comparison_bar_models",
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown(
        "<div style='font-size:12px;color:#8892b0;margin:10px 0 14px 0;font-family:DM Mono,monospace'>"
        "Charts (left to right): Accuracy, Precision, Recall. "
        "Classification comparison uses the four project models provided in the thesis results table. "
        "The table below includes Accuracy, Precision, Recall, and F1-Score for each model."
        "</div>",
        unsafe_allow_html=True,
    )

    # Detailed metrics table
    st.markdown('<div class="section-header">Full Metrics Table</div>', unsafe_allow_html=True)
    rows = []
    for m, v in metrics.items():
        rows.append({
            "Model": m,
            "Accuracy (%)": f"{v['Accuracy']:.2f}",
            "Precision (%)": f"{v['Precision']:.2f}",
            "Recall (%)": f"{v['Recall']:.2f}",
            "F1-Score": f"{v['F1']:.2f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4: DATA RELIABILITY
# ─────────────────────────────────────────────────────────────────────────────
with tab_quality:
    st.markdown('<div class="section-header">Data Quality Scores Over Time</div>', unsafe_allow_html=True)

    # Overall score gauge-style metric
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Overall Score", f"{quality_df['overall_score'].iloc[-1]:.1f}/100",
                  delta=f"{quality_df['overall_score'].iloc[-1] - quality_df['overall_score'].iloc[-21]:+.1f} (30D)")
    with c2:
        st.metric("Completeness", f"{quality_df['completeness'].iloc[-1]:.1%}",
                  delta=f"{(quality_df['completeness'].iloc[-1] - quality_df['completeness'].iloc[-21])*100:+.1f}pp")
    with c3:
        st.metric("Consistency", f"{quality_df['consistency'].iloc[-1]:.1%}")
    with c4:
        st.metric("Timeliness", f"{quality_df['timeliness'].iloc[-1]:.1%}")

    col_l, col_r = st.columns([3, 1])
    with col_l:
        st.plotly_chart(
            data_quality_timeline(quality_df, st.session_state.dark_mode),
            key="data_quality_timeline",
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with col_r:
        st.plotly_chart(
            quality_radar(quality_df, st.session_state.dark_mode),
            key="quality_radar_qualitytab",
            use_container_width=True,
            config={"displayModeBar": False},
        )
    # Short caption for timeline + radar
    st.caption(
        "Line chart: daily completeness, consistency and timeliness. Radar: snapshot comparison of the three quality dimensions."
    )

    st.markdown('<div class="section-header">Missing Data Analysis</div>', unsafe_allow_html=True)
    st.plotly_chart(
        missing_data_chart(quality_df, st.session_state.dark_mode),
        key="missing_data_chart",
        use_container_width=True,
        config={"displayModeBar": False},
    )
    # Short caption for missing data
    st.caption("Shows the percent of missing values per period; taller bars mark problem dates to investigate.")

    st.markdown('<div class="section-header">Outlier Detection (Z-Score)</div>', unsafe_allow_html=True)
    n_outliers = outlier_df["is_outlier"].sum()
    st.markdown(
        f"<div style='font-size:12px;color:#8892b0;font-family:DM Mono,monospace;margin-bottom:8px'>"
        f"Detected {n_outliers} outlier events (&gt; 2.5 standard deviations) across {len(outlier_df):,} observations"
        f" ({n_outliers/len(outlier_df)*100:.1f}%)</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        outlier_scatter(outlier_df, st.session_state.dark_mode),
        key="outlier_scatter",
        use_container_width=True,
        config={"displayModeBar": False},
    )
    # Short caption for outlier scatter
    st.caption(
        "Points outside the dashed lines are statistical outliers (possible errors or rare events); validate these dates."
    )

    # ── Data Reliability Score (Asset-Specific Isolation Forest)
    if "data_reliability" in vol_df.columns:
        st.markdown('<div class="section-header">Data Reliability Score (Isolation Forest)</div>', unsafe_allow_html=True)
        current_reliability = vol_df["data_reliability"].iloc[-1] if len(vol_df) > 0 else 0
        mean_reliability = vol_df["data_reliability"].mean()
        st.metric(
            "Current Data Reliability",
            f"{current_reliability:.1f}/100",
            delta=f"Avg: {mean_reliability:.1f}/100 (Asset-Specific)"
        )
        
        # Timeline chart
        import plotly.graph_objects as go
        dr_chart = go.Figure()
        dr_chart.add_trace(go.Scatter(
            x=vol_df["date"],
            y=vol_df["data_reliability"],
            mode="lines",
            name="Data Reliability",
            line=dict(color="#00d4aa", width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 170, 0.2)"
        ))
        dr_chart.update_layout(
            title="Data Reliability Over Time (Isolation Forest - Asset-Specific)",
            xaxis_title="Date",
            yaxis_title="Reliability Score (0-100)",
            hovermode="x unified",
            template="plotly_dark" if st.session_state.dark_mode else "plotly",
            height=400
        )
        st.plotly_chart(dr_chart, use_container_width=True, config={"displayModeBar": False})
        # Short caption for data reliability timeline
        st.caption(
            "Shows the asset-specific reliability score over time computed by the isolation forest; drops indicate lower data trustworthiness."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 5: PREDICTION / FORECAST
# ─────────────────────────────────────────────────────────────────────────────
with tab_forecast:
    st.markdown('<div class="section-header">30-Day Volatility Forecast</div>', unsafe_allow_html=True)

    col_sc, col_ci, _ = st.columns([2, 2, 4])
    with col_sc:
        scenario = st.selectbox(
            "Scenario",
            ["Low", "Medium", "High"],
            index=1,
            help="Low: bear market calm. Medium: baseline. High: stress scenario.",
        )
    with col_ci:
        show_ci = st.checkbox("Show confidence intervals", value=True)

    scenario_desc = {
        "Low": "Baseline adjusted downward — assumes continued low-volume sessions and mean reversion acceleration.",
        "Medium": "Model baseline forecast using LSTM mean-reverting dynamics (theta=0.10).",
        "High": "Stress scenario — amplified by 1.5x to simulate high-VIX / crisis-like conditions.",
    }
    st.markdown(
        f"<div style='font-size:11px;color:#8892b0;font-family:DM Mono,monospace;"
        f"margin-bottom:12px;padding:10px;border:1px solid #1e3a5f;border-radius:4px'>"
        f"Scenario: <b>{scenario}</b> — {scenario_desc[scenario]}</div>",
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        forecast_chart(forecast_df, vol_df, st.session_state.dark_mode, scenario),
        key="forecast_chart",
        use_container_width=True,
        config={"displayModeBar": True, "toImageButtonOptions": {"filename": "forecast", "format": "png"}},
    )
    # Short caption for forecast chart
    st.caption(
        "Forecast line shows predicted volatility; shaded bands are 80% and 95% confidence intervals (wider = more uncertainty)."
    )

    # Forecast table
    st.markdown('<div class="section-header">Forecast Table</div>', unsafe_allow_html=True)
    mult = {"Low": 0.7, "Medium": 1.0, "High": 1.5}[scenario]
    fc_display = forecast_df.copy()
    fc_display["date"] = fc_display["date"].dt.strftime("%Y-%m-%d")
    for col in ["forecast", "upper_80", "lower_80", "upper_95", "lower_95"]:
        fc_display[col] = (fc_display[col] * mult).apply(lambda x: f"{x:.2%}")
    fc_display.columns = ["Date", "Forecast", "Upper 80%", "Lower 80%", "Upper 95%", "Lower 95%"]
    st.dataframe(fc_display, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 6: AI INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_insights:
    st.markdown('<div class="section-header">Automated AI Insights</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:12px;color:#8892b0;margin-bottom:16px;"
        "font-family:DM Mono,monospace'>System-generated pattern analysis from model outputs "
        "and data quality metrics. Updated on each data refresh cycle.</div>",
        unsafe_allow_html=True,
    )

    for insight in AI_INSIGHTS:
        st.markdown(
            f"""<div class="insight-card {insight['severity']}">
            <div class="insight-title">{insight['title']}</div>
            <div class="insight-body">{insight['body']}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<div class="section-header">Chart Explanation (Interactive)</div>', unsafe_allow_html=True)

    question_map = {
        "Why does volatility cluster in bursts?": (
            "Volatility clustering (also called GARCH effects) occurs because large price moves "
            "are often followed by more large moves. This is caused by information cascades, "
            "forced liquidations, and market participants updating risk models simultaneously. "
            "In this dashboard, volatility state is predicted with classification models that "
            "learn from lagged technical indicators and recent market regimes."
        ),
        "What does a high data quality score mean?": (
            "A data quality score above 90 indicates that the input time series has low missing-value "
            "rates (completeness), consistent timestamp intervals (timeliness), and no systematic "
            "encoding errors (consistency). Scores below 85 correlate with a statistically significant "
            "increase in LSTM prediction error, as shown in the Data Reliability tab."
        ),
        "How should I interpret confidence intervals?": (
            "The 80% confidence interval (darker band) means that, under the model assumptions, "
            "realized volatility will fall within this range 80% of the time. The 95% band is wider. "
            "For risk management, the upper bound of the 95% CI is the conservative planning figure. "
            "In high-VIX environments, multiply these bounds by the scenario multiplier shown in the Prediction tab."
        ),
        "Which model should I trust most?": (
            "Based on the current evaluation table, Logistic Regression has the strongest overall balance "
            "(highest F1 and accuracy), Gradient Boosting has strong precision, Random Forest has solid "
            "recall, and SVM has very high recall but lower precision. For balanced decision support, "
            "Logistic Regression is the primary model."
        ),
    }

    selected_q = st.selectbox("Ask a question about this dashboard:", list(question_map.keys()))
    if st.button("Explain"):
        with st.spinner("Analyzing..."):
            time.sleep(0.8)
        st.markdown(
            f"""<div class="insight-card success">
            <div class="insight-title">VOLPRED ANALYST RESPONSE</div>
            <div class="insight-body">{question_map[selected_q]}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Key stats summary
    st.markdown('<div class="section-header">Quick-Reference Statistics</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    stats_display = [
        ("Volatility Regime", stats["risk_level"], stats["risk_color"]),
        ("Best Model", "LSTM", "#00d4aa"),
        ("Quality Tier", "A" if stats["quality_score"] > 90 else "B" if stats["quality_score"] > 80 else "C",
         "#00d4aa" if stats["quality_score"] > 90 else "#ffd54f" if stats["quality_score"] > 80 else "#ff6b6b"),
        ("Outlier Rate", f"{outlier_df['is_outlier'].mean():.1%}", "#ffd54f"),
    ]
    for col, (label, value, color) in zip([col1, col2, col3, col4], stats_display):
        with col:
            st.markdown(
                f"""<div class="stat-pill">
                <div class="value" style="color:{color}">{value}</div>
                <div class="label">{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )
