# dashboard.py

import os
import random
import sys
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.data_quality import DataQuality


st.set_page_config(
    page_title="Financial Volatility Decision Support System",
    layout="wide"
)

st.title("Financial Volatility Decision Support System")
st.caption("Next-Day Volatility Prediction with Dynamic Data Reliability Scoring")


@st.cache_resource
def load_model_files():
    model = joblib.load(r"D:\fintech\models\volatility_best_model.pkl")
    scaler = joblib.load(r"D:\fintech\models\volatility_scaler.pkl")
    return model, scaler


@st.cache_data
def prepare_dataset():
    loader = DataLoader(r"D:\fintech\data\nasdq.csv")
    raw_data = loader.load_csv()
    raw_data = loader.preprocess()

    engineer = FeatureEngineer(raw_data)
    feature_data = engineer.apply_all_features()

    quality = DataQuality(feature_data)
    quality.detect_anomalies()
    quality.calculate_reliability_score()

    return raw_data, quality.get_data()


def add_lag_features(df, lags=[1, 2, 3]):
    df = df.copy()
    lag_columns = ["Close", "Volume", "MA_5", "MA_20", "RSI", "MACD"]

    for col in lag_columns:
        if col in df.columns:
            for lag in lags:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

    return df.dropna()


def reliability_status(score):
    if score >= 90:
        return "High Reliability"
    elif score >= 70:
        return "Acceptable Reliability"
    elif score >= 50:
        return "Warning"
    return "Unreliable"


def decision_message(score):

    if score >= 70:
        return (
            "Prediction Accepted. "
            "Input data passed the reliability threshold."
        )
    elif score >= 50:
        return (
            "Prediction Generated With Warning. "
            "Input quality is moderate."
        )
    else:
        return (
            "Prediction Rejected. "
            "Input data failed the reliability threshold."
        )
    
def reliability_gauge(score):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100"},
            title={"text": "Data Reliability Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#00E5FF"},
                "steps": [
                    {"range": [0, 50], "color": "#7f1d1d"},
                    {"range": [50, 70], "color": "#9a3412"},
                    {"range": [70, 90], "color": "#854d0e"},
                    {"range": [90, 100], "color": "#14532d"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    return fig


try:
    model, scaler = load_model_files()
    raw_data, quality_data = prepare_dataset()
    model_data = add_lag_features(quality_data)

    with st.sidebar:
        if st.button("Generate New Incoming Data"):
            st.session_state["sample_seed"] = random.randint(1, 999999)
            
        st.markdown("### Data Source")
        st.info(
            "Training and evaluation: NASDAQ historical dataset\n\n"
            "Updated prediction input: Yahoo Finance API (yfinance)"
        )

        st.header("Incoming Market Data")
        mode = st.selectbox(
            "Select Input Mode",
            [
                "Latest Valid Market Data",
                "Random Historical Sample",
                "High Reliability Historical Sample",
                "Low Reliability Historical Sample"
            ]
        )

    if mode == "Latest Valid Market Data":
        valid_latest_data = model_data[
            model_data["reliability_score"].notna() &
            (model_data["reliability_score"] >= 50) &
            model_data["Close"].notna()
        ]

        selected_row = valid_latest_data.tail(1)

    elif mode == "Random Historical Sample":
        if "sample_seed" not in st.session_state:
            st.session_state["sample_seed"] = 42

        selected_row = model_data.sample(1, random_state=st.session_state["sample_seed"])
    
    elif mode == "High Reliability Historical Sample":
        high_data = model_data[model_data["reliability_score"] >= 90]
        selected_row = high_data.sample(1) if not high_data.empty else model_data.sort_values("reliability_score", ascending=False).head(1)
    else:
        low_data = model_data[model_data["reliability_score"] < 70]
        selected_row = low_data.sample(1) if not low_data.empty else model_data.sort_values("reliability_score", ascending=True).head(1)

    selected_date = selected_row.index[0]
    selected_date_str = selected_date.strftime("%Y-%m-%d")

    reliability_score = float(selected_row["reliability_score"].iloc[0])
    reliability_label = reliability_status(reliability_score)

    exclude_cols = [
        "Target",
        "Volatility_Target",
        "Log_Return",
        "Rolling_Volatility",
        "anomaly_score",
        "is_anomaly",
        "reliability_score"
    ]

    feature_columns = [col for col in model_data.columns if col not in exclude_cols]

    X_input = selected_row[feature_columns]
    X_scaled = scaler.transform(X_input)

    prediction = model.predict(X_scaled)[0]
    prediction_label = "High Volatility" if prediction == 1 else "Low Volatility"

    confidence = None
    proba_df = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        confidence = max(proba) * 100

        proba_df = pd.DataFrame({
            "Class": ["Low Volatility", "High Volatility"],
            "Probability": [proba[0] * 100, proba[1] * 100]
        })

    st.subheader("Executive Decision Summary")
    st.caption(
        "Each selected market observation is processed dynamically. "
        "The reliability score and next-day volatility forecast are recalculated for the selected input."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Next-Day Forecast", prediction_label)
    col2.metric("Model Confidence", f"{confidence:.1f}%" if confidence is not None else "N/A")
    col3.metric("Reliability Status", reliability_label)
    # col4.metric("Input Date", selected_date_str)

    gauge_col, decision_col = st.columns([1.2, 1])

    with gauge_col:
        st.plotly_chart(reliability_gauge(reliability_score), use_container_width=True)

    with decision_col:
        st.subheader("System Decision")
        st.markdown("### Decision Policy")
        
        policy_df = pd.DataFrame({
            "Reliability Range": ["0-49", "50-69", "70-100"],
            "System Action": [
                "Prediction Rejected",
                "Prediction With Warning",
                "Prediction Accepted"
            ]
        })

        st.info("""
        Decision Rules

        • Reliability < 50  → Prediction Rejected

        • Reliability 50-69 → Warning

        • Reliability ≥ 70 → Prediction Accepted
        """)

        st.dataframe(
            policy_df,
            use_container_width=True,
            hide_index=True
        )
        st.write(f"**Forecast:** {prediction_label}")
        st.write(f"**Reliability Score:** {reliability_score:.1f}/100")
        st.write(f"**Reliability Status:** {reliability_label}")
        st.write(f"**Decision Message:** {decision_message(reliability_score)}")

    if reliability_score >= 70:
        st.success(decision_message(reliability_score))
    elif reliability_score >= 50:
        st.warning(decision_message(reliability_score))
    else:
        st.error(decision_message(reliability_score))

    st.divider()

    st.subheader("System Workflow")
    st.markdown(
        """
        **Incoming Market Data → Feature Engineering → Reliability Assessment → Volatility Prediction → Decision Support**

        For each selected market observation, technical indicators are calculated, 
        the reliability of the input is evaluated using Isolation Forest, 
        and the trained Logistic Regression model generates the next-day volatility forecast.
        """
    )

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("NASDAQ Closing Price")

        price_data = raw_data.copy()
        price_data["Date"] = price_data.index.strftime("%Y-%m-%d")

        fig_price = go.Figure()
        fig_price.add_trace(
            go.Scatter(
                x=price_data["Date"],
                y=price_data["Close"],
                mode="lines",
                name="Close Price"
            )
        )

        if "Close" in selected_row.columns:
            fig_price.add_trace(
                go.Scatter(
                    x=[selected_date_str],
                    y=[float(selected_row["Close"].iloc[0])],
                    mode="markers",
                    marker=dict(color="red", size=12),
                    name="Selected Input"
                )
            )

        fig_price.update_layout(
            height=350,
            xaxis_title="Date",
            yaxis_title="Close Price",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=30, b=10)
        )

        st.plotly_chart(fig_price, use_container_width=True)

    with col_b:
        st.subheader("Reliability Score Over Time")
        reliability_data = model_data.copy()
        reliability_data["Date"] = reliability_data.index.strftime("%Y-%m-%d")

        fig_rel = go.Figure()
        fig_rel.add_trace(
            go.Scatter(
                x=reliability_data["Date"],
                y=reliability_data["reliability_score"],
                mode="lines",
                name="Reliability Score",
                fill="tozeroy"
            )
        )

        fig_rel.add_hline(
            y=70,
            line_dash="dash",
            line_color="orange",
            annotation_text="Reliability Threshold (70)"
        )

        fig_rel.add_trace(
            go.Scatter(
                x=[selected_date_str],
                y=[reliability_score],
                mode="markers",
                marker=dict(color="red", size=12),
                name="Selected Input"
            )
        )

        fig_rel.update_layout(
            height=350,
            xaxis_title="Date",
            yaxis_title="Reliability Score",
            yaxis=dict(range=[0, 105]),
            hovermode="x unified",
            margin=dict(l=10, r=10, t=30, b=10)
        )

        st.plotly_chart(fig_rel, use_container_width=True)

    st.divider()

    if proba_df is not None:
        st.subheader("Volatility Risk Probability")

        low_prob = float(proba_df.loc[proba_df["Class"] == "Low Volatility", "Probability"].iloc[0])
        high_prob = float(proba_df.loc[proba_df["Class"] == "High Volatility", "Probability"].iloc[0])

    st.caption(
        "The model estimates the probability of each next-day volatility scenario. "
        "The final forecast is selected based on the higher probability."
    )

    p1, p2, p3 = st.columns([1, 1, 1.2])

    with p1:
        st.metric("Low Volatility Probability", f"{low_prob:.1f}%")
        st.progress(min(low_prob / 100, 1.0))

    with p2:
        st.metric("High Volatility Probability", f"{high_prob:.1f}%")
        st.progress(min(high_prob / 100, 1.0))

    with p3:
        risk_value = high_prob

        fig_risk = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_value,
                number={"suffix": "%"},
                title={"text": "High Volatility Risk"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ef4444" if risk_value >= 50 else "#22c55e"},
                    "steps": [
                        {"range": [0, 35], "color": "#14532d"},
                        {"range": [35, 60], "color": "#854d0e"},
                        {"range": [60, 100], "color": "#7f1d1d"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 4},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            )
        )

        fig_risk.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig_risk, use_container_width=True)

    if high_prob >= 60:
        st.error(
            f"Risk Signal: High volatility risk is dominant with {high_prob:.1f}% probability."
        )
    elif high_prob >= 40:
        st.warning(
            f"Risk Signal: The model is uncertain. High volatility probability is {high_prob:.1f}%."
        )
    else:
        st.success(
            f"Risk Signal: Low volatility scenario is dominant. High volatility probability is only {high_prob:.1f}%."
        )

    st.divider()

    st.subheader("Prediction Drivers")

    driver_cols = [col for col in ["ATR", "RSI", "MACD"] if col in selected_row.columns]

    if driver_cols:
        d1, d2, d3 = st.columns(3)

        for box, col in zip([d1, d2, d3], driver_cols):
            box.metric(col, f"{float(selected_row[col].iloc[0]):.4f}")

        st.caption(
            "ATR represents volatility, RSI represents market momentum, and MACD represents trend and momentum behavior."
        )

    st.divider()

    st.subheader("Selected Input Feature Snapshot")

    selected_features = [
        col for col in ["ATR", "RSI", "MACD", "MA_5", "MA_20", "Volume", "Close"]
        if col in selected_row.columns
    ]

    feature_snapshot = selected_row[selected_features].T
    feature_snapshot.columns = ["Value"]

    st.dataframe(feature_snapshot, use_container_width=True)

    st.divider()

    st.subheader("Model Performance Summary")

    results_path = r"D:\fintech\data\volatility_prediction_results.csv"

    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
    else:
        st.info("Model performance file was not found.")

    st.divider()

    st.caption(
        "This dashboard is designed as a risk-oriented decision-support system. "
        "It does not provide direct investment or trading advice."
    )

except Exception as e:
    st.error("Dashboard could not be loaded.")
    st.exception(e)