import sys
import os
import random
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.data_quality import DataQuality


st.set_page_config(
    page_title="Financial Volatility Prediction: Data Reliability Scoring and Chronological Validation",
    layout="wide"
)

st.title("Financial Volatility Prediction: Data Reliability Scoring and Chronological Validation")
st.caption("Risk-Oriented Decision Support System with Data Reliability Scoring")

st.info(
    "This dashboard simulates incoming market data by selecting different historical input samples. "
    "For each selected input, the system shows the volatility forecast and the Data Reliability Score."
)


@st.cache_data
def load_processed_data():
    loader = DataLoader("data/nasdq.csv")
    raw_data = loader.load_csv()
    raw_data = loader.preprocess()

    engineer = FeatureEngineer(raw_data)
    features = engineer.apply_all_features()

    quality = DataQuality(features)
    quality.detect_anomalies()
    quality.calculate_reliability_score()

    quality_data = quality.get_data()
    return raw_data, quality_data


def add_lag_features(df, lags=[1, 2, 3]):
    lag_df = df.copy()
    features_to_lag = ["Close", "Volume", "MA_5", "MA_20", "RSI", "MACD"]

    for feature in features_to_lag:
        if feature in lag_df.columns:
            for lag in lags:
                lag_df[f"{feature}_lag{lag}"] = lag_df[feature].shift(lag)

    return lag_df.dropna()


@st.cache_resource
def load_model_files():
    model = joblib.load("models/volatility_best_model.pkl")
    scaler = joblib.load("models/volatility_scaler.pkl")
    return model, scaler


def reliability_status(score):
    if score >= 90:
        return "High"
    elif score >= 70:
        return "Moderate"
    elif score >= 50:
        return "Low"
    return "Critical"

def create_reliability_gauge(score):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100"},
            title={"text": "Data Reliability Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "white"},
                "steps": [
                    {"range": [0, 50], "color": "#7f1d1d"},
                    {"range": [50, 70], "color": "#9a3412"},
                    {"range": [70, 90], "color": "#854d0e"},
                    {"range": [90, 100], "color": "#14532d"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )

    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def choose_sample(model_data, scenario):
    if scenario == "Random Incoming Data":
        return model_data.sample(1)

    if scenario == "Reliable Incoming Data":
        subset = model_data[model_data["reliability_score"] >= 90]
        if subset.empty:
            subset = model_data.sort_values("reliability_score", ascending=False).head(50)
        return subset.sample(1)

    if scenario == "Moderate Incoming Data":
        subset = model_data[
            (model_data["reliability_score"] >= 70)
            & (model_data["reliability_score"] < 90)
        ]
        if subset.empty:
            subset = model_data.iloc[
                (model_data["reliability_score"] - 75).abs().argsort()
            ].head(50)
        return subset.sample(1)

    if scenario == "Unreliable Incoming Data":
        subset = model_data[model_data["reliability_score"] < 50]
        if subset.empty:
            subset = model_data.sort_values("reliability_score", ascending=True).head(50)
        return subset.sample(1)

    return model_data.iloc[-1:]


try:
    raw_data, quality_data = load_processed_data()
    model, scaler = load_model_files()
    model_data = add_lag_features(quality_data)

    with st.sidebar:
        st.header("Incoming Data Simulation")
        scenario = st.selectbox(
            "Select data type",
            [
                "Latest Market Data",
                "Random Incoming Data",
                "Reliable Incoming Data",
                "Moderate Incoming Data",
                "Unreliable Incoming Data"
            ]
        )

        if st.button("Generate Incoming Data"):
            st.session_state["random_seed"] = random.randint(1, 999999)

    if "random_seed" not in st.session_state:
        st.session_state["random_seed"] = 42

    random.seed(st.session_state["random_seed"])

    selected_row = choose_sample(model_data, scenario)
    selected_date = selected_row.index[0]
    reliability_score = float(selected_row["reliability_score"].iloc[0])
    status = reliability_status(reliability_score)

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

    if hasattr(model, "predict_proba"):
        confidence = max(model.predict_proba(X_scaled)[0]) * 100
    else:
        confidence = None

    st.subheader("Next-Day Volatility Forecast")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Forecast", prediction_label)

    with col2:
        st.metric("Model Confidence", f"{confidence:.1f}%" if confidence is not None else "N/A")

    with col3:
        st.plotly_chart(
            create_reliability_gauge(reliability_score),
            use_container_width=True
        )

        if reliability_score >= 90:
            st.success("High reliability")
        elif reliability_score >= 70:
            st.warning("Moderate reliability")
        elif reliability_score >= 50:
            st.warning("Low reliability")
        else:
            st.error("Critical reliability")

    with col4:
        st.metric("Input Date", selected_date.strftime("%Y-%m-%d"))

    if reliability_score >= 90:
        st.success("High reliability: the input data is strongly suitable for prediction.")
    elif reliability_score >= 70:
        st.warning("Moderate reliability: the prediction can be interpreted with caution.")
    elif reliability_score >= 50:
        st.warning("Low reliability: the input data may contain abnormal patterns.")
    else:
        st.error("Critical reliability: the prediction may not be reliable.")

    st.divider()

    st.subheader("Incoming Data Explanation")

    st.write(
        f"Selected scenario: **{scenario}**. "
        f"The system selected one input sample from the processed NASDAQ dataset, calculated its Data Reliability Score, "
        f"and produced a next-day volatility prediction. This demonstrates how the system reacts to different data-quality conditions."
    )

    st.divider()

    st.subheader("Market and Reliability Trends")

    col_price, col_reliability = st.columns(2)

    with col_price:
        st.write("**NASDAQ Closing Price**")

        price_plot = raw_data.tail(500)

        price_fig = go.Figure()
        price_fig.add_trace(
            go.Scatter(
                x=price_plot.index,
                y=price_plot["Close"],
                mode="lines",
                name="Close Price",
                line=dict(color="steelblue", width=2)
            )
        )

        price_fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Close Price",
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            hovermode="x unified"
        )

        st.plotly_chart(price_fig, use_container_width=True)

    with col_reliability:
        st.write("**Data Reliability Score Over Time**")

        reliability_plot = model_data.tail(500)

        reliability_fig = go.Figure()
        reliability_fig.add_trace(
            go.Scatter(
                x=reliability_plot.index,
                y=reliability_plot["reliability_score"],
                mode="lines",
                name="Reliability Score",
                line=dict(color="darkorange", width=2),
                fill="tozeroy",
                fillcolor="rgba(255, 165, 0, 0.2)"
            )
        )

        reliability_fig.add_hline(
            y=70,
            line_dash="dash",
            line_color="orange",
            annotation_text="Caution Threshold"
        )

        reliability_fig.add_hline(
            y=90,
            line_dash="dash",
            line_color="green",
            annotation_text="High Reliability"
        )

        reliability_fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Reliability Score",
            yaxis=dict(range=[0, 105]),
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            hovermode="x unified"
        )

        st.plotly_chart(reliability_fig, use_container_width=True)

    st.divider()

    st.subheader("Volatility Prediction History")

    history_data = model_data.tail(30)
    history_rows = []

    for idx in history_data.index:
        row = model_data.loc[idx:idx, feature_columns]
        X_hist_scaled = scaler.transform(row)
        pred_hist = model.predict(X_hist_scaled)[0]

        history_rows.append({
            "Date": idx.strftime("%Y-%m-%d"),
            "PredictionValue": 1 if pred_hist == 1 else 0,
            "PredictionLabel": "High" if pred_hist == 1 else "Low"
        })

    history_df = pd.DataFrame(history_rows)

    history_fig = go.Figure()
    history_fig.add_trace(
        go.Bar(
            x=history_df["Date"],
            y=history_df["PredictionValue"],
            text=history_df["PredictionLabel"],
            textposition="outside",
            name="Prediction"
        )
    )

    history_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Volatility Class",
        yaxis=dict(
            tickmode="array",
            tickvals=[0, 1],
            ticktext=["Low", "High"]
        ),
        height=300,
        margin=dict(l=0, r=0, t=20, b=0)
    )

    st.plotly_chart(history_fig, use_container_width=True)

    st.divider()

    st.subheader("Recent Data Samples")

    sample_table = model_data[
        ["Close", "Volume", "reliability_score", "is_anomaly"]
    ].tail(10).copy()

    sample_table["Reliability Status"] = sample_table["reliability_score"].apply(reliability_status)

    st.dataframe(
        sample_table.reset_index(),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Model Performance Summary")

    results_path = "data/volatility_prediction_results.csv"

    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
    else:
        st.info("Model results file not found. Run `python src/run_volatility_analysis.py` first.")

    st.divider()

    st.caption(
        "This dashboard is a risk-oriented decision-support interface. "
        "It does not provide direct trading recommendations."
    )

except Exception as e:
    st.error("Dashboard could not be loaded.")
    st.exception(e)