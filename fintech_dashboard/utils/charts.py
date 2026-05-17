"""
Chart factory for the Financial Volatility Dashboard.
All charts use a consistent Bloomberg-inspired dark/light theme.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# ─── Theme helpers ────────────────────────────────────────────────────────────

def get_theme(dark_mode: bool) -> dict:
    if dark_mode:
        return {
            "bg": "#0a0e1a",
            "paper": "#0f1629",
            "card": "#141b2d",
            "text": "#e8eaf6",
            "subtext": "#8892b0",
            "grid": "#1e2a45",
            "border": "#1e3a5f",
            "accent": "#00d4aa",
            "accent2": "#4fc3f7",
            "accent3": "#ffd54f",
            "danger": "#ff6b6b",
            "plot_bg": "#0f1629",
        }
    else:
        return {
            "bg": "#f0f4ff",
            "paper": "#ffffff",
            "card": "#f8faff",
            "text": "#0d1b2a",
            "subtext": "#4a5568",
            "grid": "#e2e8f0",
            "border": "#cbd5e0",
            "accent": "#0077aa",
            "accent2": "#0066cc",
            "accent3": "#d4a017",
            "danger": "#cc2244",
            "plot_bg": "#ffffff",
        }


def base_layout(t: dict, title: str = "", height: int = 380) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=t["text"], size=14, family="DM Mono, monospace")),
        height=height,
        paper_bgcolor=t["paper"],
        plot_bgcolor=t["plot_bg"],
        font=dict(color=t["subtext"], size=11, family="DM Mono, monospace"),
        margin=dict(l=16, r=16, t=40 if title else 20, b=16),
        xaxis=dict(
            gridcolor=t["grid"],
            linecolor=t["grid"],
            tickfont=dict(color=t["subtext"], size=10),
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor=t["grid"],
            linecolor=t["grid"],
            tickfont=dict(color=t["subtext"], size=10),
            showgrid=True,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=t["grid"],
            font=dict(color=t["subtext"], size=10),
        ),
        hoverlabel=dict(
            bgcolor=t["card"],
            font=dict(color=t["text"], size=11, family="DM Mono, monospace"),
            bordercolor=t["border"],
        ),
    )


# ─── Charts ───────────────────────────────────────────────────────────────────

def candlestick_chart(df: pd.DataFrame, dark_mode: bool, period: str = "1Y") -> go.Figure:
    t = get_theme(dark_mode)

    period_map = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "1Y": 252, "ALL": len(df)}
    n = min(period_map.get(period, 252), len(df))
    dff = df.tail(n)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=dff["date"],
            open=dff["open"],
            high=dff["high"],
            low=dff["low"],
            close=dff["close"],
            increasing=dict(line=dict(color=t["accent"], width=1), fillcolor=t["accent"]),
            decreasing=dict(line=dict(color=t["danger"], width=1), fillcolor=t["danger"]),
            name="Price",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Volume bars
    colors = [t["accent"] if c >= o else t["danger"]
              for c, o in zip(dff["close"], dff["open"])]
    fig.add_trace(
        go.Bar(
            x=dff["date"],
            y=dff["volume"],
            marker_color=colors,
            marker_opacity=0.6,
            name="Volume",
            showlegend=False,
        ),
        row=2, col=1,
    )

    layout = base_layout(t, height=440)
    layout.update(
        xaxis2=dict(
            gridcolor=t["grid"], linecolor=t["grid"],
            tickfont=dict(color=t["subtext"], size=10),
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(gridcolor=t["grid"], tickprefix="$",
                   tickfont=dict(color=t["subtext"], size=10)),
        yaxis2=dict(gridcolor=t["grid"], tickfont=dict(color=t["subtext"], size=10)),
    )
    fig.update_layout(**layout)
    return fig


def volatility_comparison_chart(df: pd.DataFrame, dark_mode: bool, period: str = "1Y") -> go.Figure:
    t = get_theme(dark_mode)
    period_map = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "1Y": 252, "ALL": len(df)}
    n = min(period_map.get(period, 252), len(df))
    dff = df.tail(n)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dff["date"], y=dff["realized_vol"],
        name="Realized Volatility",
        line=dict(color=t["text"], width=2),
        fill="tozeroy",
        fillcolor=f"rgba({'40,44,70' if dark_mode else '200,210,240'},0.15)",
    ))
    fig.add_trace(go.Scatter(
        x=dff["date"], y=dff["lstm_pred"],
        name="LSTM Prediction",
        line=dict(color=t["accent"], width=1.5, dash="solid"),
    ))
    fig.add_trace(go.Scatter(
        x=dff["date"], y=dff["arima_pred"],
        name="ARIMA Prediction",
        line=dict(color=t["accent2"], width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=dff["date"], y=dff["gbm_pred"],
        name="GBM Prediction",
        line=dict(color=t["accent3"], width=1.5, dash="dash"),
    ))

    layout = base_layout(t, height=360)
    layout["yaxis"].update(tickformat=".1%")
    fig.update_layout(**layout)
    return fig


def forecast_chart(forecast_df: pd.DataFrame, hist_df: pd.DataFrame, dark_mode: bool, scenario: str = "Medium") -> go.Figure:
    t = get_theme(dark_mode)

    scenario_mult = {"Low": 0.7, "Medium": 1.0, "High": 1.5}
    mult = scenario_mult.get(scenario, 1.0)

    hist_tail = hist_df[["date", "realized_vol"]].tail(30)

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=hist_tail["date"], y=hist_tail["realized_vol"],
        name="Historical",
        line=dict(color=t["subtext"], width=2),
    ))

    # 95% CI
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["date"], forecast_df["date"].iloc[::-1]]),
        y=pd.concat([forecast_df["upper_95"] * mult, (forecast_df["lower_95"] * mult).iloc[::-1]]),
        fill="toself",
        fillcolor=f"rgba({'0,212,170' if dark_mode else '0,100,180'},0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI",
        showlegend=True,
    ))
    # 80% CI
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["date"], forecast_df["date"].iloc[::-1]]),
        y=pd.concat([forecast_df["upper_80"] * mult, (forecast_df["lower_80"] * mult).iloc[::-1]]),
        fill="toself",
        fillcolor=f"rgba({'0,212,170' if dark_mode else '0,100,180'},0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% CI",
        showlegend=True,
    ))
    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["forecast"] * mult,
        name="Forecast",
        line=dict(color=t["accent"], width=2.5),
    ))

    layout = base_layout(t, height=360)
    layout["yaxis"].update(tickformat=".1%")
    fig.update_layout(**layout)
    return fig


def model_comparison_bar(metrics: dict, dark_mode: bool) -> go.Figure:
    t = get_theme(dark_mode)
    models = list(metrics.keys())
    accuracy = [metrics[m]["Accuracy"] for m in models]
    precision = [metrics[m]["Precision"] for m in models]
    recall = [metrics[m]["Recall"] for m in models]
    colors = [metrics[m]["color"] for m in models]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Accuracy (%)", "Precision (%)", "Recall (%)"],
    )

    for i, vals in enumerate([accuracy, precision, recall], 1):
        fig.add_trace(go.Bar(
            x=models, y=vals,
            marker_color=colors,
            marker_line_width=0,
            showlegend=False,
            text=[f"{v:.2f}%" for v in vals],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(color=t["text"], size=10),
        ), row=1, col=i)

    layout = base_layout(t, height=300)
    layout["showlegend"] = False
    for key in ["xaxis", "xaxis2", "xaxis3"]:
        layout[key] = dict(gridcolor=t["grid"], tickfont=dict(color=t["subtext"], size=11))
    for key in ["yaxis", "yaxis2", "yaxis3"]:
        layout[key] = dict(
            gridcolor=t["grid"],
            tickfont=dict(color=t["subtext"], size=10),
            showgrid=True,
            range=[0, 108],
            ticksuffix="%",
        )
    layout["margin"] = dict(t=90, b=40, l=40, r=20)
    fig.update_layout(**layout)
    return fig


def training_curve_chart(df: pd.DataFrame, dark_mode: bool) -> go.Figure:
    t = get_theme(dark_mode)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["epoch"], y=df["lstm_train_loss"],
        name="Train Loss",
        line=dict(color=t["accent"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df["epoch"], y=df["lstm_val_loss"],
        name="Val Loss",
        line=dict(color=t["danger"], width=2, dash="dash"),
    ))

    layout = base_layout(t, height=300)
    layout["xaxis"].update(title="Epoch")
    layout["yaxis"].update(title="Loss (MSE)")
    fig.update_layout(**layout)
    return fig


def error_distribution_chart(vol_df: pd.DataFrame, dark_mode: bool) -> go.Figure:
    t = get_theme(dark_mode)
    errors = vol_df["realized_vol"] - vol_df["lstm_pred"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=40,
        marker_color=t["accent"],
        marker_opacity=0.8,
        name="LSTM Errors",
    ))
    fig.add_vline(x=0, line_color=t["danger"], line_dash="dash", line_width=1.5)

    layout = base_layout(t, height=300)
    layout["xaxis"].update(title="Prediction Error")
    layout["yaxis"].update(title="Frequency")
    fig.update_layout(**layout)
    return fig


def data_quality_timeline(quality_df: pd.DataFrame, dark_mode: bool) -> go.Figure:
    t = get_theme(dark_mode)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=quality_df["date"], y=quality_df["completeness"] * 100,
        name="Completeness", line=dict(color=t["accent"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=quality_df["date"], y=quality_df["consistency"] * 100,
        name="Consistency", line=dict(color=t["accent2"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=quality_df["date"], y=quality_df["timeliness"] * 100,
        name="Timeliness", line=dict(color=t["accent3"], width=2),
    ))

    # Danger zone
    fig.add_hrect(y0=0, y1=90, fillcolor=t["danger"], opacity=0.04, line_width=0)

    layout = base_layout(t, height=320)
    layout["yaxis"].update(range=[70, 101], title="Score (%)")
    fig.update_layout(**layout)
    return fig


def missing_data_chart(quality_df: pd.DataFrame, dark_mode: bool) -> go.Figure:
    t = get_theme(dark_mode)
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=quality_df["date"].tail(60),
        y=quality_df["missing_pct"].tail(60),
        marker_color=[t["danger"] if v > 5 else t["accent"] for v in quality_df["missing_pct"].tail(60)],
        name="Missing %",
    ))

    layout = base_layout(t, height=280)
    layout["yaxis"].update(title="Missing Data (%)")
    fig.update_layout(**layout)
    return fig


def outlier_scatter(outlier_df: pd.DataFrame, dark_mode: bool) -> go.Figure:
    t = get_theme(dark_mode)

    normal = outlier_df[~outlier_df["is_outlier"]]
    anomaly = outlier_df[outlier_df["is_outlier"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal["date"], y=normal["z_score"],
        mode="markers",
        marker=dict(color=t["accent"], size=3, opacity=0.5),
        name="Normal",
    ))
    fig.add_trace(go.Scatter(
        x=anomaly["date"], y=anomaly["z_score"],
        mode="markers",
        marker=dict(color=t["danger"], size=7, symbol="diamond"),
        name="Outlier",
    ))
    fig.add_hline(y=2.5, line_color=t["danger"], line_dash="dash", line_width=1)
    fig.add_hline(y=-2.5, line_color=t["danger"], line_dash="dash", line_width=1)

    layout = base_layout(t, height=300)
    layout["yaxis"].update(title="Z-Score")
    fig.update_layout(**layout)
    return fig


def quality_radar(quality_df: pd.DataFrame, dark_mode: bool) -> go.Figure:
    t = get_theme(dark_mode)
    latest = quality_df.iloc[-1]
    categories = ["Completeness", "Consistency", "Timeliness", "Completeness"]
    values = [
        latest["completeness"] * 100,
        latest["consistency"] * 100,
        latest["timeliness"] * 100,
        latest["completeness"] * 100,
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor=f"rgba({'0,212,170' if dark_mode else '0,119,170'},0.2)",
        line=dict(color=t["accent"], width=2),
        name="Quality",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[70, 100], color=t["subtext"], gridcolor=t["grid"]),
            angularaxis=dict(color=t["subtext"]),
            bgcolor=t["plot_bg"],
        ),
        paper_bgcolor=t["paper"],
        font=dict(color=t["subtext"], size=11),
        margin=dict(l=30, r=30, t=30, b=30),
        height=280,
        showlegend=False,
    )
    return fig
