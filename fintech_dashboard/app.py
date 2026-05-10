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
from datetime import datetime

# ── Local modules ──────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

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


# ── Inject custom CSS ─────────────────────────────────────────────────────────
st.markdown(inject_css(st.session_state.dark_mode), unsafe_allow_html=True)


# ── Data (cached) ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_data(counter=0):
    price_df = generate_price_series(n=500)
    vol_df = generate_volatility_series(price_df)
    forecast_df = generate_future_forecast(n_days=30, last_vol=vol_df["realized_vol"].iloc[-1])
    quality_df = generate_data_quality()
    outlier_df = generate_outliers(price_df)
    metrics = generate_model_metrics()
    training_df = generate_training_curves()
    stats = get_overview_stats(vol_df, quality_df)
    return vol_df, forecast_df, quality_df, outlier_df, metrics, training_df, stats

vol_df, forecast_df, quality_df, outlier_df, metrics, training_df, stats = load_data(
    st.session_state.refresh_counter
)


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
    asset = st.selectbox(
        "", ["S&P 500 (SPX)", "NASDAQ (NDX)", "Bitcoin (BTC)", "Gold (XAUUSD)", "EUR/USD"],
        label_visibility="collapsed",
    )

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
        st.metric("LSTM RMSE", f"{stats['lstm_rmse']:.4f}", delta="-12.3% vs ARIMA")
    with c3:
        st.metric("LSTM R²", f"{stats['lstm_r2']:.3f}", delta="+0.025 vs GBM")
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
            bar_w = int(m["R2"] * 100)
            st.markdown(
                f"""<div class="stat-pill">
                <div style="font-size:13px;font-weight:600;
                    color:{m['color']};margin-bottom:6px">{model_name}</div>
                <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px">
                    <span style="color:#8892b0">RMSE</span>
                    <span style="font-family:DM Mono,monospace">{m['RMSE']:.4f}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px">
                    <span style="color:#8892b0">MAE</span>
                    <span style="font-family:DM Mono,monospace">{m['MAE']:.4f}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px">
                    <span style="color:#8892b0">R²</span>
                    <span style="font-family:DM Mono,monospace">{m['R2']:.3f}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:8px">
                    <span style="color:#8892b0">Dir. Acc.</span>
                    <span style="font-family:DM Mono,monospace">{m['Dir_Accuracy']:.1%}</span>
                </div>
                <div style="background:#1e3a5f;border-radius:2px;height:4px">
                    <div style="background:{m['color']};width:{bar_w}%;height:4px;border-radius:2px"></div>
                </div>
                <div style="text-align:right;font-size:9px;color:#8892b0;margin-top:2px">{m['R2']*100:.0f}% R²</div>
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

    st.markdown('<div class="section-header">Realized vs Predicted Volatility</div>', unsafe_allow_html=True)
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

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="section-header">LSTM Training History</div>', unsafe_allow_html=True)
        st.plotly_chart(
            training_curve_chart(training_df, st.session_state.dark_mode),
            key="training_curve_lstm",
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with col_right:
        st.markdown('<div class="section-header">LSTM Error Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(
            error_distribution_chart(vol_df, st.session_state.dark_mode),
            key="error_distribution_lstm",
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # Detailed metrics table
    st.markdown('<div class="section-header">Full Metrics Table</div>', unsafe_allow_html=True)
    rows = []
    for m, v in metrics.items():
        rows.append({
            "Model": m,
            "RMSE": f"{v['RMSE']:.4f}",
            "MAE": f"{v['MAE']:.4f}",
            "MAPE (%)": f"{v['MAPE']:.1f}",
            "R2 Score": f"{v['R2']:.4f}",
            "Directional Accuracy": f"{v['Dir_Accuracy']:.1%}",
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

    st.markdown('<div class="section-header">Missing Data Analysis</div>', unsafe_allow_html=True)
    st.plotly_chart(
        missing_data_chart(quality_df, st.session_state.dark_mode),
        key="missing_data_chart",
        use_container_width=True,
        config={"displayModeBar": False},
    )

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
            "The LSTM model captures this temporal dependency through its memory cells, "
            "which is why it outperforms ARIMA on this metric."
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
            "LSTM dominates on all metrics (RMSE, MAE, R2, directional accuracy) due to its ability "
            "to capture long-range temporal dependencies. ARIMA is a useful baseline for regime changes "
            "because it is more interpretable. GBM offers a strong price-volume feature-based alternative. "
            "For ensemble approaches, a weighted combination of LSTM (0.6) + GBM (0.4) tends to be "
            "most robust out of sample."
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
