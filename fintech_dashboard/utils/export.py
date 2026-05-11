"""
PDF report generation for the dashboard.
"""

import io
from datetime import datetime


def generate_pdf_report(overview_stats: dict, metrics: dict) -> bytes:
    """
    Generate a simple HTML-based PDF report.
    Returns bytes of a minimal PDF-like HTML that can be printed.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    risk = overview_stats["risk_level"]
    risk_colors = {"LOW": "#00d4aa", "MEDIUM": "#ffd54f", "HIGH": "#ff6b6b"}
    risk_color = risk_colors.get(risk, "#ccc")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<style>
  body {{ font-family: 'Courier New', monospace; background: #0a0e1a; color: #e8eaf6;
          margin: 0; padding: 32px; }}
  h1   {{ font-size: 20px; color: #00d4aa; border-bottom: 1px solid #1e3a5f; padding-bottom: 8px; }}
  h2   {{ font-size: 13px; color: #8892b0; letter-spacing: 0.14em; text-transform: uppercase; margin-top: 28px; }}
  .meta {{ font-size: 11px; color: #8892b0; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 12px; }}
  th    {{ text-align: left; color: #8892b0; font-size: 10px; letter-spacing: 0.1em;
           text-transform: uppercase; padding: 6px 12px; border-bottom: 1px solid #1e3a5f; }}
  td    {{ padding: 8px 12px; border-bottom: 1px solid #1a2035; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 3px;
             background: {risk_color}22; color: {risk_color};
             font-size: 11px; border: 1px solid {risk_color}; }}
</style>
</head>
<body>
<h1>VOLATILITY PREDICTION - ANALYST REPORT</h1>
<p class="meta">Generated: {now} &nbsp;|&nbsp; System: VOLPRED v2.1 &nbsp;|&nbsp;
Risk Level: <span class="badge">{risk}</span></p>

<h2>Overview Metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Current Realized Volatility</td><td>{overview_stats['current_vol']:.2%}</td></tr>
<tr><td>30-Day Vol Change</td><td>{overview_stats['vol_change_pct']:+.1f}%</td></tr>
<tr><td>Data Quality Score</td><td>{overview_stats['quality_score']:.1f} / 100</td></tr>
<tr><td>Total Records</td><td>{overview_stats['total_records']:,}</td></tr>
<tr><td>Average Missing Data</td><td>{overview_stats['missing_pct']:.2f}%</td></tr>
</table>

<h2>Model Performance</h2>
<table>
<tr><th>Model</th><th>Accuracy (%)</th><th>Precision (%)</th><th>Recall (%)</th><th>F1-Score</th></tr>
{''.join(
    f"<tr><td>{m}</td><td>{v['Accuracy']:.2f}</td><td>{v['Precision']:.2f}</td>"
    f"<td>{v['Recall']:.2f}</td><td>{v['F1']:.2f}</td></tr>"
    for m, v in metrics.items()
)}
</table>

<h2>Disclaimer</h2>
<p style="font-size:11px; color:#4a5568;">
This report is generated from backtested model outputs using mock financial data for demonstration.
It does not constitute financial advice. Past volatility performance does not guarantee future results.
</p>
</body>
</html>"""

    return html.encode("utf-8")
