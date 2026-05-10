"""
Custom CSS injection for Streamlit dashboard.
"""

DARK_COLORS = {
    "bg": "#0a0e1a",
    "surface": "#0f1629",
    "card": "#141b2d",
    "card2": "#1a2035",
    "border": "#1e3a5f",
    "text": "#e8eaf6",
    "subtext": "#8892b0",
    "accent": "#00d4aa",
    "accent2": "#4fc3f7",
    "accent3": "#ffd54f",
    "danger": "#ff6b6b",
    "warn": "#ffb347",
    "success": "#00d4aa",
}

LIGHT_COLORS = {
    "bg": "#eef2ff",
    "surface": "#ffffff",
    "card": "#f8faff",
    "card2": "#f0f4ff",
    "border": "#cbd5e0",
    "text": "#0d1b2a",
    "subtext": "#4a5568",
    "accent": "#0077aa",
    "accent2": "#0066cc",
    "accent3": "#c9930a",
    "danger": "#cc2244",
    "warn": "#c05200",
    "success": "#0e7c51",
}


def inject_css(dark_mode: bool) -> str:
    c = DARK_COLORS if dark_mode else LIGHT_COLORS

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {{
    font-family: 'Space Grotesk', sans-serif;
    background-color: {c['bg']} !important;
    color: {c['text']} !important;
}}

.stApp {{
    background-color: {c['bg']} !important;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background-color: {c['surface']} !important;
    border-right: 1px solid {c['border']} !important;
}}

section[data-testid="stSidebar"] * {{
    color: {c['text']} !important;
}}

/* ── Metric cards ── */
div[data-testid="metric-container"] {{
    background: {c['card']} !important;
    border: 1px solid {c['border']} !important;
    border-radius: 8px !important;
    padding: 16px !important;
}}

div[data-testid="metric-container"] label {{
    color: {c['subtext']} !important;
    font-size: 10px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-family: 'DM Mono', monospace !important;
}}

div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {c['text']} !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 24px !important;
    font-weight: 500 !important;
}}

div[data-testid="stMetricDelta"] {{
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {c['surface']} !important;
    border-bottom: 1px solid {c['border']} !important;
    gap: 0px;
}}

.stTabs [data-baseweb="tab"] {{
    background-color: transparent !important;
    color: {c['subtext']} !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 10px 20px !important;
    border-radius: 0 !important;
    font-family: 'DM Mono', monospace !important;
    border-bottom: 2px solid transparent !important;
}}

.stTabs [aria-selected="true"] {{
    color: {c['accent']} !important;
    border-bottom: 2px solid {c['accent']} !important;
}}

/* ── Selectbox / radio ── */
.stSelectbox > div > div,
.stRadio > div {{
    background: {c['card']} !important;
    border-color: {c['border']} !important;
    color: {c['text']} !important;
}}

/* ── Plotly chart containers ── */
.stPlotlyChart {{
    background: {c['card']} !important;
    border: 1px solid {c['border']} !important;
    border-radius: 8px !important;
    padding: 4px !important;
}}

/* ── Section headers ── */
.section-header {{
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {c['accent']};
    border-bottom: 1px solid {c['border']};
    padding-bottom: 6px;
    margin-bottom: 16px;
    margin-top: 8px;
}}

/* ── Insight cards ── */
.insight-card {{
    background: {c['card']};
    border: 1px solid {c['border']};
    border-left: 3px solid;
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 10px;
}}

.insight-card.success {{ border-left-color: {c['success']}; }}
.insight-card.warning {{ border-left-color: {c['warn']}; }}
.insight-card.error   {{ border-left-color: {c['danger']}; }}

.insight-title {{
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    font-weight: 500;
    color: {c['text']};
    margin-bottom: 4px;
}}

.insight-body {{
    font-size: 13px;
    color: {c['subtext']};
    line-height: 1.5;
}}

/* ── Risk badge ── */
.risk-badge {{
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.12em;
    padding: 4px 12px;
    border-radius: 3px;
    font-weight: 500;
}}

/* ── Stat pill ── */
.stat-pill {{
    background: {c['card2']};
    border: 1px solid {c['border']};
    border-radius: 6px;
    padding: 12px 16px;
    text-align: center;
    font-family: 'DM Mono', monospace;
}}

.stat-pill .value {{
    font-size: 20px;
    font-weight: 500;
    color: {c['accent']};
}}

.stat-pill .label {{
    font-size: 9px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {c['subtext']};
    margin-top: 2px;
}}

/* ── Divider ── */
hr {{
    border-color: {c['border']} !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: {c['bg']}; }}
::-webkit-scrollbar-thumb {{ background: {c['border']}; border-radius: 4px; }}

/* ── Top ticker bar ── */
.ticker-bar {{
    background: {c['surface']};
    border-bottom: 1px solid {c['border']};
    padding: 6px 20px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    display: flex;
    gap: 32px;
    color: {c['subtext']};
}}

.ticker-up   {{ color: {c['accent']}; }}
.ticker-down {{ color: {c['danger']}; }}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}}
</style>
"""
