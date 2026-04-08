"""
Precision Sniper v7.5 — Walk-Forward Analysis Dashboard
Streamlit + Plotly + yfinance
"""

import os
import re
import time
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

from strategy import StrategyParams, run_backtest, compute_stats
from walk_forward import run_walk_forward, PARAM_GRID
from strategy_registry import list_strategies, get_strategy, STRATEGY_REGISTRY
from custom_loader import (
    STRATEGY_TEMPLATE, compile_custom_strategy,
    save_strategy_to_disk, list_saved_strategies, load_saved_strategy,
)
from bt_bw_atr_strategy import run_bt_backtest, run_bt_generic, run_bt_generic_fast, resample_ohlcv
from param_sync import save_params, load_params, has_saved_params, list_saved_params


# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Precision Trader",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Force dark theme + Custom CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Global: deep charcoal base with warm tones ── */
    .stApp, .main, [data-testid="stAppViewContainer"] {
        background-color: #1a1a2e !important;
        color: #e0dfe4 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    [data-testid="stHeader"] {
        background: linear-gradient(180deg, #1a1a2e 0%, transparent 100%) !important;
    }

    /* ── Sidebar: rich dark plum ── */
    [data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #16162a 0%, #1e1b30 100%) !important;
        border-right: 1px solid rgba(242, 196, 206, 0.1) !important;
    }
    button[data-testid="baseButton-header"] {
        background-color: #e14658 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
    }
    button[data-testid="baseButton-header"] svg {
        stroke: #fff !important;
    }

    /* ── Inputs ── */
    .stTextInput > div > div, .stNumberInput > div > div,
    .stTextArea > div > div, [data-testid="stFileUploader"] {
        background-color: #22213a !important;
        border: 1px solid #3a3556 !important;
        color: #e0dfe4 !important;
        border-radius: 8px !important;
    }

    /* ── Dropdowns ── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background-color: #22213a !important;
        border: 1px solid #3a3556 !important;
        color: #e0dfe4 !important;
        border-radius: 8px !important;
    }
    [data-baseweb="popover"], [data-baseweb="menu"],
    [data-baseweb="select"] [role="listbox"],
    ul[role="listbox"] {
        background-color: #2a2945 !important;
        border: 1px solid #3a3556 !important;
        border-radius: 8px !important;
    }
    [data-baseweb="menu"] li, [role="option"] {
        background-color: #2a2945 !important;
        color: #e0dfe4 !important;
        border-bottom: 1px solid rgba(255,255,255,0.04) !important;
    }
    [data-baseweb="menu"] li:hover, [role="option"]:hover {
        background-color: #3d3960 !important;
        color: #fff !important;
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        background-color: #201f35 !important;
        border: 1px solid #3a3556 !important;
        border-radius: 10px;
    }

    /* ── Tabs: coral accent ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        border-bottom: 2px solid #2a2945;
        gap: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #7a7793 !important;
        padding: 10px 20px;
        font-weight: 500;
        transition: color 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #c0bdd6 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #f58f7c !important;
        border-bottom: 3px solid #f58f7c !important;
        font-weight: 600;
    }

    /* ── Metric cards: glass-morphism style ── */
    .metric-card {
        background: linear-gradient(135deg, #22213a 0%, #2a2945 100%);
        border: 1px solid rgba(242, 196, 206, 0.12);
        border-radius: 14px;
        padding: 18px 22px;
        text-align: center;
        margin: 4px;
        backdrop-filter: blur(10px);
        transition: border-color 0.3s, transform 0.2s;
    }
    .metric-card:hover {
        border-color: rgba(245, 143, 124, 0.3);
        transform: translateY(-1px);
    }
    .metric-card .label {
        color: #8a86a6;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .metric-card .value {
        color: #f0eff4;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 6px;
    }
    .metric-card .value.green { color: #8df688; }
    .metric-card .value.red { color: #f6506a; }
    .metric-card .value.yellow { color: #f5c87c; }

    /* ── Section headers: soft coral ── */
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f2c4ce;
        border-bottom: 1px solid rgba(245, 143, 124, 0.2);
        padding-bottom: 8px;
        margin: 24px 0 14px 0;
        letter-spacing: 0.02em;
        background: linear-gradient(90deg, rgba(245, 143, 124, 0.06) 0%, transparent 100%);
        padding: 10px 12px 8px 12px;
        border-radius: 6px 6px 0 0;
    }

    /* ── Tables ── */
    .fold-table th { background: #22213a !important; }
    [data-testid="stDataFrame"] { background-color: #1e1d33 !important; border-radius: 10px; }
    [data-testid="stDataFrame"] th { background-color: #2a2945 !important; }

    /* ── Progress bar: coral gradient ── */
    div[data-testid="stProgress"] > div { height: 10px; border-radius: 5px; }
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #e14658 0%, #f58f7c 100%) !important;
        border-radius: 5px;
    }

    /* ── Primary buttons: coral gradient ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #e14658 0%, #f58f7c 100%) !important;
        border: none !important;
        color: #fff !important;
        font-weight: 600;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        letter-spacing: 0.02em;
        transition: opacity 0.2s, transform 0.2s;
    }
    .stButton > button[kind="primary"]:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
    .stButton > button[kind="secondary"],
    .stButton > button:not([kind="primary"]) {
        background-color: #2a2945 !important;
        border: 1px solid #3a3556 !important;
        color: #c0bdd6 !important;
        border-radius: 10px !important;
        transition: background-color 0.2s;
    }
    .stButton > button:not([kind="primary"]):hover {
        background-color: #3d3960 !important;
    }

    /* ── Download buttons ── */
    .stDownloadButton > button {
        background-color: #2a2945 !important;
        border: 1px solid #3a3556 !important;
        color: #c0bdd6 !important;
        border-radius: 10px !important;
    }

    /* ── Checkboxes ── */
    [data-testid="stCheckbox"] label span { color: #e0dfe4 !important; }

    /* ── Alerts ── */
    [data-testid="stAlert"] {
        background-color: #22213a !important;
        border-color: #3a3556 !important;
        border-radius: 10px !important;
    }

    /* ── Markdown ── */
    .stMarkdown, .stMarkdown p, .stMarkdown li { color: #d6d4e0 !important; }
    .stMarkdown h1 { color: #f0eff4 !important; font-weight: 700; letter-spacing: -0.02em; }
    .stMarkdown h2 { color: #f2c4ce !important; font-weight: 600; }
    .stMarkdown h3 { color: #f58f7c !important; font-weight: 600; }
    .stMarkdown h4 { color: #8df688 !important; font-weight: 600; }

    /* ── Expander headers: coral accent ── */
    [data-testid="stExpander"] summary span {
        color: #f2c4ce !important;
        font-weight: 600 !important;
    }

    /* ── Selectbox / radio labels ── */
    .stSelectbox label, .stRadio label, .stNumberInput label,
    .stTextInput label, .stSlider label, .stCheckbox label,
    .stMultiSelect label, .stDateInput label, .stFileUploader label {
        color: #c0bdd6 !important;
        font-weight: 500 !important;
    }

    /* ── Tab active: brighter coral ── */
    .stTabs [aria-selected="true"] {
        color: #f58f7c !important;
        border-bottom: 3px solid #f58f7c !important;
        font-weight: 700 !important;
    }

    /* ── Success box: mint green tint ── */
    [data-testid="stAlert"][data-baseweb*="positive"],
    .element-container .stSuccess {
        border-left: 4px solid #8df688 !important;
    }
    .stMarkdown code { background-color: #2a2945 !important; color: #f2c4ce !important; border-radius: 4px; }
    .stMarkdown pre { background-color: #1e1d33 !important; border: 1px solid #3a3556 !important; border-radius: 8px !important; }

    /* ── Captions ── */
    .stCaption, small { color: #7a7793 !important; }

    /* ── Radio buttons ── */
    [data-testid="stRadio"] label { color: #d6d4e0 !important; }

    /* ── Sliders: coral thumb, plum track ── */
    [data-testid="stSlider"] [role="slider"] {
        background-color: #f58f7c !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child {
        background-color: #2a2945 !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {
        background-color: #562f54 !important;
    }

    /* ── Metrics (built-in st.metric) ── */
    [data-testid="stMetric"] {
        background-color: #22213a;
        border: 1px solid #3a3556;
        border-radius: 12px;
        padding: 12px 16px;
    }
    [data-testid="stMetric"] label { color: #8a86a6 !important; }
    [data-testid="stMetricValue"] { color: #f0eff4 !important; }

    /* ── Success/info boxes ── */
    .stSuccess { background-color: rgba(141, 246, 136, 0.08) !important; border-color: rgba(141, 246, 136, 0.2) !important; }
    .stInfo { background-color: rgba(245, 143, 124, 0.08) !important; border-color: rgba(245, 143, 124, 0.2) !important; }

    /* ── Dividers ── */
    hr { border-color: #2a2945 !important; opacity: 0.5; }

    /* ── Scrollbars ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #1a1a2e; }
    ::-webkit-scrollbar-thumb { background: #3a3556; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #4d4870; }

    /* ── Strategy color badges ── */
    .strat-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 4px 0 8px 0;
    }
    .strat-badge .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }
    .strat-legend {
        font-size: 0.72rem;
        line-height: 1.8;
        margin-top: 6px;
    }
    .strat-legend .legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 1px 0;
    }
    .strat-legend .legend-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }
    .strat-legend .legend-group {
        font-weight: 600;
        font-size: 0.74rem;
        margin-top: 6px;
        margin-bottom: 2px;
        letter-spacing: 0.04em;
    }
</style>
""", unsafe_allow_html=True)

# (JS injection removed — Streamlit sandboxes iframes, so using emoji prefixes instead)


# ──────────────────────────────────────────────────────────────────────────────
# Strategy Color Map — uses the app's palette for at-a-glance identification
# MNQ = greens (mint family from #8df688), MYM = coral/pink family (#f58f7c),
# 1H MNQ = yellow/amber family (#f5c87c), Other = purple/muted family (#c0bdd6)
# ──────────────────────────────────────────────────────────────────────────────

STRATEGY_COLORS = {
    # ── MNQ 1-min strategies: GREEN shades ──
    "combined_all_1m":  "#8df688",
    "orb_ny_1m":        "#6cd968",
    "orb_tokyo_1m":     "#52c94e",
    "orb_london_1m":    "#3cb938",
    "rbr_ny_1m":        "#5ee89a",
    "rbr_tokyo_1m":     "#44d88a",
    "rbr_london_1m":    "#2cc87a",
    # ── MYM 1-min strategies: CORAL / PINK shades ──
    "mym_combined_all_1m":  "#f58f7c",
    "mym_orb_ny_1m":        "#f2a894",
    "mym_orb_tokyo_1m":     "#efc1ac",
    "mym_orb_london_1m":    "#f2c4ce",
    "mym_rbr_ny_1m":        "#e14658",
    "mym_rbr_tokyo_1m":     "#c73e4f",
    "mym_rbr_london_1m":    "#ad3646",
    # ── 1H MNQ strategies: YELLOW / AMBER shades ──
    "orb_1h":              "#f5c87c",
    "tokyo_orb_1h":        "#e8b86e",
    "london_orb_1h":       "#dba860",
    "ny_rbr_1h":           "#f5d99a",
    "tokyo_rbr_1h":        "#e8c88a",
    "london_rbr_1h":       "#dbb87a",
    "precision_sniper_1h": "#f5e6b8",
    # ── Other strategies: PURPLE / LAVENDER shades ──
    "precision_sniper":    "#c0bdd6",
    "ema_crossover":       "#b0add0",
    "rsi_reversion":       "#a09dca",
    "macd_supertrend":     "#908dc4",
    "butterworth_atr":     "#807dbe",
    "butterworth_atr_1h":  "#706db8",
    "bw_atr_optimized":    "#605db2",
}

# Emoji prefix per strategy for dropdown visibility (colored circles)
# MNQ 1-min = green, MYM 1-min = red/orange, 1H = yellow, Other = purple
STRATEGY_EMOJI = {
    # MNQ 1-min — green circles with sub-type markers
    "combined_all_1m":     "\U0001F7E2",          # green circle
    "orb_ny_1m":           "\U0001F7E9",          # green square
    "orb_tokyo_1m":        "\U0001F7E9",          # green square
    "orb_london_1m":       "\U0001F7E9",          # green square
    "rbr_ny_1m":           "\U00002705",          # green check
    "rbr_tokyo_1m":        "\U00002705",          # green check
    "rbr_london_1m":       "\U00002705",          # green check
    # MYM 1-min — orange/red circles with sub-type markers
    "mym_combined_all_1m": "\U0001F7E0",          # orange circle
    "mym_orb_ny_1m":       "\U0001F7E7",          # orange square
    "mym_orb_tokyo_1m":    "\U0001F7E7",          # orange square
    "mym_orb_london_1m":   "\U0001F7E7",          # orange square
    "mym_rbr_ny_1m":       "\U0001F534",          # red circle
    "mym_rbr_tokyo_1m":    "\U0001F534",          # red circle
    "mym_rbr_london_1m":   "\U0001F534",          # red circle
    # 1H MNQ — yellow
    "orb_1h":              "\U0001F7E1",          # yellow circle
    "tokyo_orb_1h":        "\U0001F7E1",          # yellow circle
    "london_orb_1h":       "\U0001F7E1",          # yellow circle
    "ny_rbr_1h":           "\U0001F7E8",          # yellow square
    "tokyo_rbr_1h":        "\U0001F7E8",          # yellow square
    "london_rbr_1h":       "\U0001F7E8",          # yellow square
    "precision_sniper_1h": "\U0001F7E1",          # yellow circle
    # Other — purple
    "precision_sniper":    "\U0001F7E3",          # purple circle
    "ema_crossover":       "\U0001F7E3",          # purple circle
    "rsi_reversion":       "\U0001F7E3",          # purple circle
    "macd_supertrend":     "\U0001F7E3",          # purple circle
    "butterworth_atr":     "\U0001F7E3",          # purple circle
    "butterworth_atr_1h":  "\U0001F7E3",          # purple circle
    "bw_atr_optimized":    "\U0001F7E3",          # purple circle
}

# Session names to emphasize with Unicode bold text in dropdowns
_BOLD_MAP = str.maketrans({
    'N': '\U0001D411', 'Y': '\U0001D418',  # NY
    'T': '\U0001D413', 'o': '\U0001D428', 'k': '\U0001D424', 'y': '\U0001D432',  # Tokyo
    'L': '\U0001D40B', 'd': '\U0001D41D', 'n': '\U0001D427',  # London
})

def _bold_session(name: str) -> str:
    """Replace session names (NY, Tokyo, London) with Unicode bold equivalents."""
    import re
    def _bold_word(m):
        return ''.join(
            chr(ord(c) - ord('A') + 0x1D400) if 'A' <= c <= 'Z'
            else chr(ord(c) - ord('a') + 0x1D41A) if 'a' <= c <= 'z'
            else c
            for c in m.group(0)
        )
    return re.sub(r'\b(NY|Tokyo|London)\b', _bold_word, name)


def _format_strategy_name(key: str, name: str) -> str:
    """Format a strategy name with emoji prefix and bold sessions for dropdown display."""
    emoji = STRATEGY_EMOJI.get(key, "\u26AA")   # default: white circle
    label = _bold_session(name)
    return f"{emoji}  {label}"


# Color group labels for the legend
STRATEGY_COLOR_GROUPS = {
    "MNQ 1-min":  "#8df688",
    "MYM 1-min":  "#f58f7c",
    "MNQ 1-hour": "#f5c87c",
    "Other":      "#c0bdd6",
}


def _strat_color(key: str) -> str:
    """Get the color for a strategy key, with fallback."""
    return STRATEGY_COLORS.get(key, "#c0bdd6")


def _render_strat_badge(key: str, name: str):
    """Render a colored badge showing the strategy's color-coded identity."""
    color = _strat_color(key)
    st.markdown(
        f'<div class="strat-badge" style="background: {color}15; border: 1px solid {color}40;">'
        f'<span class="dot" style="background: {color};"></span>'
        f'<span style="color: {color};">{name}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_strat_legend():
    """Render a compact color legend for all strategy groups."""
    html = '<div class="strat-legend">'
    for group_name, group_color in STRATEGY_COLOR_GROUPS.items():
        html += f'<div class="legend-group" style="color: {group_color};">{group_name}</div>'
        for key, color in STRATEGY_COLORS.items():
            # Determine which group this strategy belongs to
            in_group = False
            if group_name == "MNQ 1-min" and key in (
                "combined_all_1m", "orb_ny_1m", "orb_tokyo_1m", "orb_london_1m",
                "rbr_ny_1m", "rbr_tokyo_1m", "rbr_london_1m",
            ):
                in_group = True
            elif group_name == "MYM 1-min" and key.startswith("mym_"):
                in_group = True
            elif group_name == "MNQ 1-hour" and key in (
                "orb_1h", "tokyo_orb_1h", "london_orb_1h", "ny_rbr_1h",
                "tokyo_rbr_1h", "london_rbr_1h", "precision_sniper_1h",
            ):
                in_group = True
            elif group_name == "Other" and key in (
                "precision_sniper", "ema_crossover", "rsi_reversion",
                "macd_supertrend", "butterworth_atr", "butterworth_atr_1h", "bw_atr_optimized",
            ):
                in_group = True
            if in_group:
                # Short display name
                short = key.replace("mym_", "").replace("_1m", "").replace("_1h", "").replace("_", " ").upper()
                html += (
                    f'<div class="legend-item">'
                    f'<span class="legend-dot" style="background: {color};"></span>'
                    f'<span style="color: {color};">{short}</span>'
                    f'</div>'
                )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Built-in Dataset Registry — add new futures here as you acquire data
# ──────────────────────────────────────────────────────────────────────────────

BUILTIN_DATASETS = {
    "mnq_1m": {
        "label": "MNQ 1-Min (built-in, 6+ years)",
        "file": "MNQ_1min_continuous.csv",
        "caption": "MNQ continuous front-month, 1-min OHLCV bars, Jan 2020 - Mar 2026 (2.2M bars)",
        "loader": "load_builtin_mnq",
    },
    "mym_1m": {
        "label": "MYM 1-Min (built-in, 4+ years)",
        "file": "MYM_1min_continuous.csv",
        "caption": "MYM continuous front-month, 1-min OHLCV bars, Jan 2022 - Apr 2026 (1.5M bars)",
        "loader": "load_builtin_mym",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten columns, ensure lowercase OHLCV, localize to US/Eastern."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    rename_map = {"adj close": "close", "adj_close": "close"}
    df = df.rename(columns=rename_map)

    # Add synthetic volume if missing (strategy still works, volume filters become neutral)
    if "volume" not in df.columns:
        df["volume"] = 1000  # constant placeholder

    required = {"open", "high", "low", "close", "volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("US/Eastern")
    return df.dropna(subset=["open", "high", "low", "close"])


def load_from_csv(uploaded_file) -> pd.DataFrame:
    """
    Parse a user-uploaded CSV/XLSX of 15m OHLCV data.
    Expected columns (case-insensitive): datetime/date/time, open, high, low, close, volume.
    """
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        raw = pd.read_excel(uploaded_file)
    else:
        raw = pd.read_csv(uploaded_file)

    raw.columns = [str(c).lower().strip() for c in raw.columns]

    dt_col = None
    for candidate in ["datetime", "date", "time", "timestamp", "bar_time"]:
        if candidate in raw.columns:
            dt_col = candidate
            break
    if dt_col is None:
        dt_col = raw.columns[0]

    # Detect Unix timestamps (all numeric, large values)
    col_vals = pd.to_numeric(raw[dt_col], errors="coerce")
    if col_vals.notna().all() and (col_vals > 1e9).all():
        raw[dt_col] = pd.to_datetime(col_vals, unit="s", utc=True)
    else:
        raw[dt_col] = pd.to_datetime(raw[dt_col], infer_datetime_format=True)

    raw = raw.set_index(dt_col).sort_index()
    return _normalize_df(raw)


@st.cache_data(show_spinner=False, ttl=1800)
def load_from_yfinance(ticker: str, interval: str = "15m") -> pd.DataFrame:
    """
    Fetch data from yfinance at the requested interval.
    Limits per interval:
        15m  → last 58 days
        1h   → last 730 days (~2 years)
        1d   → last 10 years
    """
    period_map = {
        "15m": "58d",
        "1h":  "730d",
        "1d":  "10y",
    }
    period = period_map.get(interval, "58d")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
    if raw.empty:
        return pd.DataFrame()
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    return _normalize_df(raw)


def _load_builtin_csv(name: str) -> pd.DataFrame:
    """Load a built-in CSV, preferring .csv.gz if available."""
    base = os.path.join(os.path.dirname(__file__), "data")
    gz_path = os.path.join(base, name + ".gz")
    csv_path = os.path.join(base, name)
    if os.path.exists(gz_path):
        df = pd.read_csv(gz_path, index_col=0, parse_dates=True, compression="gzip")
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        return pd.DataFrame()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return _normalize_df(df)


@st.cache_data(show_spinner=False)
def load_builtin_mnq():
    return _load_builtin_csv("MNQ_1min_continuous.csv")


@st.cache_data(show_spinner=False)
def load_builtin_mym():
    return _load_builtin_csv("MYM_1min_continuous.csv")


def load_data_from_source(data_source, uploaded_file, ticker, timeframe):
    """Unified data loader that handles CSV, yfinance, and built-in MNQ/MYM sources."""
    if data_source == "Upload CSV/XLSX":
        if uploaded_file is None:
            return pd.DataFrame(), "Upload a data file in the sidebar first."
        try:
            return load_from_csv(uploaded_file), ""
        except Exception as e:
            return pd.DataFrame(), f"Parse error: {e}"
    elif "MNQ 1-Min" in data_source:
        df = load_builtin_mnq()
        if df.empty:
            return df, "Built-in MNQ data file not found."
        return df, ""
    elif "MYM 1-Min" in data_source:
        df = load_builtin_mym()
        if df.empty:
            return df, "Built-in MYM data file not found."
        return df, ""
    else:
        df = load_from_yfinance(ticker, interval=timeframe)
        return df, "" if not df.empty else "No data from yfinance."


def metric_card(label: str, value: str, color: str = "") -> str:
    cls = f"value {color}" if color else "value"
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="{cls}">{value}</div>
    </div>"""


def metric_card_dd(dd_pct: float, capital: float) -> str:
    """Special metric card for Max Drawdown showing dollars + percentage."""
    dd_dollars = abs(dd_pct) * capital
    return f"""
    <div class="metric-card">
        <div class="label">MAX DD</div>
        <div class="value red">${dd_dollars:,.0f}</div>
        <div style="color:#8a86a6;font-size:0.75rem;margin-top:2px;">{dd_pct*100:.1f}%</div>
    </div>"""


def fmt_pct(v: float) -> str:
    return f"{v:+.2f}%"


def fmt_dollar(v: float) -> str:
    return f"${v:,.0f}"


def color_for(v: float, zero_is_neutral: bool = True) -> str:
    if v > 0:
        return "green"
    elif v < 0:
        return "red"
    return "yellow" if zero_is_neutral else ""


def build_equity_fig(
    oos_equity: pd.Series,
    is_equity: pd.Series,
    initial_capital: float,
) -> go.Figure:
    fig = go.Figure()

    if not is_equity.empty:
        fig.add_trace(go.Scatter(
            x=is_equity.index,
            y=is_equity.values,
            name="In-Sample (best per fold)",
            line=dict(color="#ff9800", width=1.5, dash="dot"),
            opacity=0.7,
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>IS</extra>",
        ))

    if not oos_equity.empty:
        colors = []
        for v in oos_equity.values:
            colors.append("#00e676" if v >= initial_capital else "#ff5252")

        fig.add_trace(go.Scatter(
            x=oos_equity.index,
            y=oos_equity.values,
            name="Out-of-Sample (Walk-Forward)",
            line=dict(color="#2979ff", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(41,121,255,0.08)",
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>OOS</extra>",
        ))

    fig.add_hline(y=initial_capital, line_dash="dash", line_color="#ffffff",
                  opacity=0.3, annotation_text="Initial Capital")

    fig.update_layout(
        template="plotly_dark",
        title=dict(text="Walk-Forward Equity Curves", font=dict(size=18, color="#e2e8f0")),
        xaxis=dict(title="Date", showgrid=True, gridcolor="#2a2e45"),
        yaxis=dict(title="Equity ($)", showgrid=True, gridcolor="#2a2e45",
                   tickformat="$,.0f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        margin=dict(l=60, r=30, t=60, b=40),
    )
    return fig


def build_gantt(windows: list, folds: list) -> go.Figure:
    fig = go.Figure()

    fold_lookup = {f["fold"]: f for f in folds}

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        fold_num  = i + 1
        fold_data = fold_lookup.get(fold_num, {})
        oos_pnl   = fold_data.get("oos_stats", {}).get("total_pnl", 0)
        pnl_str   = f"OOS P&L: ${oos_pnl:,.0f}" if fold_data else ""

        # Training bar
        fig.add_trace(go.Bar(
            x=[(tr_e - tr_s).days],
            y=[f"Fold {fold_num}"],
            base=[tr_s.strftime("%Y-%m-%d")],
            orientation="h",
            marker_color="rgba(41,121,255,0.7)",
            name="Training" if i == 0 else None,
            showlegend=(i == 0),
            hovertemplate=(
                f"<b>Fold {fold_num} — Training</b><br>"
                f"{tr_s:%Y-%m-%d} → {tr_e:%Y-%m-%d}<br>"
                f"Duration: {(tr_e - tr_s).days} days<extra></extra>"
            ),
        ))

        # Test bar
        bar_color = "rgba(0,230,118,0.75)" if oos_pnl >= 0 else "rgba(255,82,82,0.75)"
        fig.add_trace(go.Bar(
            x=[(te_e - te_s).days],
            y=[f"Fold {fold_num}"],
            base=[te_s.strftime("%Y-%m-%d")],
            orientation="h",
            marker_color=bar_color,
            name="Blind Test" if i == 0 else None,
            showlegend=(i == 0),
            hovertemplate=(
                f"<b>Fold {fold_num} — Blind Test</b><br>"
                f"{te_s:%Y-%m-%d} → {te_e:%Y-%m-%d}<br>"
                f"{pnl_str}<extra></extra>"
            ),
        ))

    fig.update_layout(
        template="plotly_dark",
        title=dict(text="Walk-Forward Windows — Training (Blue) | Test (Green/Red)", font=dict(size=16)),
        barmode="overlay",
        xaxis=dict(
            type="date",
            title="Date",
            showgrid=True,
            gridcolor="#2a2e45",
        ),
        yaxis=dict(autorange="reversed", showgrid=False),
        height=max(300, len(windows) * 42 + 80),
        margin=dict(l=80, r=30, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_drawdown_fig(equity: pd.Series) -> go.Figure:
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index,
        y=dd.values,
        fill="tozeroy",
        fillcolor="rgba(255,82,82,0.18)",
        line=dict(color="#ff5252", width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra>Drawdown</extra>",
        name="Drawdown",
    ))

    fig.update_layout(
        template="plotly_dark",
        title="OOS Drawdown",
        xaxis=dict(showgrid=True, gridcolor="#2a2e45"),
        yaxis=dict(title="Drawdown (%)", tickformat=".1f", showgrid=True, gridcolor="#2a2e45"),
        height=250,
        margin=dict(l=60, r=30, t=50, b=40),
    )
    return fig


def build_monthly_returns(equity: pd.Series) -> go.Figure:
    monthly = equity.resample("ME").last()
    monthly_ret = monthly.pct_change().dropna() * 100

    colors = ["#00e676" if v >= 0 else "#ff5252" for v in monthly_ret.values]

    fig = go.Figure(go.Bar(
        x=monthly_ret.index,
        y=monthly_ret.values,
        marker_color=colors,
        hovertemplate="%{x|%b %Y}<br>%{y:+.2f}%<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        title="Monthly OOS Returns",
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Return (%)", tickformat="+.1f", showgrid=True, gridcolor="#2a2e45"),
        height=280,
        margin=dict(l=60, r=30, t=50, b=40),
    )
    return fig


def build_fold_comparison(folds: list) -> go.Figure:
    if not folds:
        return go.Figure()

    fold_nums = [f["fold"] for f in folds]
    is_pnl    = [f["is_stats"].get("total_pnl", 0) for f in folds]
    oos_pnl   = [f["oos_stats"].get("total_pnl", 0) for f in folds]
    is_sharpe  = [f["is_stats"].get("sharpe", 0) for f in folds]
    oos_sharpe = [f["oos_stats"].get("sharpe", 0) for f in folds]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("P&L by Fold (IS vs OOS)", "Sharpe by Fold (IS vs OOS)"))

    fig.add_trace(go.Bar(name="IS P&L", x=[f"F{n}" for n in fold_nums], y=is_pnl,
                         marker_color="rgba(255,152,0,0.7)",
                         hovertemplate="Fold %{x}<br>$%{y:,.0f}<extra>IS</extra>"),
                  row=1, col=1)
    fig.add_trace(go.Bar(name="OOS P&L", x=[f"F{n}" for n in fold_nums], y=oos_pnl,
                         marker_color=["rgba(0,230,118,0.8)" if v >= 0 else "rgba(255,82,82,0.8)" for v in oos_pnl],
                         hovertemplate="Fold %{x}<br>$%{y:,.0f}<extra>OOS</extra>"),
                  row=1, col=1)

    fig.add_trace(go.Bar(name="IS Sharpe", x=[f"F{n}" for n in fold_nums], y=is_sharpe,
                         marker_color="rgba(255,152,0,0.7)", showlegend=False,
                         hovertemplate="Fold %{x}<br>%{y:.2f}<extra>IS Sharpe</extra>"),
                  row=1, col=2)
    fig.add_trace(go.Bar(name="OOS Sharpe", x=[f"F{n}" for n in fold_nums], y=oos_sharpe,
                         marker_color=["rgba(0,230,118,0.8)" if v >= 0 else "rgba(255,82,82,0.8)" for v in oos_sharpe],
                         showlegend=False,
                         hovertemplate="Fold %{x}<br>%{y:.2f}<extra>OOS Sharpe</extra>"),
                  row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        height=360,
        margin=dict(l=40, r=30, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=0.5),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — configuration
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<h2 style="color: #f2c4ce;">Precision Trader</h2>', unsafe_allow_html=True)

    # ── Strategy selector ────────────────────────────────────────────────────
    st.markdown("### Strategy")
    strat_list = list_strategies()
    strat_names = {k: name for k, name, _ in strat_list}
    strat_descs = {k: desc for k, _, desc in strat_list}

    selected_strategy_key = st.selectbox(
        "Select Strategy",
        options=[k for k, _, _ in strat_list],
        format_func=lambda k: _format_strategy_name(k, strat_names[k]),
        index=0,
    )
    _render_strat_badge(selected_strategy_key, strat_names[selected_strategy_key])
    st.caption(strat_descs[selected_strategy_key])
    active_strategy = get_strategy(selected_strategy_key)

    # ── Data source ──────────────────────────────────────────────────────────
    st.markdown("### Data Source")

    timeframe = st.selectbox(
        "Timeframe",
        options=["1m", "15m", "1h", "1d"],
        index=0,
        help=(
            "1m  — tick-level backtesting (upload CSV, or yfinance last 7 days)\n"
            "15m — intraday scalping/day-trading (yfinance: last 60 days, or upload CSV)\n"
            "1h  — intraday swing (yfinance: last ~2 years automatically)\n"
            "1d  — daily swing/position (yfinance: last 10 years automatically)"
        ),
    )

    TF_INFO = {
        "1m":  {"yf_limit": "last 7 days only", "yf_period": "7d"},
        "15m": {"yf_limit": "~58 days", "yf_period": "58d"},
        "1h":  {"yf_limit": "~2 years (730 days)", "yf_period": "730d"},
        "1d":  {"yf_limit": "~10 years", "yf_period": "10y"},
    }

    _source_options = [
        "Upload CSV/XLSX",
        f"Live from yfinance ({TF_INFO[timeframe]['yf_limit']})",
        "MNQ 1-Min (built-in, 6+ years)",
        "MYM 1-Min (built-in, 4+ years)",
    ]

    # Default to built-in MNQ for 1m timeframe
    _default_src = 2 if timeframe == "1m" else 0

    data_source = st.radio(
        "Data Source",
        _source_options,
        index=_default_src,
        help=(
            "Upload your own OHLCV data, pull from Yahoo Finance, "
            "or use the built-in MNQ 1-min OHLCV dataset (2.2M bars, Jan 2020 - Mar 2026)."
        ),
    )

    uploaded_file = None
    ticker = "MNQ=F"

    if data_source == "Upload CSV/XLSX":
        uploaded_file = st.file_uploader(
            f"Upload {timeframe} OHLCV data",
            type=["csv", "xlsx", "xls"],
            help=(
                "Required columns: datetime (or date/time/timestamp), open, high, low, close, volume.\n"
                "Export from TradingView → right-click chart → Export chart data.\n"
                "Or from NinjaTrader / TradeStation as CSV."
            ),
        )
        if uploaded_file:
            st.caption(f"File: **{uploaded_file.name}**")
    elif "yfinance" in data_source:
        ticker = st.text_input("Ticker (yfinance)", value="NQ=F" if timeframe != "15m" else "MNQ=F")
        st.info(f"yfinance {timeframe} data: {TF_INFO[timeframe]['yf_limit']}.")
    elif "MNQ 1-Min" in data_source:
        st.caption("MNQ continuous front-month, 1-min OHLCV bars, Jan 2020 - Mar 2026 (2.2M bars)")
    elif "MYM 1-Min" in data_source:
        st.caption("MYM continuous front-month, 1-min OHLCV bars, Jan 2022 - Apr 2026 (1.5M bars)")
    # ── Walk-forward settings ────────────────────────────────────────────────
    st.markdown("### Walk-Forward Settings")
    train_months = st.slider("Training Window (months)", 6, 24, 12)
    test_months  = st.slider("Test Window (months)", 1, 6, 3)

    st.markdown("### General")
    initial_capital = st.number_input("Initial Capital ($)", value=50000, step=1000)

    # ── Dynamic parameter grid from selected strategy ────────────────────────
    st.markdown("### Parameter Grid to Optimize")
    grid_options = active_strategy.sidebar_grid_options()

    custom_grid = {}
    with st.expander("Edit Grid (advanced)", expanded=False):
        for param_key, opts in grid_options.items():
            selected = st.multiselect(
                opts["label"],
                options=opts["options"],
                default=opts["default"],
                key=f"grid_{selected_strategy_key}_{param_key}",
            )
            custom_grid[param_key] = selected if selected else opts["default"]

    # If grid expander wasn't opened, use defaults
    if not custom_grid:
        custom_grid = active_strategy.default_grid()

    # ── Save / Load parameters ────────────────────────────────────────────────
    st.markdown("### Save / Load Params")

    # Load saved params if they exist
    if has_saved_params(selected_strategy_key):
        saved = load_params(selected_strategy_key)
        st.caption(f"Saved params found for **{strat_names[selected_strategy_key]}**")
        if st.button("Load Saved Params", key="load_params_btn"):
            st.session_state["loaded_params"] = saved
            st.success("Loaded! Re-run to apply.")
            st.rerun()

    save_col1, save_col2 = st.columns(2)
    with save_col1:
        if st.button("Save Current Params", key="save_params_btn"):
            # Build single-value params from grid (take first value of each)
            single_params = {}
            for k, v in custom_grid.items():
                single_params[k] = v[0] if isinstance(v, list) and v else v
            single_params["initial_capital"] = initial_capital
            path = save_params(selected_strategy_key, single_params)
            st.success(f"Saved to `{path}`")
    with save_col2:
        # Show saved params as JSON download
        single_for_export = {}
        for k, v in custom_grid.items():
            single_for_export[k] = v[0] if isinstance(v, list) and v else v
        single_for_export["initial_capital"] = initial_capital
        import json
        st.download_button(
            "Export as JSON",
            data=json.dumps(single_for_export, indent=2),
            file_name=f"{selected_strategy_key}_params.json",
            mime="application/json",
            key="export_params_btn",
        )

    st.markdown("---")
    run_button = st.button("Run Walk-Forward Analysis", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("**Cost Assumptions**")
    st.caption("Exchange fee: 0.10% per trade")
    st.caption("Slippage:     0.05% per trade")
    st.caption("Total:        0.15% per side")

    st.markdown("---")
    with st.expander("Strategy Color Legend", expanded=False):
        _render_strat_legend()


# ──────────────────────────────────────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<h1 style="background: linear-gradient(90deg, #f58f7c, #f2c4ce); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700;">Precision Trader</h1>', unsafe_allow_html=True)

from optimizer import run_optimization
tab_wfa, tab_opt, tab_bt, tab_analysis, tab_custom, tab_about = st.tabs(["Walk-Forward", "Optimizer", "Backtrader", "Analysis", "Custom Strategy", "About"])

# ──────────────────────────────────────────────────────────────────────────────
# Tab 1: Walk-Forward
# ──────────────────────────────────────────────────────────────────────────────

with tab_wfa:

    if run_button:
        total_combos = 1
        for v in custom_grid.values():
            total_combos *= len(v)

        # Load data
        with st.spinner("Loading data..."):
            df, _err = load_data_from_source(data_source, uploaded_file, ticker, timeframe)
        if _err:
            st.error(_err)
            st.stop()
        if df.empty:
            st.error("No data available. Check your data source settings.")
            st.stop()

        data_label = uploaded_file.name if uploaded_file else ticker
        st.success(
            f"Loaded **{len(df):,}** {timeframe} bars of **{data_label}**  "
            f"({df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d})"
        )

        import multiprocessing as mp
        n_cores = max(1, mp.cpu_count() - 1)

        # Progress bar + status
        progress_bar = st.progress(0.0, text="Initialising...")
        status_text  = st.empty()

        st.info(
            f"**Grid:** {total_combos:,} combos/fold  |  "
            f"**Parallelism:** {n_cores} CPU cores  |  "
            f"**Est. time:** ~{max(1, total_combos * 0.75 / n_cores / 60):.0f} min/fold"
        )

        def progress_cb(fold_i, total_folds, phase, pct):
            progress_bar.progress(min(pct, 0.99), text=f"Fold {fold_i}/{total_folds} — {phase}")
            status_text.markdown(
                f"**Optimising:** Fold **{fold_i}** / {total_folds} &nbsp;|&nbsp; "
                f"{total_combos:,} combos × {n_cores} cores"
            )

        try:
            wfa_result = run_walk_forward(
                df,
                train_months=train_months,
                test_months=test_months,
                param_grid=custom_grid,
                progress_cb=progress_cb,
                n_workers=n_cores,
                strategy_key=selected_strategy_key,
            )
        except Exception as e:
            st.error(f"Error during walk-forward: {e}")
            st.stop()

        progress_bar.progress(1.0, text="Complete!")
        status_text.empty()

        if not wfa_result or not wfa_result.get("folds"):
            st.warning("No complete folds produced. Try a longer date range.")
            st.stop()

        folds      = wfa_result["folds"]
        oos_equity = wfa_result["oos_equity"]
        is_equity  = wfa_result["is_equity"]
        oos_stats  = wfa_result["oos_stats"]
        windows    = wfa_result["windows"]

        # Store in session state for persistence
        st.session_state["wfa_result"] = wfa_result
        st.session_state["ticker"]     = ticker

    # ── Render results ────────────────────────────────────────────────────────
    if "wfa_result" not in st.session_state:
        st.info("Configure parameters in the sidebar and click **Run Walk-Forward Analysis** to begin.")
    else:
        wfa_result = st.session_state["wfa_result"]
        folds      = wfa_result["folds"]
        oos_equity = wfa_result["oos_equity"]
        is_equity  = wfa_result["is_equity"]
        oos_stats  = wfa_result["oos_stats"]
        windows    = wfa_result["windows"]
        _ticker    = st.session_state.get("ticker", ticker)

        # ── KPI row ───────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Overall Walk-Forward Performance</div>', unsafe_allow_html=True)

        n_folds     = len(folds)
        total_pnl   = oos_stats.get("total_pnl", 0)
        win_rate    = oos_stats.get("win_rate", 0) * 100
        sharpe      = oos_stats.get("sharpe", 0)
        pf          = oos_stats.get("profit_factor", 0)
        mdd         = oos_stats.get("max_drawdown", 0) * 100
        n_trades    = oos_stats.get("total_trades", 0)
        expectancy  = oos_stats.get("expectancy", 0)
        net_ret     = oos_stats.get("net_return_pct", 0)

        winning_folds = sum(1 for f in folds if f["oos_stats"].get("total_pnl", 0) > 0)

        cols = st.columns(9)
        metrics = [
            ("Total OOS P&L",      fmt_dollar(total_pnl),       color_for(total_pnl)),
            ("Net Return",         fmt_pct(net_ret),             color_for(net_ret)),
            ("Sharpe Ratio",       f"{sharpe:.2f}",              color_for(sharpe)),
            ("Profit Factor",      f"{pf:.2f}",                  color_for(pf - 1)),
            ("Win Rate",           f"{win_rate:.1f}%",           color_for(win_rate - 50)),
            ("Max Drawdown",       f"{mdd:.1f}%",                "red" if mdd < -5 else "yellow"),
            ("Total Trades",       str(n_trades),                ""),
            ("Expectancy",         fmt_dollar(expectancy),       color_for(expectancy)),
            ("Profitable Folds",   f"{winning_folds}/{n_folds}", color_for(winning_folds - n_folds / 2)),
        ]
        for col, (label, value, color) in zip(cols, metrics):
            with col:
                st.markdown(metric_card(label, value, color), unsafe_allow_html=True)

        st.markdown("---")

        # ── Gantt chart ────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Walk-Forward Windows</div>', unsafe_allow_html=True)
        st.plotly_chart(build_gantt(windows, folds), use_container_width=True)

        # ── Equity curves ──────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Equity Curves — IS vs OOS</div>', unsafe_allow_html=True)
        st.plotly_chart(
            build_equity_fig(oos_equity, is_equity, initial_capital),
            use_container_width=True
        )

        # ── Secondary charts ───────────────────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(build_drawdown_fig(oos_equity), use_container_width=True)
        with c2:
            st.plotly_chart(build_monthly_returns(oos_equity), use_container_width=True)

        # ── Fold comparison ────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Fold-by-Fold Comparison</div>', unsafe_allow_html=True)
        st.plotly_chart(build_fold_comparison(folds), use_container_width=True)

        # ── Side-by-side metrics table ────────────────────────────────────────
        st.markdown('<div class="section-header">IS vs OOS Metrics (per fold)</div>', unsafe_allow_html=True)

        rows = []
        for f in folds:
            is_s  = f["is_stats"]
            oos_s = f["oos_stats"]
            rows.append({
                "Fold":           f["fold"],
                "Train":          f"{f['train_start']:%Y-%m-%d} → {f['train_end']:%Y-%m-%d}",
                "Test":           f"{f['test_start']:%Y-%m-%d} → {f['test_end']:%Y-%m-%d}",
                "IS P&L":         f"${is_s.get('total_pnl', 0):,.0f}",
                "OOS P&L":        f"${oos_s.get('total_pnl', 0):,.0f}",
                "IS Trades":      is_s.get("total_trades", 0),
                "OOS Trades":     oos_s.get("total_trades", 0),
                "IS Win%":        f"{is_s.get('win_rate', 0)*100:.1f}%",
                "OOS Win%":       f"{oos_s.get('win_rate', 0)*100:.1f}%",
                "IS Sharpe":      f"{is_s.get('sharpe', 0):.2f}",
                "OOS Sharpe":     f"{oos_s.get('sharpe', 0):.2f}",
                "IS PF":          f"{min(is_s.get('profit_factor', 0), 99):.2f}",
                "OOS PF":         f"{min(oos_s.get('profit_factor', 0), 99):.2f}",
                "OOS MaxDD":      f"{oos_s.get('max_drawdown', 0)*100:.1f}%",
            })

        fold_df = pd.DataFrame(rows)
        st.dataframe(
            fold_df.set_index("Fold"),
            use_container_width=True,
            height=min(400, 40 + len(folds) * 38),
        )

        # ── Best params per fold ───────────────────────────────────────────────
        st.markdown('<div class="section-header">Best Parameters per Fold</div>', unsafe_allow_html=True)

        param_keys = ["ema_fast_len", "ema_slow_len", "ema_trend_len",
                      "min_score", "rsi_len", "fixed_risk_pts",
                      "tp1_rr", "tp2_rr", "st_factor"]
        param_rows = []
        for f in folds:
            bp = f.get("best_params", {})
            row = {"Fold": f["fold"]}
            row.update({k: bp.get(k, "—") for k in param_keys})
            param_rows.append(row)

        st.dataframe(pd.DataFrame(param_rows).set_index("Fold"), use_container_width=True)

        # ── OOS trade log ──────────────────────────────────────────────────────
        with st.expander("OOS Trade Log", expanded=False):
            trades_df = pd.DataFrame(wfa_result["oos_trades"])
            if not trades_df.empty:
                trades_df["pnl"] = trades_df["pnl"].round(2)
                st.dataframe(trades_df, use_container_width=True, height=400)
            else:
                st.info("No trades recorded.")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2: Strategy Optimizer
# ──────────────────────────────────────────────────────────────────────────────

with tab_opt:
    st.markdown("### Strategy Optimizer")
    st.markdown("Find the **exact best parameters** for any strategy. Toggle which params to test.")

    # Strategy selector
    opt_strat_list = list_strategies()
    opt_strat_names = {k: name for k, name, _ in opt_strat_list}

    opt_strategy_key = st.selectbox(
        "Strategy to Optimize",
        options=[k for k, _, _ in opt_strat_list],
        format_func=lambda k: _format_strategy_name(k, opt_strat_names[k]),
        index=0,
        key="opt_strat_select",
    )
    _render_strat_badge(opt_strategy_key, opt_strat_names[opt_strategy_key])
    opt_active = get_strategy(opt_strategy_key)
    opt_grid_opts = opt_active.sidebar_grid_options()
    _opt_saved = load_params(opt_strategy_key)

    # Build flexible parameter grid — 3 modes per param
    st.markdown("#### Parameter Grid")
    st.caption("Each parameter: **Range** (start/stop/step), **Manual** (type any values), or **Locked** (single value).")
    if _opt_saved:
        st.caption(f"Locked defaults use your saved parameters for **{opt_strat_names[opt_strategy_key]}**.")

    opt_grid = {}
    param_keys_list = list(opt_grid_opts.keys())

    for pk in param_keys_list:
        opts = opt_grid_opts[pk]
        default_val = opts["default"][0] if opts["default"] else opts["options"][0]
        is_float = isinstance(default_val, float)
        # Use saved param as locked default if available
        saved_val = _opt_saved.get(pk) if _opt_saved else None
        if saved_val is not None:
            try:
                locked_default = float(saved_val) if is_float else int(saved_val)
            except (ValueError, TypeError):
                locked_default = default_val
        else:
            locked_default = default_val

        with st.expander(f"{opts['label']}", expanded=False):
            mode = st.radio(
                "Mode",
                ["Range", "Manual", "Locked"],
                index=0,
                horizontal=True,
                key=f"opt_mode_{opt_strategy_key}_{pk}",
            )

            if mode == "Range":
                rc1, rc2, rc3 = st.columns(3)
                if is_float:
                    with rc1:
                        r_start = st.number_input("Start", value=float(min(opts["options"])), step=0.5, key=f"opt_rs_{opt_strategy_key}_{pk}")
                    with rc2:
                        r_stop = st.number_input("Stop", value=float(max(opts["options"])), step=0.5, key=f"opt_re_{opt_strategy_key}_{pk}")
                    with rc3:
                        r_step = st.number_input("Step", value=round((float(max(opts["options"])) - float(min(opts["options"]))) / max(len(opts["options"]) - 1, 1), 2), min_value=0.01, step=0.1, key=f"opt_rst_{opt_strategy_key}_{pk}")
                    # Generate range
                    import numpy as _np
                    vals = _np.arange(r_start, r_stop + r_step * 0.5, r_step).tolist()
                    vals = [round(v, 4) for v in vals if v <= r_stop + 0.001]
                else:
                    with rc1:
                        r_start = st.number_input("Start", value=int(min(opts["options"])), step=1, key=f"opt_rs_{opt_strategy_key}_{pk}")
                    with rc2:
                        r_stop = st.number_input("Stop", value=int(max(opts["options"])), step=1, key=f"opt_re_{opt_strategy_key}_{pk}")
                    with rc3:
                        r_step = st.number_input("Step", value=max(1, (int(max(opts["options"])) - int(min(opts["options"]))) // max(len(opts["options"]) - 1, 1)), min_value=1, step=1, key=f"opt_rst_{opt_strategy_key}_{pk}")
                    vals = list(range(int(r_start), int(r_stop) + 1, int(r_step)))

                st.caption(f"Testing **{len(vals)}** values: {vals[:15]}{'...' if len(vals) > 15 else ''}")
                opt_grid[pk] = vals

            elif mode == "Manual":
                manual_str = st.text_input(
                    "Enter values (comma-separated)",
                    value=", ".join(str(v) for v in opts["default"]),
                    key=f"opt_manual_{opt_strategy_key}_{pk}",
                )
                try:
                    if is_float:
                        vals = [float(v.strip()) for v in manual_str.split(",") if v.strip()]
                    else:
                        vals = [int(float(v.strip())) for v in manual_str.split(",") if v.strip()]
                    st.caption(f"Testing **{len(vals)}** values: {vals}")
                except ValueError:
                    vals = opts["default"]
                    st.warning("Could not parse values. Using defaults.")
                opt_grid[pk] = vals if vals else opts["default"]

            else:  # Locked
                if is_float:
                    locked = st.number_input(
                        "Locked value",
                        value=locked_default, step=0.1,
                        key=f"opt_lock_{opt_strategy_key}_{pk}",
                    )
                else:
                    locked = st.number_input(
                        "Locked value",
                        value=int(locked_default), step=1,
                        key=f"opt_lock_{opt_strategy_key}_{pk}",
                    )
                opt_grid[pk] = [locked]

    # ── Data source (independent from sidebar) ─────────────────────────────
    st.markdown("#### Data Source")
    opt_dataset = st.selectbox(
        "Select dataset",
        options=list(BUILTIN_DATASETS.keys()) + ["Upload CSV/XLSX", "yfinance"],
        format_func=lambda k: BUILTIN_DATASETS[k]["label"] if k in BUILTIN_DATASETS else k,
        index=0,
        key="opt_dataset",
    )
    opt_uploaded = None
    opt_ticker = "NQ=F"
    if opt_dataset == "Upload CSV/XLSX":
        opt_uploaded = st.file_uploader("Upload OHLCV data for optimizer", type=["csv", "xlsx", "xls"], key="opt_file_upload")
    elif opt_dataset == "yfinance":
        opt_ticker = st.text_input("Ticker", value="NQ=F", key="opt_ticker")
    elif opt_dataset in BUILTIN_DATASETS:
        st.caption(BUILTIN_DATASETS[opt_dataset]["caption"])

    # Date range for optimizer
    st.markdown('<span style="color:#f58f7c;font-weight:600;">Date Range</span>', unsafe_allow_html=True)
    opt_date_range = st.radio(
        "Optimize on",
        ["Last 30 days", "Last 90 days", "Last 6 months", "Last 1 year", "Last 2 years", "All data", "Custom"],
        index=3,
        horizontal=True,
        key="opt_date_range",
    )
    opt_start_date = None
    opt_end_date = None
    if opt_date_range == "Custom":
        _odr1, _odr2 = st.columns(2)
        with _odr1:
            opt_start_date = st.date_input("Start", value=date.today() - pd.Timedelta(days=365), key="opt_start")
        with _odr2:
            opt_end_date = st.date_input("End", value=date.today(), key="opt_end")
    elif opt_date_range != "All data":
        _opt_days = {"Last 30 days": 30, "Last 90 days": 90, "Last 6 months": 180, "Last 1 year": 365, "Last 2 years": 730}
        opt_start_date = date.today() - pd.Timedelta(days=_opt_days[opt_date_range])
        opt_end_date = date.today()

    # Resample option for large datasets
    st.markdown('<span style="color:#f58f7c;font-weight:600;">Timeframe</span>', unsafe_allow_html=True)
    opt_resample = st.radio(
        "Run strategy on timeframe",
        ["1m", "5m", "15m", "1h", "1d"],
        index=0,
        horizontal=True,
        key="opt_resample",
    )

    # Combo count
    total_combos_opt = 1
    for v in opt_grid.values():
        total_combos_opt *= len(v)

    import multiprocessing as _mp
    n_cores_opt = max(1, _mp.cpu_count() - 1)
    # Backtrader runs ~1.5s per combo vs 0.07s for custom engine
    est_time = max(1, total_combos_opt * 1.5 / n_cores_opt / 60)

    st.info(
        f"**{total_combos_opt:,}** parameter combinations  |  "
        f"**{n_cores_opt}** CPU cores  |  "
        f"Est. **~{est_time:.0f} min** (backtrader broker simulation)"
    )

    opt_top_n = st.slider("Show top N results", 5, 100, 20, key="opt_top_n")

    # ── Broker Settings (must match Backtrader tab) ──────────────────────────
    st.markdown("#### Broker & Risk Settings")
    st.caption("These must match your Backtrader tab settings for consistent results.")
    opt_broker_cols = st.columns(3)
    with opt_broker_cols[0]:
        opt_capital = st.number_input("Capital ($)", value=50000, step=1000, key="opt_capital")
    with opt_broker_cols[1]:
        opt_use_max_trade = st.checkbox("Max Loss Per Trade", value=True, key="opt_max_trade_on")
        opt_max_trade_loss = st.number_input("Max Trade Loss ($)", value=300.0, step=50.0, key="opt_max_trade_val")
    with opt_broker_cols[2]:
        opt_use_daily_max = st.checkbox("Daily Max Loss", value=True, key="opt_daily_max_on")
        opt_daily_max_loss = st.number_input("Daily Max Loss ($)", value=1000.0, step=50.0, key="opt_daily_max_val")

    # ── Ranking Metric ───────────────────────────────────────────────────────
    st.markdown("#### Rank Results By")
    rank_metric_options = {
        "Composite Score (Sharpe × PF × (1-DD))": "score",
        "Total P&L ($)":                           "total_pnl",
        "Profit Factor":                           "profit_factor",
        "Sharpe Ratio":                            "sharpe",
        "Win Rate":                                "win_rate",
        "Max Drawdown (lowest)":                   "max_drawdown",
        "Total Trades":                            "total_trades",
        "Net Return %":                            "net_return_pct",
    }
    rank_label = st.selectbox(
        "Sort optimization results by",
        options=list(rank_metric_options.keys()),
        index=0,
        key="opt_rank_metric",
    )
    opt_rank_key = rank_metric_options[rank_label]

    # ── Engine Selection ─────────────────────────────────────────────────────
    st.markdown("#### Backtest Engine")
    opt_engine = st.radio(
        "Engine",
        ["Native", "Backtrader"],
        index=0,
        horizontal=True,
        key="opt_engine",
    )

    # Reset flag so cached results display correctly
    st.session_state["_opt_just_ran"] = False

    if st.button("Run Optimizer", type="primary", key="opt_run"):
        st.session_state["_opt_just_ran"] = True
        # Load data from optimizer's own source
        df_opt = pd.DataFrame()
        if opt_dataset in BUILTIN_DATASETS:
            _ds_info = BUILTIN_DATASETS[opt_dataset]
            with st.spinner(f"Loading built-in {_ds_info['label']}..."):
                df_opt = globals()[_ds_info["loader"]]()
        elif opt_dataset == "Upload CSV/XLSX":
            if opt_uploaded is None:
                st.error("Upload a data file above.")
                st.stop()
            try:
                df_opt = load_from_csv(opt_uploaded)
            except Exception as e:
                st.error(f"Parse error: {e}")
                st.stop()
        elif opt_dataset == "yfinance":
            with st.spinner(f"Fetching data for {opt_ticker}..."):
                df_opt = load_from_yfinance(opt_ticker, interval=opt_resample)

        if df_opt.empty:
            st.error("No data available. Check ticker or upload a file.")
            st.stop()

        # Apply date range filter
        if opt_start_date is not None and opt_end_date is not None:
            _os = pd.Timestamp(opt_start_date)
            _oe = pd.Timestamp(opt_end_date) + pd.Timedelta(days=1)
            if df_opt.index.tz is not None:
                _os = _os.tz_localize(df_opt.index.tz)
                _oe = _oe.tz_localize(df_opt.index.tz)
            df_opt = df_opt.loc[_os:_oe]
            if df_opt.empty:
                st.error("No data in selected date range.")
                st.stop()

        # Resample if needed
        _opt_resample_map = {"1m": None, "5m": "5min", "15m": "15min", "1h": "1h", "1d": "1D"}
        _opt_resample_rule = _opt_resample_map.get(opt_resample)
        _pre_len = len(df_opt)
        if _opt_resample_rule is not None:
            df_opt = resample_ohlcv(df_opt, _opt_resample_rule)

        opt_tf_label = opt_resample if _opt_resample_rule else ("1m" if opt_dataset in BUILTIN_DATASETS else opt_resample)
        st.success(
            f"Loaded **{_pre_len:,}** bars"
            + (f" → resampled to **{len(df_opt):,}** bars ({opt_resample})" if _opt_resample_rule else f" ({opt_tf_label})")
            + f" — {df_opt.index[0]:%Y-%m-%d} to {df_opt.index[-1]:%Y-%m-%d}"
        )

        progress_opt = st.progress(0.0, text="Starting optimizer...")
        status_opt   = st.empty()

        def opt_progress_cb(completed, total):
            pct = completed / total
            progress_opt.progress(min(pct, 0.99), text=f"{completed:,} / {total:,} combos tested")
            status_opt.markdown(f"**{completed:,}** / {total:,} tested  |  {n_cores_opt} cores")

        # Extra fixed params — ensures optimizer uses same broker/risk settings as Backtrader tab
        # Only include risk params if they're NOT already in the optimization grid
        opt_extra_params = {
            "initial_capital": float(opt_capital),
            "point_value": 2.0,
            "use_max_trade_loss": opt_use_max_trade,
            "use_daily_max_loss": opt_use_daily_max,
        }
        if "max_trade_loss" not in opt_grid:
            opt_extra_params["max_trade_loss"] = opt_max_trade_loss
        if "daily_max_loss" not in opt_grid:
            opt_extra_params["daily_max_loss"] = opt_daily_max_loss

        try:
            _use_native_engine = "Native" in opt_engine
            opt_result = run_optimization(
                df_opt,
                strategy_key=opt_strategy_key,
                param_grid=opt_grid,
                n_workers=n_cores_opt,
                progress_cb=opt_progress_cb,
                top_n=opt_top_n,
                extra_params=opt_extra_params,
                rank_by=opt_rank_key,
                use_native=_use_native_engine,
            )
        except Exception as e:
            st.error(f"Optimizer failed: {e}")
            st.stop()

        progress_opt.progress(1.0, text="Complete!")
        status_opt.empty()

        best = opt_result["best"]
        top_results = opt_result["top_n"]
        all_opt_results = opt_result["results"]

        # Save data source info so equity curve viewer can reload
        if opt_dataset in BUILTIN_DATASETS:
            _df_info = {"source": f"builtin_{opt_dataset}", "loader": BUILTIN_DATASETS[opt_dataset]["loader"], "resample": opt_resample}
        elif opt_dataset == "Upload CSV/XLSX":
            _df_info = {"source": "upload"}
        elif opt_dataset == "yfinance":
            _df_info = {"source": "yfinance", "ticker": opt_ticker, "tf": opt_resample}

        # Persist results in session_state so they survive tab switches
        st.session_state["opt_results"] = {
            "best": best,
            "top_n": top_results,
            "all_results": all_opt_results,
            "strategy_key": opt_strategy_key,
            "top_n_count": opt_top_n,
            "df_info": _df_info,
            "extra_params": opt_extra_params,
            "rank_by": opt_rank_key,
            "rank_label": rank_label,
        }

        if best is None or best["score"] <= -999:
            st.warning("No valid parameter combinations found. Try a wider grid or more data.")
        else:
            # ── Best result banner ───────────────────────────────────────────
            _rank_label = st.session_state.get("opt_results", {}).get("rank_label", "Composite Score")
            st.markdown(f'<div class="section-header">Best Parameters Found — Ranked by {_rank_label}</div>', unsafe_allow_html=True)

            best_cols = st.columns(7)
            best_metrics = [
                ("P&L",     fmt_dollar(best.get("total_pnl", 0)),    color_for(best.get("total_pnl", 0))),
                ("Return",  fmt_pct(best.get("net_return_pct", 0)),   color_for(best.get("net_return_pct", 0))),
                ("Sharpe",  f"{best.get('sharpe', 0):.2f}",           color_for(best.get("sharpe", 0))),
                ("PF",      f"{min(best.get('profit_factor', 0), 99):.2f}", color_for(best.get("profit_factor", 0) - 1)),
                ("Win Rate", f"{best.get('win_rate', 0)*100:.1f}%",   color_for(best.get("win_rate", 0) - 0.5)),
                ("Max DD",  f"{best.get('max_drawdown', 0)*100:.1f}%", "red"),
                ("Trades",  str(best.get("total_trades", 0)),         ""),
            ]
            for col, (label, value, clr) in zip(best_cols, best_metrics):
                with col:
                    st.markdown(metric_card(label, value, clr), unsafe_allow_html=True)

            # ── Best params display ──────────────────────────────────────────
            st.markdown("**Winning Parameters:**")
            bp = best["params"]
            param_strs = [f"**{k}** = `{v}`" for k, v in bp.items()]
            st.markdown(" &nbsp;|&nbsp; ".join(param_strs))

            # Save best params button
            scol1, scol2 = st.columns(2)
            with scol1:
                if st.button("Save Best Params", key="opt_save_best"):
                    from param_sync import save_params as sp
                    path = sp(opt_strategy_key, bp)
                    st.success(f"Saved to `{path}`")
            with scol2:
                import json as _json
                st.download_button(
                    "Export Best as JSON",
                    data=_json.dumps(bp, indent=2),
                    file_name=f"{opt_strategy_key}_best_params.json",
                    mime="application/json",
                    key="opt_export_best",
                )

            # ── Top N results table ──────────────────────────────────────────
            st.markdown(f'<div class="section-header">Top {len(top_results)} Results</div>', unsafe_allow_html=True)

            table_rows = []
            for rank, r in enumerate(top_results, 1):
                row = {
                    "Rank": rank,
                    "Score": f"{r['score']:.4f}",
                    "P&L": f"${r['total_pnl']:,.0f}",
                    "WR": f"{r['win_rate']*100:.1f}%",
                    "PF": f"{r['profit_factor']:.2f}",
                    "Sharpe": f"{r['sharpe']:.2f}",
                    "MaxDD": f"{r['max_drawdown']*100:.1f}%",
                    "Trades": r["total_trades"],
                }
                row.update(r["params"])
                table_rows.append(row)

            results_df = pd.DataFrame(table_rows)
            st.dataframe(results_df.set_index("Rank"), use_container_width=True, height=min(700, 40 + len(top_results) * 38))

            # ── Download buttons for optimizer results ──────────────────────
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    f"Download Top {len(top_results)} Results (CSV)",
                    data=results_df.to_csv(index=False),
                    file_name=f"{opt_strategy_key}_top{len(top_results)}_results.csv",
                    mime="text/csv",
                    key="opt_dl_top_n",
                )
            with dl_col2:
                if "opt_results" in st.session_state and st.session_state["opt_results"].get("all_results"):
                    all_rows = []
                    for rank_i, r_i in enumerate(st.session_state["opt_results"]["all_results"], 1):
                        arow = {
                            "Rank": rank_i,
                            "Score": r_i["score"],
                            "P&L": r_i.get("total_pnl", 0),
                            "WR": r_i.get("win_rate", 0),
                            "PF": r_i.get("profit_factor", 0),
                            "Sharpe": r_i.get("sharpe", 0),
                            "MaxDD": r_i.get("max_drawdown", 0),
                            "Trades": r_i.get("total_trades", 0),
                        }
                        arow.update(r_i["params"])
                        all_rows.append(arow)
                    all_df = pd.DataFrame(all_rows)
                    st.download_button(
                        f"Download All {len(all_rows)} Results (CSV)",
                        data=all_df.to_csv(index=False),
                        file_name=f"{opt_strategy_key}_all_results.csv",
                        mime="text/csv",
                        key="opt_dl_all",
                    )

            # ── Equity curve viewer for individual results ─────────────────
            st.markdown('<div class="section-header">View Equity Curve</div>', unsafe_allow_html=True)
            st.caption("Select a result to see its equity curve via backtrader.")

            rank_options = [f"#{i+1} — P&L: ${r['total_pnl']:,.0f} | WR: {r['win_rate']*100:.0f}% | PF: {r['profit_factor']:.2f}" for i, r in enumerate(top_results)]
            selected_rank = st.selectbox("Select result", options=range(len(rank_options)), format_func=lambda i: rank_options[i], key="opt_eq_select")

            if st.button("Show Equity Curve", key="opt_eq_btn"):
                sel_params = top_results[selected_rank]["params"]
                _opt_extra = st.session_state.get("opt_results", {}).get("extra_params", {})
                sel_params_full = {**sel_params, **_opt_extra}
                with st.spinner(f"Running #{selected_rank+1} through backtrader..."):
                    try:
                        from bt_bw_atr_strategy import run_bt_generic_fast
                        if "opt_results" in st.session_state and "df_info" in st.session_state["opt_results"]:
                            _info = st.session_state["opt_results"]["df_info"]
                            if _info["source"] == "upload":
                                st.error("Re-upload the file to view equity curves.")
                            elif "loader" in _info:
                                _df_eq = globals()[_info["loader"]]()
                                _rs = {"1m": None, "5m": "5min", "15m": "15min", "1h": "1h", "1d": "1D"}.get(_info.get("resample", "1m"))
                                if _rs:
                                    _df_eq = resample_ohlcv(_df_eq, _rs)
                            else:
                                _df_eq = load_from_yfinance(_info["ticker"], interval=_info["tf"])
                                eq_result = run_bt_generic(_df_eq, opt_strategy_key, sel_params_full)
                                # Store in session_state so it persists across reruns
                                st.session_state["opt_eq_result"] = {
                                    "stats": eq_result["stats"],
                                    "equity": eq_result["equity"],
                                    "rank": selected_rank,
                                }
                    except Exception as e:
                        st.error(f"Failed: {e}")

            # Display stored equity curve (persists across reruns)
            if "opt_eq_result" in st.session_state:
                _stored = st.session_state["opt_eq_result"]
                eq_s = _stored["stats"]
                eq_eq = _stored["equity"]
                _rank = _stored["rank"]

                eq_cols = st.columns(5)
                eq_mets = [
                    ("P&L", fmt_dollar(eq_s.get("total_pnl", 0)), color_for(eq_s.get("total_pnl", 0))),
                    ("Sharpe", f"{eq_s.get('sharpe', 0):.2f}", color_for(eq_s.get("sharpe", 0))),
                    ("PF", f"{min(eq_s.get('profit_factor', 0), 99):.2f}", color_for(eq_s.get("profit_factor", 0) - 1)),
                    ("Win Rate", f"{eq_s.get('win_rate', 0)*100:.1f}%", color_for(eq_s.get("win_rate", 0) - 0.5)),
                    ("Trades", str(eq_s.get("total_trades", 0)), ""),
                ]
                for col, (label, value, clr) in zip(eq_cols, eq_mets):
                    with col:
                        st.markdown(metric_card(label, value, clr), unsafe_allow_html=True)

                if not eq_eq.empty:
                    fig_opt_eq = go.Figure(go.Scatter(
                        x=eq_eq.index, y=eq_eq.values,
                        line=dict(color="#e8862a", width=2),
                        fill="tozeroy", fillcolor="rgba(232,134,42,0.08)",
                        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:,.0f}<extra></extra>",
                    ))
                    fig_opt_eq.add_hline(y=50000, line_dash="dash", line_color="#ffffff", opacity=0.3)
                    fig_opt_eq.update_layout(
                        template="plotly_dark",
                        title=f"Equity Curve — Result #{_rank+1}",
                        yaxis=dict(tickformat="$,.0f"),
                        height=400,
                        margin=dict(l=60, r=30, t=50, b=40),
                        plot_bgcolor="#0c1222", paper_bgcolor="#0c1222",
                    )
                    st.plotly_chart(fig_opt_eq, use_container_width=True)

                    st.download_button(
                        f"Download Equity Curve #{_rank+1} (CSV)",
                        data=pd.DataFrame({"date": eq_eq.index, "equity": eq_eq.values}).to_csv(index=False),
                        file_name=f"equity_rank{_rank+1}.csv",
                        mime="text/csv",
                        key="opt_eq_dl",
                    )

            # ── Parameter sensitivity heatmap ────────────────────────────────
            if len(top_results) > 5:
                st.markdown('<div class="section-header">Parameter Distribution (Top Results)</div>', unsafe_allow_html=True)
                # Show which values appear most in top results
                param_freq = {}
                for pk in opt_grid.keys():
                    if len(opt_grid[pk]) > 1:  # only show optimized params
                        freq = {}
                        for r in top_results:
                            val = r["params"].get(pk)
                            if val is not None:
                                freq[val] = freq.get(val, 0) + 1
                        if freq:
                            param_freq[pk] = freq

                if param_freq:
                    freq_cols = st.columns(min(len(param_freq), 3))
                    for idx, (pk, freq) in enumerate(param_freq.items()):
                        col = freq_cols[idx % len(freq_cols)]
                        with col:
                            label = opt_grid_opts.get(pk, {}).get("label", pk)
                            sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
                            fig_bar = go.Figure(go.Bar(
                                x=[str(v) for v, _ in sorted_freq],
                                y=[cnt for _, cnt in sorted_freq],
                                marker_color="#e8862a",
                            ))
                            fig_bar.update_layout(
                                template="plotly_dark",
                                title=label,
                                height=250,
                                margin=dict(l=30, r=10, t=40, b=30),
                                xaxis=dict(title="Value"),
                                yaxis=dict(title=f"Count in Top {len(top_results)}"),
                                plot_bgcolor="#0c1222",
                                paper_bgcolor="#0c1222",
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)

    # ── Show persisted results when returning to tab (button not clicked) ────
    if not st.session_state.get("_opt_just_ran") and "opt_results" in st.session_state:
        _cached = st.session_state["opt_results"]
        _c_best = _cached["best"]
        _c_top = _cached["top_n"]
        if _c_best is not None and _c_best.get("score", -999) > -999:
            st.markdown("---")
            st.markdown(f'<div class="section-header">Previous Optimizer Results ({_cached["strategy_key"]})</div>', unsafe_allow_html=True)

            _c_best_cols = st.columns(7)
            _c_best_metrics = [
                ("P&L",     fmt_dollar(_c_best.get("total_pnl", 0)),    color_for(_c_best.get("total_pnl", 0))),
                ("Return",  fmt_pct(_c_best.get("net_return_pct", 0)),   color_for(_c_best.get("net_return_pct", 0))),
                ("Sharpe",  f"{_c_best.get('sharpe', 0):.2f}",           color_for(_c_best.get("sharpe", 0))),
                ("PF",      f"{min(_c_best.get('profit_factor', 0), 99):.2f}", color_for(_c_best.get("profit_factor", 0) - 1)),
                ("Win Rate", f"{_c_best.get('win_rate', 0)*100:.1f}%",   color_for(_c_best.get("win_rate", 0) - 0.5)),
                ("Max DD",  f"{_c_best.get('max_drawdown', 0)*100:.1f}%", "red"),
                ("Trades",  str(_c_best.get("total_trades", 0)),         ""),
            ]
            for col, (label, value, clr) in zip(_c_best_cols, _c_best_metrics):
                with col:
                    st.markdown(metric_card(label, value, clr), unsafe_allow_html=True)

            st.markdown("**Winning Parameters:**")
            _c_bp = _c_best["params"]
            st.markdown(" &nbsp;|&nbsp; ".join(f"**{k}** = `{v}`" for k, v in _c_bp.items()))

            # Rebuild top N table
            _c_table_rows = []
            for _c_rank, _c_r in enumerate(_c_top, 1):
                _c_row = {
                    "Rank": _c_rank,
                    "Score": f"{_c_r['score']:.4f}",
                    "P&L": f"${_c_r['total_pnl']:,.0f}",
                    "WR": f"{_c_r['win_rate']*100:.1f}%",
                    "PF": f"{_c_r['profit_factor']:.2f}",
                    "Sharpe": f"{_c_r['sharpe']:.2f}",
                    "MaxDD": f"{_c_r['max_drawdown']*100:.1f}%",
                    "Trades": _c_r["total_trades"],
                }
                _c_row.update(_c_r["params"])
                _c_table_rows.append(_c_row)
            _c_results_df = pd.DataFrame(_c_table_rows)
            st.dataframe(_c_results_df.set_index("Rank"), use_container_width=True, height=min(700, 40 + len(_c_top) * 38))

            # Download buttons for cached results
            _c_dl1, _c_dl2 = st.columns(2)
            with _c_dl1:
                st.download_button(
                    f"Download Top {len(_c_top)} Results (CSV)",
                    data=_c_results_df.to_csv(index=False),
                    file_name=f"{_cached['strategy_key']}_top{len(_c_top)}_results.csv",
                    mime="text/csv",
                    key="opt_dl_top_n_cached",
                )
            with _c_dl2:
                if _cached.get("all_results"):
                    _c_all_rows = []
                    for _c_ri, _c_ri_data in enumerate(_cached["all_results"], 1):
                        _c_arow = {
                            "Rank": _c_ri,
                            "Score": _c_ri_data["score"],
                            "P&L": _c_ri_data.get("total_pnl", 0),
                            "WR": _c_ri_data.get("win_rate", 0),
                            "PF": _c_ri_data.get("profit_factor", 0),
                            "Sharpe": _c_ri_data.get("sharpe", 0),
                            "MaxDD": _c_ri_data.get("max_drawdown", 0),
                            "Trades": _c_ri_data.get("total_trades", 0),
                        }
                        _c_arow.update(_c_ri_data["params"])
                        _c_all_rows.append(_c_arow)
                    _c_all_df = pd.DataFrame(_c_all_rows)
                    st.download_button(
                        f"Download All {len(_c_all_rows)} Results (CSV)",
                        data=_c_all_df.to_csv(index=False),
                        file_name=f"{_cached['strategy_key']}_all_results.csv",
                        mime="text/csv",
                        key="opt_dl_all_cached",
                    )


# ──────────────────────────────────────────────────────────────────────────────
# Tab: Analysis
# ──────────────────────────────────────────────────────────────────────────────

with tab_analysis:
    from tab_analysis import render_analysis_tab
    render_analysis_tab(
        list_strategies_fn=list_strategies,
        get_strategy_fn=get_strategy,
        load_params_fn=load_params,
        save_params_fn=save_params,
        load_from_yfinance_fn=load_from_yfinance,
        load_from_csv_fn=load_from_csv,
        run_bt_generic_fn=run_bt_generic,
        metric_card_fn=metric_card,
        fmt_dollar_fn=fmt_dollar,
        color_for_fn=color_for,
        data_source=data_source,
        uploaded_file=uploaded_file,
        timeframe=timeframe,
        ticker=ticker,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tab: Custom Strategy
# ──────────────────────────────────────────────────────────────────────────────

with tab_custom:
    st.markdown("### Upload & Run Custom Strategies")
    st.markdown(
        "Paste your Python strategy code below (or load a saved one). "
        "It plugs directly into the walk-forward engine with full optimization."
    )

    # ── Load saved strategies ────────────────────────────────────────────────
    saved_files = list_saved_strategies()
    col_load, col_save = st.columns([2, 1])

    with col_load:
        if saved_files:
            selected_saved = st.selectbox(
                "Load saved strategy",
                options=["(new)"] + saved_files,
                key="custom_load_select",
            )
        else:
            selected_saved = "(new)"
            st.caption("No saved strategies yet.")

    # Determine initial code
    if "custom_code" not in st.session_state:
        st.session_state["custom_code"] = STRATEGY_TEMPLATE

    if selected_saved != "(new)" and selected_saved:
        try:
            st.session_state["custom_code"] = load_saved_strategy(selected_saved)
        except Exception as e:
            st.error(f"Could not load {selected_saved}: {e}")

    # ── Code editor ──────────────────────────────────────────────────────────
    code = st.text_area(
        "Strategy Code (Python)",
        value=st.session_state.get("custom_code", STRATEGY_TEMPLATE),
        height=500,
        key="custom_code_editor",
    )

    # ── File upload alternative ──────────────────────────────────────────────
    uploaded_py = st.file_uploader("Or upload a .py file", type=["py"], key="custom_py_upload")
    if uploaded_py is not None:
        code = uploaded_py.read().decode("utf-8")
        st.session_state["custom_code"] = code
        st.success(f"Loaded **{uploaded_py.name}**")

    # ── Save button ──────────────────────────────────────────────────────────
    with col_save:
        save_name = st.text_input("Save as", value="my_strategy.py", key="custom_save_name")
        if st.button("Save Strategy", key="custom_save_btn"):
            try:
                path = save_strategy_to_disk(code, save_name)
                st.success(f"Saved to `{path}`")
            except Exception as e:
                st.error(f"Save failed: {e}")

    st.markdown("---")

    # ── Compile & validate ───────────────────────────────────────────────────
    col_compile, col_run_custom = st.columns(2)

    with col_compile:
        if st.button("Validate Code", type="secondary", key="custom_validate"):
            custom_strat, err = compile_custom_strategy(code)
            if err:
                st.error("Compilation failed:")
                st.code(err, language="text")
            else:
                grid = custom_strat.default_grid()
                combos = 1
                for v in grid.values():
                    combos *= len(v)
                st.success(
                    f"**{custom_strat.name}** compiled successfully!  \n"
                    f"Grid: {combos:,} parameter combos  |  "
                    f"Params: {', '.join(grid.keys())}"
                )

    with col_run_custom:
        run_custom = st.button(
            "Run Walk-Forward with Custom Strategy",
            type="primary",
            key="custom_run_wfa",
        )

    if run_custom:
        # Compile the code
        custom_strat, err = compile_custom_strategy(code)
        if err:
            st.error("Cannot run — code has errors:")
            st.code(err, language="text")
            st.stop()

        # Register it temporarily
        STRATEGY_REGISTRY["_custom_user"] = custom_strat

        # Load data (same logic as main tab)
        df_custom = pd.DataFrame()
        df_custom, _err = load_data_from_source(data_source, uploaded_file, ticker, timeframe)
        if _err:
            st.error(_err)
            st.stop()
        if df_custom.empty:
            st.error("No data available.")
            st.stop()

        st.success(
            f"Loaded **{len(df_custom):,}** bars  |  "
            f"Strategy: **{custom_strat.name}**  |  "
            f"Grid: **{sum(1 for _ in __import__('itertools').product(*custom_strat.default_grid().values())):,}** combos"
        )

        import multiprocessing as mp
        n_cores = max(1, mp.cpu_count() - 1)

        custom_grid = custom_strat.default_grid()
        total_combos_custom = 1
        for v in custom_grid.values():
            total_combos_custom *= len(v)

        progress_bar_c = st.progress(0.0, text="Initialising...")
        status_text_c = st.empty()

        st.info(
            f"**Grid:** {total_combos_custom:,} combos/fold  |  "
            f"**Cores:** {n_cores}  |  "
            f"**Est:** ~{max(1, total_combos_custom * 0.75 / n_cores / 60):.0f} min/fold"
        )

        def progress_cb_c(fold_i, total_folds, phase, pct):
            progress_bar_c.progress(min(pct, 0.99), text=f"Fold {fold_i}/{total_folds} — {phase}")
            status_text_c.markdown(f"**Optimising custom strategy** — Fold {fold_i}/{total_folds}")

        try:
            wfa_custom = run_walk_forward(
                df_custom,
                train_months=train_months,
                test_months=test_months,
                param_grid=custom_grid,
                progress_cb=progress_cb_c,
                n_workers=n_cores,
                strategy_key="_custom_user",
            )
        except Exception as e:
            st.error(f"Walk-forward failed: {e}")
            st.stop()

        progress_bar_c.progress(1.0, text="Complete!")
        status_text_c.empty()

        if not wfa_custom or not wfa_custom.get("folds"):
            st.warning("No folds completed. Try a longer date range or different parameters.")
        else:
            oos_stats_c = wfa_custom["oos_stats"]
            folds_c     = wfa_custom["folds"]
            oos_eq_c    = wfa_custom["oos_equity"]
            is_eq_c     = wfa_custom["is_equity"]

            # KPI row
            st.markdown('<div class="section-header">Custom Strategy — Walk-Forward Results</div>', unsafe_allow_html=True)

            cols_c = st.columns(7)
            metrics_c = [
                ("OOS P&L",       fmt_dollar(oos_stats_c.get("total_pnl", 0)),    color_for(oos_stats_c.get("total_pnl", 0))),
                ("Net Return",    fmt_pct(oos_stats_c.get("net_return_pct", 0)),   color_for(oos_stats_c.get("net_return_pct", 0))),
                ("Sharpe",        f"{oos_stats_c.get('sharpe', 0):.2f}",            color_for(oos_stats_c.get("sharpe", 0))),
                ("Profit Factor", f"{min(oos_stats_c.get('profit_factor', 0), 99):.2f}", color_for(oos_stats_c.get("profit_factor", 0) - 1)),
                ("Win Rate",      f"{oos_stats_c.get('win_rate', 0)*100:.1f}%",     color_for(oos_stats_c.get("win_rate", 0) - 0.5)),
                ("Max DD",        f"{oos_stats_c.get('max_drawdown', 0)*100:.1f}%", "red"),
                ("Trades",        str(oos_stats_c.get("total_trades", 0)),          ""),
            ]
            for col, (label, value, color) in zip(cols_c, metrics_c):
                with col:
                    st.markdown(metric_card(label, value, color), unsafe_allow_html=True)

            # Equity curve
            st.plotly_chart(
                build_equity_fig(oos_eq_c, is_eq_c, initial_capital),
                use_container_width=True,
            )

            # Gantt
            st.plotly_chart(
                build_gantt(wfa_custom["windows"], folds_c),
                use_container_width=True,
            )

            # Fold comparison
            st.plotly_chart(build_fold_comparison(folds_c), use_container_width=True)

            # Drawdown + monthly
            dd_col, mr_col = st.columns(2)
            with dd_col:
                st.plotly_chart(build_drawdown_fig(oos_eq_c), use_container_width=True)
            with mr_col:
                st.plotly_chart(build_monthly_returns(oos_eq_c), use_container_width=True)

            # Trade log
            with st.expander("Custom Strategy OOS Trade Log"):
                trades_df_c = pd.DataFrame(wfa_custom["oos_trades"])
                if not trades_df_c.empty:
                    trades_df_c["pnl"] = trades_df_c["pnl"].round(2)
                    st.dataframe(trades_df_c, use_container_width=True, height=400)

            # Best params per fold
            with st.expander("Best Parameters per Fold"):
                param_rows_c = []
                for f in folds_c:
                    row = {"Fold": f["fold"]}
                    row.update(f.get("best_params", {}))
                    param_rows_c.append(row)
                st.dataframe(pd.DataFrame(param_rows_c).set_index("Fold"), use_container_width=True)


# (Single Backtest tab removed — use Backtrader tab instead)
# Single Backtest tab removed — Backtrader tab replaces it
if False:  # Dead code block, kept for reference
    st.markdown("### Single Backtest — Full Period")
    st.markdown(
        "Run the strategy with fixed default parameters on the full dataset.  \n"
        "Uses the same data source selected in the sidebar (upload or yfinance)."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        sb_risk   = st.number_input("Fixed Risk Points", value=25.0, step=0.5, key="sb_risk")
        sb_tp1    = st.number_input("TP1 R:R", value=2.0, step=0.1, key="sb_tp1")
    with col_b:
        sb_tp2    = st.number_input("TP2 R:R", value=3.5, step=0.1, key="sb_tp2")
        sb_score  = st.number_input("Min Confluence Score", value=5, step=1, key="sb_score")

    if st.button("▶ Run Single Backtest", type="secondary"):
        df_sb = pd.DataFrame()
        with st.spinner("Loading data..."):
            df_sb, _err = load_data_from_source(data_source, uploaded_file, ticker, timeframe)
        if _err:
            st.error(_err)
            st.stop()
        if df_sb.empty:
            st.error("No data.")
        else:
            params_sb = StrategyParams(
                fixed_risk_pts=sb_risk,
                tp1_rr=sb_tp1,
                tp2_rr=sb_tp2,
                min_score=sb_score,
                initial_capital=50000.0,
                exchange_fee_pct=0.0010,
                slippage_pct=0.0005,
            )
            with st.spinner("Running backtest..."):
                res = run_backtest(df_sb, params_sb)

            stats = res["stats"]
            eq    = res["equity"]

            # Metrics
            m_cols = st.columns(6)
            sb_metrics = [
                ("P&L",        fmt_dollar(stats.get("total_pnl", 0)),    color_for(stats.get("total_pnl", 0))),
                ("Sharpe",     f"{stats.get('sharpe', 0):.2f}",           color_for(stats.get("sharpe", 0))),
                ("PF",         f"{min(stats.get('profit_factor', 0), 99):.2f}", color_for(stats.get("profit_factor", 0) - 1)),
                ("Win Rate",   f"{stats.get('win_rate', 0)*100:.1f}%",    color_for(stats.get("win_rate", 0) - 0.5)),
                ("Max DD",     f"{stats.get('max_drawdown', 0)*100:.1f}%", "red"),
                ("Trades",     str(stats.get("total_trades", 0)),         ""),
            ]
            for col, (label, value, clr) in zip(m_cols, sb_metrics):
                with col:
                    st.markdown(metric_card(label, value, clr), unsafe_allow_html=True)

            # Equity chart
            fig_eq = go.Figure(go.Scatter(
                x=eq.index, y=eq.values,
                line=dict(color="#2979ff", width=2),
                fill="tozeroy", fillcolor="rgba(41,121,255,0.08)",
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
            ))
            fig_eq.add_hline(y=50000, line_dash="dash", line_color="#ffffff", opacity=0.3)
            fig_eq.update_layout(
                template="plotly_dark",
                title="Single Backtest Equity Curve",
                yaxis=dict(tickformat="$,.0f"),
                height=420,
                margin=dict(l=60, r=30, t=50, b=40),
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            # Trade log
            trades_df2 = pd.DataFrame(res["trades"])
            if not trades_df2.empty:
                trades_df2["pnl"] = trades_df2["pnl"].round(2)
                st.dataframe(trades_df2, use_container_width=True, height=350)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 4: Backtrader Engine
# ──────────────────────────────────────────────────────────────────────────────

with tab_bt:
    st.markdown("### Backtrader Engine")
    st.markdown(
        "Run **any strategy** through **backtrader's broker simulation** "
        "with proper commission, slippage, and margin modeling."
    )

    # Strategy selector
    bt_strat_list = list_strategies()
    bt_strat_names = {k: name for k, name, _ in bt_strat_list}
    bt_strat_descs = {k: desc for k, _, desc in bt_strat_list}

    bt_strategy_key = st.selectbox(
        "Select Strategy",
        options=[k for k, _, _ in bt_strat_list],
        format_func=lambda k: _format_strategy_name(k, bt_strat_names[k]),
        index=0,
        key="bt_strat_select",
    )
    _render_strat_badge(bt_strategy_key, bt_strat_names[bt_strategy_key])
    st.caption(bt_strat_descs[bt_strategy_key])
    bt_active_strat = get_strategy(bt_strategy_key)

    # Dynamic parameter inputs from selected strategy
    bt_grid_opts = bt_active_strat.sidebar_grid_options()

    # Load saved params if available (from param_sync or optimizer)
    _bt_saved = load_params(bt_strategy_key)
    if _bt_saved:
        st.caption(f"Saved parameters loaded for **{bt_strat_names[bt_strategy_key]}**")

    bt_capital = st.number_input(
        "Capital ($)",
        value=int(_bt_saved.get("initial_capital", 50000)),
        step=1000,
        key="bt_cap",
    )

    bt_user_params = {}

    # Group parameters by prefix for multi-strategy scripts (e.g., orb_ny_, rbr_tk_)
    _all_params = {}
    # Merge grid options with frozen params so ALL params are visible
    _frozen = bt_active_strat.frozen_params()
    for pk, val in _frozen.items():
        if pk not in bt_grid_opts and pk not in ("initial_capital", "point_value", "fee_per_contract",
                                                     "use_max_trade_loss", "max_trade_loss",
                                                     "use_daily_max_loss", "daily_max_loss"):
            _label = pk.replace("_", " ").title()
            if isinstance(val, float):
                _all_params[pk] = {"label": _label, "default": val, "type": "float"}
            elif isinstance(val, int):
                _all_params[pk] = {"label": _label, "default": val, "type": "int"}
            elif isinstance(val, bool):
                _all_params[pk] = {"label": _label, "default": val, "type": "bool"}
    for pk, opts in bt_grid_opts.items():
        grid_default = opts["default"][0] if opts["default"] else opts["options"][0]
        _all_params[pk] = {"label": opts["label"], "default": grid_default,
                           "type": "float" if isinstance(grid_default, float) else "int"}

    # Detect if params have sub-strategy prefixes (e.g., orb_ny_, rbr_tk_)
    _prefix_map = {}
    _no_prefix = {}
    _known_prefixes = {
        "orb_ny_": "ORB NY", "orb_tk_": "ORB Tokyo", "orb_ld_": "ORB London",
        "rbr_ny_": "RBR NY", "rbr_tk_": "RBR Tokyo", "rbr_ld_": "RBR London",
    }
    for pk, info in _all_params.items():
        matched = False
        for prefix, group_name in _known_prefixes.items():
            if pk.startswith(prefix):
                if group_name not in _prefix_map:
                    _prefix_map[group_name] = {}
                _prefix_map[group_name][pk] = info
                matched = True
                break
        if not matched:
            _no_prefix[pk] = info

    def _render_param_inputs(param_dict, key_prefix=""):
        """Render number inputs for a dict of params in 2 columns."""
        cols = st.columns(2)
        keys = list(param_dict.keys())
        for idx, pk in enumerate(keys):
            info = param_dict[pk]
            saved_val = _bt_saved.get(pk)
            if saved_val is not None:
                try:
                    default_val = type(info["default"])(saved_val)
                except (ValueError, TypeError):
                    default_val = info["default"]
            else:
                default_val = info["default"]
            col = cols[idx % 2]
            with col:
                if info["type"] == "bool":
                    bt_user_params[pk] = st.checkbox(
                        info["label"], value=bool(default_val),
                        key=f"bt_p_{bt_strategy_key}_{key_prefix}{pk}")
                elif info["type"] == "float":
                    bt_user_params[pk] = st.number_input(
                        info["label"], value=float(default_val), step=0.25,
                        key=f"bt_p_{bt_strategy_key}_{key_prefix}{pk}")
                else:
                    bt_user_params[pk] = st.number_input(
                        info["label"], value=int(default_val), step=1,
                        key=f"bt_p_{bt_strategy_key}_{key_prefix}{pk}")

    if _prefix_map:
        # Multi-strategy: separate expander per sub-strategy
        if _no_prefix:
            with st.expander("General Parameters", expanded=False):
                _render_param_inputs(_no_prefix, "gen_")
        for group_name, group_params in sorted(_prefix_map.items()):
            with st.expander(f"{group_name} Parameters", expanded=False):
                _render_param_inputs(group_params, f"{group_name}_")
    else:
        # Single strategy: one expander
        with st.expander("Strategy Parameters", expanded=True):
            _render_param_inputs(_all_params)

    # Check if the strategy already exposes risk $ amounts as tunable params
    _strat_has_max_trade = "max_trade_loss" in bt_grid_opts
    _strat_has_daily_max = "daily_max_loss" in bt_grid_opts

    with st.expander("Risk Limits", expanded=True):
        if _strat_has_max_trade or _strat_has_daily_max:
            st.caption("Dollar amounts are set in Strategy Parameters above. Use these toggles to enable/disable.")
        rl_col1, rl_col2 = st.columns(2)
        with rl_col1:
            bt_use_max_trade = st.checkbox(
                "Enable Max Loss Per Trade",
                value=_bt_saved.get("use_max_trade_loss", True),
                key="bt_max_trade_on",
            )
            if not _strat_has_max_trade:
                bt_max_trade_loss = st.number_input(
                    "Max Loss Per Trade ($)",
                    value=float(_bt_saved.get("max_trade_loss", 300.0)),
                    step=50.0,
                    key="bt_max_trade_val",
                )
            else:
                bt_max_trade_loss = 300.0  # fallback, won't be used
        with rl_col2:
            bt_use_daily_max = st.checkbox(
                "Enable Daily Max Loss",
                value=_bt_saved.get("use_daily_max_loss", True),
                key="bt_daily_max_on",
            )
            if not _strat_has_daily_max:
                bt_daily_max_loss = st.number_input(
                    "Daily Max Loss ($)",
                    value=float(_bt_saved.get("daily_max_loss", 1000.0)),
                    step=50.0,
                    key="bt_daily_max_val",
                )
            else:
                bt_daily_max_loss = 1000.0  # fallback, won't be used

    # ── Settings (collapsible) ───────────────────────────────────────────────
    with st.expander("Date Range / Timeframe / Engine", expanded=False):
        st.markdown('<span style="color:#f58f7c;font-weight:600;">Dataset</span>', unsafe_allow_html=True)
        bt_dataset = st.selectbox(
            "Select dataset",
            options=list(BUILTIN_DATASETS.keys()) + ["Upload CSV/XLSX", "yfinance"],
            format_func=lambda k: BUILTIN_DATASETS[k]["label"] if k in BUILTIN_DATASETS else k,
            index=0,
            key="bt_dataset",
        )
        bt_uploaded_file = None
        bt_yf_ticker = "NQ=F"
        if bt_dataset == "Upload CSV/XLSX":
            bt_uploaded_file = st.file_uploader("Upload OHLCV data", type=["csv", "xlsx", "xls"], key="bt_file_upload")
        elif bt_dataset == "yfinance":
            bt_yf_ticker = st.text_input("Ticker (yfinance)", value="NQ=F", key="bt_yf_ticker")
        elif bt_dataset in BUILTIN_DATASETS:
            ds = BUILTIN_DATASETS[bt_dataset]
            st.caption(ds["caption"])

        st.markdown('<span style="color:#f58f7c;font-weight:600;">Date Range</span>', unsafe_allow_html=True)
        _date_range_opt = st.radio(
            "Backtest period",
            options=["Last 7 days", "Last 30 days", "Last 90 days", "Last 365 days", "Last 2 years", "All data", "Custom range"],
            index=3,
            horizontal=True,
            key="bt_date_range",
        )

        _bt_start_date = None
        _bt_end_date = None
        if _date_range_opt == "Custom range":
            _dr_col1, _dr_col2 = st.columns(2)
            with _dr_col1:
                _bt_start_date = st.date_input("Start date", value=date.today() - pd.Timedelta(days=365), key="bt_start_date")
            with _dr_col2:
                _bt_end_date = st.date_input("End date", value=date.today(), key="bt_end_date")
        elif _date_range_opt != "All data":
            _days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90, "Last 365 days": 365, "Last 2 years": 730}
            _bt_start_date = date.today() - pd.Timedelta(days=_days_map[_date_range_opt])
            _bt_end_date = date.today()

        st.markdown('<span style="color:#f58f7c;font-weight:600;">Backtest Timeframe</span>', unsafe_allow_html=True)
        bt_timeframe = st.radio(
            "Run strategy on",
            ["1m", "5m", "15m", "1h", "1d"],
            index=3,
            horizontal=True,
            key="bt_timeframe",
        )
        _bt_resample_map = {
            "1m": None,
            "5m": "5min",
            "15m": "15min",
            "1h": "1h",
            "1d": "1D",
        }
        _bt_resample_rule = _bt_resample_map.get(bt_timeframe)

        st.markdown('<span style="color:#f58f7c;font-weight:600;">Backtest Engine</span>', unsafe_allow_html=True)
        bt_engine = st.radio(
            "Engine",
            ["Backtrader", "Native"],
            index=0,
            horizontal=True,
            key="bt_engine",
        )

    st.markdown("---")

    if st.button("Run Backtrader Backtest", type="primary", key="bt_run"):
        # Load data from the dataset selector in this tab
        df_bt = pd.DataFrame()
        _err = None
        with st.spinner("Loading data..."):
            if bt_dataset in BUILTIN_DATASETS:
                loader_name = BUILTIN_DATASETS[bt_dataset]["loader"]
                df_bt = globals()[loader_name]()
                if df_bt.empty:
                    _err = f"Built-in {bt_dataset} data file not found."
            elif bt_dataset == "Upload CSV/XLSX":
                if bt_uploaded_file is None:
                    _err = "Please upload a CSV/XLSX file."
                else:
                    try:
                        df_bt = load_from_csv(bt_uploaded_file)
                        df_bt = _normalize_df(df_bt)
                    except Exception as e:
                        _err = str(e)
            elif bt_dataset == "yfinance":
                try:
                    df_bt = load_from_yfinance(bt_yf_ticker, interval=bt_timeframe)
                except Exception as e:
                    _err = str(e)
        if _err:
            st.error(_err)
            st.stop()
        if df_bt.empty:
            st.error("No data available.")
            st.stop()

        # Apply date filter
        if _bt_start_date and _bt_end_date:
            _start_ts = pd.Timestamp(_bt_start_date)
            _end_ts = pd.Timestamp(_bt_end_date) + pd.Timedelta(days=1)
            if df_bt.index.tz is not None:
                _start_ts = _start_ts.tz_localize(df_bt.index.tz)
                _end_ts = _end_ts.tz_localize(df_bt.index.tz)
            df_bt = df_bt.loc[_start_ts:_end_ts]

        if df_bt.empty:
            st.error("No data in selected date range.")
            st.stop()

        # Resample if needed
        if _bt_resample_rule is not None:
            _pre_len = len(df_bt)
            df_bt = resample_ohlcv(df_bt, _bt_resample_rule)
            st.caption(f"Resampled {_pre_len:,} bars → {len(df_bt):,} bars ({bt_timeframe})")

        # Build params: strategy params take priority, Risk Limits expander is fallback
        bt_params = {
            "initial_capital": float(bt_capital),
            "point_value": 2.0,
            "use_max_trade_loss": bt_use_max_trade,
            "use_daily_max_loss": bt_use_daily_max,
        }
        # Only use Risk Limits expander values if the strategy doesn't expose them as params
        if "max_trade_loss" not in bt_user_params:
            bt_params["max_trade_loss"] = bt_max_trade_loss
        if "daily_max_loss" not in bt_user_params:
            bt_params["daily_max_loss"] = bt_daily_max_loss
        # Strategy params override everything above
        bt_params.update(bt_user_params)

        _bt_engine_label = "native engine" if "Native" in bt_engine else "backtrader"
        with st.spinner(f"Running **{bt_strat_names[bt_strategy_key]}** through {_bt_engine_label}..."):
            try:
                if "Native" in bt_engine:
                    bt_result = run_bt_generic_fast(df_bt, bt_strategy_key, bt_params)
                else:
                    bt_result = run_bt_generic(df_bt, bt_strategy_key, bt_params)
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.stop()

        # Auto-save params on every backtest run
        save_params(bt_strategy_key, bt_params)

        # Store results in session_state so they persist
        st.session_state["bt_result"] = {
            "result": bt_result,
            "capital": float(bt_capital),
            "strategy_key": bt_strategy_key,
            "strategy_name": bt_strat_names[bt_strategy_key],
            "date_range": f"{_bt_start_date} to {_bt_end_date}",
            "df": df_bt,  # store raw data for chart rendering
            "params": bt_params,
        }

    # ── Display results (from session_state so they persist) ──────────────
    if "bt_result" in st.session_state:
        _btr = st.session_state["bt_result"]
        bt_result = _btr["result"]
        _bt_cap = _btr["capital"]
        bt_stats = bt_result["stats"]
        bt_eq = bt_result["equity"]

        st.success(
            f"**{_btr['strategy_name']}** — {_btr['date_range']}  |  "
            f"**{bt_stats['total_trades']}** trades  |  "
            f"Final: **${bt_stats['final_value']:,.0f}**"
        )

        # Metrics
        bt_cols = st.columns(7)
        _dd_pct = bt_stats.get("max_drawdown", 0) * 100
        _dd_dollars = abs(bt_stats.get("max_drawdown", 0)) * _bt_cap
        bt_metrics = [
            ("P&L",     fmt_dollar(bt_stats.get("total_pnl", 0)),    color_for(bt_stats.get("total_pnl", 0))),
            ("Return",  fmt_pct(bt_stats.get("net_return_pct", 0)),   color_for(bt_stats.get("net_return_pct", 0))),
            ("Sharpe",  f"{bt_stats.get('sharpe', 0):.2f}",           color_for(bt_stats.get("sharpe", 0))),
            ("PF",      f"{min(bt_stats.get('profit_factor', 0), 99):.2f}", color_for(bt_stats.get("profit_factor", 0) - 1)),
            ("Win Rate", f"{bt_stats.get('win_rate', 0)*100:.1f}%",   color_for(bt_stats.get("win_rate", 0) - 0.5)),
            ("Max DD",  "special_dd", "red"),
            ("Trades",  str(bt_stats.get("total_trades", 0)),         ""),
        ]
        for col, (label, value, clr) in zip(bt_cols, bt_metrics):
            with col:
                if value == "special_dd":
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="label">MAX DD</div>'
                        f'<div class="value red">${_dd_dollars:,.0f}</div>'
                        f'<div style="color:#8a86a6;font-size:0.75rem;margin-top:2px;">{_dd_pct:.1f}%</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(metric_card(label, value, clr), unsafe_allow_html=True)

        # ── Sub-tabs: Chart | Equity Curve | Trade List ──────────────────
        bt_sub_chart, bt_sub_eq, bt_sub_trades = st.tabs(["Chart", "Equity Curve", "Trade List"])

        with bt_sub_chart:
            from trade_chart import render_trade_chart
            _chart_df = st.session_state["bt_result"].get("df", pd.DataFrame())
            _chart_trades = bt_result.get("trades", [])
            _chart_key = _btr["strategy_key"]
            _chart_params = st.session_state["bt_result"].get("params", {})
            if not _chart_df.empty and _chart_trades:
                render_trade_chart(_chart_df, _chart_trades, _chart_key, _chart_params)
            else:
                st.info("No chart data available. Run a backtest first.")

        with bt_sub_eq:
            # Equity curve
            if not bt_eq.empty:
                fig_bt_eq = go.Figure(go.Scatter(
                    x=bt_eq.index, y=bt_eq.values,
                    line=dict(color="#2979ff", width=2),
                    fill="tozeroy", fillcolor="rgba(41,121,255,0.08)",
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:,.0f}<extra></extra>",
                ))
                fig_bt_eq.add_hline(y=_bt_cap, line_dash="dash", line_color="#ffffff", opacity=0.3)
                fig_bt_eq.update_layout(
                    template="plotly_dark",
                    title="Backtrader Equity Curve",
                    yaxis=dict(tickformat="$,.0f"),
                    height=420,
                    margin=dict(l=60, r=30, t=50, b=40),
                )
                st.plotly_chart(fig_bt_eq, use_container_width=True)

                # Drawdown
                st.plotly_chart(build_drawdown_fig(bt_eq), use_container_width=True)
            else:
                st.info("No equity data available.")

            # Analyzer details
            with st.expander("Backtrader Analyzer Details"):
                analyzers = bt_result.get("analyzers", {})
                a_col1, a_col2 = st.columns(2)
                with a_col1:
                    st.markdown("**Drawdown**")
                    dd = analyzers.get("drawdown", {})
                    st.json({
                        "max_drawdown_pct": dd.get("max", {}).get("drawdown", 0),
                        "max_drawdown_dollars": dd.get("max", {}).get("moneydown", 0),
                        "max_drawdown_len_bars": dd.get("max", {}).get("len", 0),
                    })
                with a_col2:
                    st.markdown("**Trade Summary**")
                    ta = analyzers.get("trades", {})
                    st.json({
                        "total_closed": ta.get("total", {}).get("closed", 0),
                        "won": ta.get("won", {}).get("total", 0),
                        "lost": ta.get("lost", {}).get("total", 0),
                        "avg_win": ta.get("won", {}).get("pnl", {}).get("average", 0),
                        "avg_loss": ta.get("lost", {}).get("pnl", {}).get("average", 0),
                        "longest_win_streak": ta.get("streak", {}).get("won", {}).get("longest", 0),
                        "longest_loss_streak": ta.get("streak", {}).get("lost", {}).get("longest", 0),
                    })

        with bt_sub_trades:
            # ── Styled Trade List ──────────────────────────────────────────
            bt_trades_df = pd.DataFrame()
            if bt_result.get("trades"):
                bt_trades_df = pd.DataFrame(bt_result["trades"])

                # Build a clean display table
                display_df = pd.DataFrame()
                display_df["Date"] = pd.to_datetime(bt_trades_df["entry_time"]).dt.strftime("%Y-%m-%d")
                display_df["Entry Time"] = pd.to_datetime(bt_trades_df["entry_time"]).dt.strftime("%H:%M")
                display_df["Exit Time"] = pd.to_datetime(bt_trades_df["exit_time"]).dt.strftime("%H:%M")
                display_df["Signal"] = bt_trades_df["direction"].map({1: "LONG", -1: "SHORT"})
                display_df["Entry Price"] = bt_trades_df["entry_price"].round(2)
                display_df["Exit Price"] = bt_trades_df["exit_price"].round(2)
                display_df["Contracts"] = bt_trades_df["contracts"]
                display_df["Entry Label"] = bt_trades_df.get("entry_type", "—")
                display_df["Exit Label"] = bt_trades_df.get("exit_reason", "—")
                display_df["P&L"] = bt_trades_df["pnl"].round(2)

                # Summary — pull won/lost directly from backtrader analyzer for exact match
                _analyzers = bt_result.get("analyzers", {})
                _ta = _analyzers.get("trades", {})
                _bt_total = _ta.get("total", {}).get("closed", bt_stats.get("total_trades", 0))
                _bt_wins = _ta.get("won", {}).get("total", 0)
                _bt_losses = _ta.get("lost", {}).get("total", 0)
                _bt_pnl = bt_stats.get("total_pnl", 0)
                _bt_wr = _bt_wins / _bt_total if _bt_total > 0 else 0.0

                _sum_col1, _sum_col2, _sum_col3, _sum_col4 = st.columns(4)
                with _sum_col1:
                    st.metric("Total Trades", _bt_total)
                with _sum_col2:
                    st.metric("Winners", f"{_bt_wins} ({_bt_wr*100:.1f}%)" if _bt_total else "0")
                with _sum_col3:
                    st.metric("Losers", f"{_bt_losses} ({(1-_bt_wr)*100:.1f}%)" if _bt_total else "0")
                with _sum_col4:
                    st.metric("Net P&L", f"${_bt_pnl:,.2f}")

                st.caption(f"Table shows {len(display_df)} individual exits (partial fills from multi-contract entries are listed separately).")
                st.markdown("")

                # Style P&L column: green for positive, red for negative
                def _style_pnl(val):
                    if val > 0:
                        return "color: #00e676; font-weight: bold"
                    elif val < 0:
                        return "color: #ff5252; font-weight: bold"
                    return "color: #888"

                def _style_signal(val):
                    if val == "LONG":
                        return "color: #00e676"
                    elif val == "SHORT":
                        return "color: #ff5252"
                    return ""

                styled = display_df.style.map(
                    _style_pnl, subset=["P&L"]
                ).map(
                    _style_signal, subset=["Signal"]
                ).format({
                    "Entry Price": "{:.2f}",
                    "Exit Price": "{:.2f}",
                    "P&L": "${:,.2f}",
                })

                st.dataframe(styled, use_container_width=True, height=500)

                # Download
                st.download_button(
                    "Download Trade List (CSV)",
                    data=display_df.to_csv(index=False),
                    file_name=f"{_btr['strategy_key']}_trade_list.csv",
                    mime="text/csv",
                    key="bt_dl_trade_list",
                )
            else:
                st.info("No trades recorded. Run a backtest first.")

        # ── Download buttons for Backtrader results ─────────────────────────
        st.markdown("#### Downloads")
        bt_dl_col1, bt_dl_col2 = st.columns(2)
        with bt_dl_col1:
            if not bt_eq.empty:
                eq_csv_df = pd.DataFrame({"date": bt_eq.index, "equity": bt_eq.values})
                st.download_button(
                    "Download Equity Curve (CSV)",
                    data=eq_csv_df.to_csv(index=False),
                    file_name=f"{_btr['strategy_key']}_equity_curve.csv",
                    mime="text/csv",
                    key="bt_dl_equity",
                )
            else:
                st.caption("No equity data to download.")
        with bt_dl_col2:
            if not bt_trades_df.empty:
                st.download_button(
                    "Download Full Trade Log (CSV)",
                    data=bt_trades_df.to_csv(index=False),
                    file_name=f"{_btr['strategy_key']}_trade_log.csv",
                    mime="text/csv",
                    key="bt_dl_trades",
                )
            else:
                st.caption("No trade data to download.")

    # ── Strategy Code Viewer & Save/Update ────────────────────────────────────
    st.markdown("---")
    st.markdown("### Strategy Code Viewer")
    st.caption("View and copy the source code for any strategy. Modify defaults by saving optimized parameters back into the code.")

    # Map strategy keys to their source files
    _STRAT_FILE_MAP = {
        "combined_all_1m":  "strat_combined_all_1m.py",
        "orb_ny_1m":        "strat_orb_ny_1m.py",
        "orb_tokyo_1m":     "strat_orb_tokyo_1m.py",
        "orb_london_1m":    "strat_orb_london_1m.py",
        "rbr_ny_1m":        "strat_rbr_ny_1m.py",
        "rbr_tokyo_1m":     "strat_rbr_tokyo_1m.py",
        "rbr_london_1m":    "strat_rbr_london_1m.py",
        "precision_sniper_1h": "strat_precision_sniper_1h.py",
        "orb_1h":           "strat_orb_1h.py",
        "tokyo_orb_1h":     "strat_tokyo_orb_1h.py",
        "london_orb_1h":    "strat_london_orb_1h.py",
        "ny_rbr_1h":        "strat_ny_rbr_1h.py",
        "tokyo_rbr_1h":     "strat_tokyo_rbr_1h.py",
        "london_rbr_1h":    "strat_london_rbr_1h.py",
        "butterworth_atr":  "strat_butterworth_atr.py",
        "bw_atr_optimized": "strat_bw_atr_optimized.py",
        "butterworth_atr_1h": "strat_butterworth_atr.py",
        "precision_sniper": None,
        "ema_crossover":    None,
        "rsi_reversion":    None,
        "macd_supertrend":  None,
    }

    # Also map to PineScript files if they exist
    _PINE_FILE_MAP = {
        "combined_all_1m":     os.path.expanduser("~/MNQ_Combined_All_Sessions_1min.pine"),
        "precision_sniper_1h": os.path.expanduser("~/MNQ_Precision_Sniper_1HR.pine"),
        "orb_1h":           os.path.expanduser("~/MNQ_ORB_1HR.pine"),
        "tokyo_orb_1h":     os.path.expanduser("~/MNQ_Tokyo_ORB_1HR.pine"),
        "london_orb_1h":    os.path.expanduser("~/MNQ_London_ORB_1HR.pine"),
        "ny_rbr_1h":        os.path.expanduser("~/MNQ_NY_RBR_1HR.pine"),
        "tokyo_rbr_1h":     os.path.expanduser("~/MNQ_Tokyo_RBR_1HR.pine"),
        "london_rbr_1h":    os.path.expanduser("~/MNQ_London_RBR_1HR.pine"),
    }

    code_view_key = bt_strategy_key  # use the same strategy selected above

    _py_file = _STRAT_FILE_MAP.get(code_view_key)
    _pine_path = _PINE_FILE_MAP.get(code_view_key)

    # Python source — collapsible expander
    _py_label = f"Python Source — {_py_file}" if _py_file else f"Python Source — {bt_strat_names[code_view_key]}"
    with st.expander(_py_label, expanded=False):
        if _py_file:
            _py_path = os.path.join(os.path.dirname(__file__), _py_file)
            if os.path.exists(_py_path):
                with open(_py_path, "r") as f:
                    _py_code = f.read()
                st.code(_py_code, language="python", line_numbers=True)
                st.download_button(
                    f"Download {_py_file}",
                    data=_py_code,
                    file_name=_py_file,
                    mime="text/x-python",
                    key="bt_dl_py_code",
                )
            else:
                st.info(f"Source file `{_py_file}` not found.")
        else:
            st.info(
                f"**{bt_strat_names[code_view_key]}** is defined inline in "
                f"`strategy_registry.py` (no separate strat file)."
            )

    # PineScript source — collapsible expander
    _pine_label = f"PineScript Source — {os.path.basename(_pine_path)}" if _pine_path and os.path.exists(_pine_path) else f"PineScript Source — {bt_strat_names[code_view_key]}"
    with st.expander(_pine_label, expanded=False):
        if _pine_path and os.path.exists(_pine_path):
            with open(_pine_path, "r") as f:
                _pine_code = f.read()
            st.code(_pine_code, language="javascript", line_numbers=True)
            st.download_button(
                f"Download {os.path.basename(_pine_path)}",
                data=_pine_code,
                file_name=os.path.basename(_pine_path),
                mime="text/plain",
                key="bt_dl_pine_code",
            )
        else:
            st.info(f"No PineScript source available for **{bt_strat_names[code_view_key]}**.")

    # ── Save & Update Strategy Defaults ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### Save & Update Strategy Defaults")
    st.caption(
        "Write the current parameter values back into the strategy's Python source file "
        "as new defaults. This updates the code so future runs start with these values."
    )

    _updatable_file = _STRAT_FILE_MAP.get(bt_strategy_key)
    if _updatable_file:
        _update_path = os.path.join(os.path.dirname(__file__), _updatable_file)

        # Show current params that would be saved
        _current_bt_params = {**bt_user_params}
        with st.expander("Parameters to save as new defaults"):
            st.json(_current_bt_params)

        _upd_col1, _upd_col2 = st.columns(2)
        with _upd_col1:
            if st.button("Save & Update Code Defaults", type="primary", key="bt_update_code"):
                if os.path.exists(_update_path):
                    try:
                        with open(_update_path, "r") as f:
                            code_content = f.read()

                        changes_made = 0
                        for pname, pval in _current_bt_params.items():
                            # Match patterns like: params.get("param_name", DEFAULT_VALUE)
                            # Handles both float and int defaults
                            pattern = rf'(params\.get\(\s*"{pname}"\s*,\s*)([^)]+)(\s*\))'
                            if re.search(pattern, code_content):
                                if isinstance(pval, float):
                                    replacement = rf'\g<1>{pval}\3'
                                else:
                                    replacement = rf'\g<1>{pval}\3'
                                code_content = re.sub(pattern, replacement, code_content)
                                changes_made += 1

                        with open(_update_path, "w") as f:
                            f.write(code_content)

                        # Also save to param_sync
                        save_params(bt_strategy_key, {**_current_bt_params, "initial_capital": float(bt_capital)})

                        st.success(
                            f"Updated **{changes_made}** default values in `{_updatable_file}` "
                            f"and saved to `saved_params/{bt_strategy_key}.json`."
                        )
                    except Exception as e:
                        st.error(f"Failed to update: {e}")
                else:
                    st.error(f"File `{_updatable_file}` not found.")

        with _upd_col2:
            if st.button("Save Params Only (no code change)", key="bt_save_only"):
                save_params(bt_strategy_key, {**_current_bt_params, "initial_capital": float(bt_capital)})
                st.success(f"Saved to `saved_params/{bt_strategy_key}.json`")
    else:
        st.info(
            f"**{bt_strat_names[bt_strategy_key]}** doesn't have a separate strategy file to update. "
            f"Use 'Save Params Only' to persist your settings."
        )
        if st.button("Save Params Only", key="bt_save_only_inline"):
            save_params(bt_strategy_key, {**bt_user_params, "initial_capital": float(bt_capital)})
            st.success(f"Saved to `saved_params/{bt_strategy_key}.json`")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 5: About
# ──────────────────────────────────────────────────────────────────────────────

with tab_about:
    st.markdown("""
### Precision Trader — How It Works

Precision Trader is a strategy testing platform that lets you backtest, optimize,
and validate any trading strategy on futures data. Here's what each tab does.

---

#### Walk-Forward Analysis

Prevents curve-fitting by splitting your data into rolling windows:

1. **Training window** — the optimizer tests every parameter combination and picks the best
2. **Blind test window** — the winning parameters run on unseen data
3. **Stitch** — only the blind-test (out-of-sample) results are kept

You see what the strategy *actually* would have done on data it never trained on.

| Setting | What it controls |
|---------|-----------------|
| Training months | How much history the optimizer learns from |
| Test months | How much unseen data it's tested on |
| Parameter grid | The values to search through (sidebar) |

---

#### Custom Strategy

Paste or upload your own Python strategy code. The app compiles it, validates it,
and plugs it into the walk-forward engine. A working template is provided.

Your code defines 3 things:
- `STRATEGY_NAME` — what to call it
- `PARAM_GRID` — the parameters to optimize
- `run_backtest(df, params)` — your entry/exit logic

---

#### Single Backtest

Runs the selected strategy once with fixed parameters on the full dataset.
Good for quick checks before running the full walk-forward.

---

#### Backtrader Engine

Runs any strategy through **backtrader's broker simulation** for more realistic results.
Backtrader models:
- Proper futures margin ($1,000/contract for MNQ)
- Commission ($0.62/contract/side)
- Order fill on next bar (not same bar)
- Built-in analyzers: Sharpe ratio, drawdown, trade statistics

---

#### Data Sources

| Source | Timeframes | History |
|--------|-----------|---------|
| **Upload CSV** | Any | Unlimited (your own export) |
| **yfinance 15m** | 15-minute | Last 58 days |
| **yfinance 1h** | 1-hour | Last 2 years |
| **yfinance 1d** | Daily | Last 10 years |

---

#### Built-in Strategies

Select from the sidebar dropdown. Each strategy has its own parameter grid
that the optimizer searches through.

| Strategy | Description |
|----------|-------------|
| BW-ATR Optimized | Butterworth filter + ATR vol-momentum, prop firm safe |
| BW-ATR (15m) | Same math, tuned for 15-minute bars |
| BW-ATR (1H) | Same math, tuned for 1-hour bars |
| Precision Sniper v7.5 | EMA crossover + confluence scoring |
| EMA Crossover | Classic fast/slow EMA cross |
| RSI Mean Reversion | Buy oversold, sell overbought with trend filter |
| MACD + Supertrend | MACD histogram cross filtered by Supertrend |
| Precision Sniper 1H | EMA crossover + pullback + 10-point confluence, prop-firm safe |
| ORB 1H (NY) | NY session first-hour range breakout, 3 contracts |
| Tokyo ORB 1H | Tokyo session first-hour range breakout, 6 contracts |
| London ORB 1H | London session first-hour range breakout, 3 contracts |
| NY RBR 1H | EMA 9/21 + Rally-Base-Rally, NY session, 2 contracts |
| Tokyo RBR 1H | EMA 9/21 + Rally-Base-Rally, Tokyo session, 2 contracts |
| London RBR 1H | EMA 9/21 + Rally-Base-Rally, London session, 6 contracts |

---

#### Saving Parameters

- **Save Current Params** — saves to disk so other scripts can load them
- **Export as JSON** — download to your computer
- **Auto-save** — Backtrader tab saves params on every run
- **Save & Update Code Defaults** — writes optimized values back into the strategy source file
- **Load in Python:**
```python
from param_sync import load_params
params = load_params("bw_atr_optimized")
```

---

#### Strategy Code Viewer

In the Backtrader tab, scroll down to view:
- **Python source** — the full `strat_*.py` file with copy/download
- **PineScript source** — the matching `.pine` file (for ORB and RBR strategies)

---

#### Risk Limits

Available in the Backtrader tab:
- **Max Loss Per Trade** — emergency exit if one trade loses more than $X
- **Daily Max Loss** — shuts down trading for the day after $X in losses

---

#### Cost Model

| Component | Rate |
|-----------|------|
| Exchange fee | $0.62/contract/side (backtrader) or 0.10% (custom engine) |
| Slippage | Built into broker sim (backtrader) or 0.05% (custom engine) |

---

*Built for testing. Not financial advice. Past performance does not guarantee future results.*
""")
