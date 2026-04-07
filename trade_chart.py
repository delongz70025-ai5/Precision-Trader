"""
Trade Chart — Interactive candlestick chart with trade markers and strategy indicators.
Shows recent price action with buy/sell signals and strategy-specific overlays.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def render_trade_chart(
    df: pd.DataFrame,
    trades: list,
    strategy_key: str,
    params: dict,
):
    """Render an interactive candlestick chart with trade markers and indicators."""

    if df.empty:
        st.info("No data to chart.")
        return

    # Determine how much data to show based on bar frequency
    _freq_mins = _estimate_bar_freq_mins(df)
    if _freq_mins <= 1:
        default_bars = 400   # ~6-7 hours of 1-min — enough to see candles clearly
        max_bars = 2000
    elif _freq_mins <= 5:
        default_bars = 300
        max_bars = 1500
    elif _freq_mins <= 15:
        default_bars = 250
        max_bars = 800
    elif _freq_mins <= 60:
        default_bars = 200
        max_bars = 600
    else:
        default_bars = 120
        max_bars = 300

    n_bars = st.slider(
        "Bars to show", min_value=50, max_value=min(max_bars, len(df)),
        value=min(default_bars, len(df)), step=50, key="chart_bars",
    )

    # Slice to recent data
    chart_df = df.iloc[-n_bars:].copy()

    # Build the figure
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.8, 0.2],
    )

    # ── Candlestick ──────────────────────────────────────────────────────────
    # Use OHLC bars rendered as shapes for crisp wicks at any zoom level
    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df["open"],
        high=chart_df["high"],
        low=chart_df["low"],
        close=chart_df["close"],
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",
        decreasing_fillcolor="#ef5350",
        increasing_line_width=1.5,
        decreasing_line_width=1.5,
        whiskerwidth=0.8,
        name="Price",
        showlegend=False,
    ), row=1, col=1)

    # ── Volume bars ──────────────────────────────────────────────────────────
    if "volume" in chart_df.columns:
        vol_colors = ["#26a69a" if c >= o else "#ef5350"
                      for c, o in zip(chart_df["close"], chart_df["open"])]
        fig.add_trace(go.Bar(
            x=chart_df.index,
            y=chart_df["volume"],
            marker_color=vol_colors,
            opacity=0.4,
            name="Volume",
            showlegend=False,
        ), row=2, col=1)

    # ── Strategy-specific indicators ─────────────────────────────────────────
    _add_strategy_indicators(fig, chart_df, strategy_key, params)

    # ── Trade markers ────────────────────────────────────────────────────────
    _add_trade_markers(fig, chart_df, trades)

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        height=700,
        margin=dict(l=50, r=60, t=20, b=30),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16162a",
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="top", y=1.02, xanchor="left", x=0,
            font=dict(size=10, color="#c0bdd6"),
            bgcolor="rgba(0,0,0,0)",
        ),
        yaxis=dict(
            gridcolor="#2a2945", tickformat=".2f",
            title=None, side="right",
        ),
        yaxis2=dict(
            gridcolor="#2a2945", title=None, side="right",
        ),
        xaxis=dict(gridcolor="#2a2945"),
        xaxis2=dict(gridcolor="#2a2945"),
    )

    # Remove weekend gaps for futures (continuous trading)
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])],
    )

    st.plotly_chart(fig, use_container_width=True)


def _estimate_bar_freq_mins(df: pd.DataFrame) -> float:
    """Estimate the bar frequency in minutes from the index."""
    if len(df) < 3:
        return 60
    diffs = pd.Series(df.index).diff().dropna().dt.total_seconds() / 60
    return float(diffs.median())


def _add_trade_markers(fig, chart_df, trades):
    """Add buy/sell markers at entry/exit points."""
    if not trades:
        return

    chart_start = chart_df.index[0]
    chart_end = chart_df.index[-1]

    # Collect entries and exits within the chart window
    buy_times, buy_prices, buy_labels = [], [], []
    sell_times, sell_prices, sell_labels = [], [], []
    exit_times, exit_prices, exit_labels, exit_colors = [], [], [], []

    for t in trades:
        entry_t = t.get("entry_time")
        exit_t = t.get("exit_time")
        direction = t.get("direction", 0)
        entry_px = t.get("entry_price", 0)
        exit_px = t.get("exit_price", 0)
        reason = t.get("exit_reason", "")
        pnl = t.get("pnl", 0)

        # Entry markers
        if entry_t is not None and chart_start <= entry_t <= chart_end:
            if direction == 1:
                buy_times.append(entry_t)
                buy_prices.append(entry_px)
                buy_labels.append(f"LONG @ {entry_px:.2f}")
            elif direction == -1:
                sell_times.append(entry_t)
                sell_prices.append(entry_px)
                sell_labels.append(f"SHORT @ {entry_px:.2f}")

        # Exit markers
        if exit_t is not None and chart_start <= exit_t <= chart_end:
            exit_times.append(exit_t)
            exit_prices.append(exit_px)
            _pnl_str = f"+${pnl:.0f}" if pnl >= 0 else f"-${abs(pnl):.0f}"
            exit_labels.append(f"{reason} @ {exit_px:.2f} ({_pnl_str})")
            exit_colors.append("#8df688" if pnl >= 0 else "#f6506a")

    # Buy markers (green triangles up)
    if buy_times:
        fig.add_trace(go.Scatter(
            x=buy_times, y=buy_prices, mode="markers",
            marker=dict(symbol="triangle-up", size=14, color="#8df688",
                        line=dict(width=1, color="#fff")),
            name="Long Entry",
            text=buy_labels,
            hovertemplate="%{text}<extra></extra>",
        ), row=1, col=1)

    # Sell markers (red triangles down)
    if sell_times:
        fig.add_trace(go.Scatter(
            x=sell_times, y=sell_prices, mode="markers",
            marker=dict(symbol="triangle-down", size=14, color="#f6506a",
                        line=dict(width=1, color="#fff")),
            name="Short Entry",
            text=sell_labels,
            hovertemplate="%{text}<extra></extra>",
        ), row=1, col=1)

    # Exit markers (diamonds)
    if exit_times:
        fig.add_trace(go.Scatter(
            x=exit_times, y=exit_prices, mode="markers",
            marker=dict(symbol="diamond", size=9, color=exit_colors,
                        line=dict(width=1, color="#fff")),
            name="Exit",
            text=exit_labels,
            hovertemplate="%{text}<extra></extra>",
        ), row=1, col=1)


def _add_strategy_indicators(fig, chart_df, strategy_key, params):
    """Add strategy-specific indicator overlays based on strategy type."""

    close = chart_df["close"]
    high = chart_df["high"]
    low = chart_df["low"]
    idx = chart_df.index

    # Detect strategy type from the key
    key_lower = strategy_key.lower()

    # ── ORB strategies: show range high/low lines ────────────────────────────
    if "orb" in key_lower:
        _add_orb_indicators(fig, chart_df, strategy_key, params)

    # ── RBR strategies: show EMA lines ───────────────────────────────────────
    elif "rbr" in key_lower:
        fast_len = int(params.get("fast_len", 8))
        slow_len = int(params.get("slow_len", 20))
        ema_fast = close.ewm(span=fast_len, adjust=False).mean()
        ema_slow = close.ewm(span=slow_len, adjust=False).mean()

        fig.add_trace(go.Scatter(
            x=idx, y=ema_fast, mode="lines",
            line=dict(color="#8df688", width=1.5, dash="solid"),
            name=f"EMA {fast_len}",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=idx, y=ema_slow, mode="lines",
            line=dict(color="#f58f7c", width=1.5, dash="solid"),
            name=f"EMA {slow_len}",
        ), row=1, col=1)

    # ── Precision Sniper: show EMAs + supertrend ─────────────────────────────
    elif "precision" in key_lower or "sniper" in key_lower:
        fast_len = int(params.get("ema_fast_len", 9))
        slow_len = int(params.get("ema_slow_len", 21))
        trend_len = int(params.get("ema_trend_len", 50))

        ema_fast = close.ewm(span=fast_len, adjust=False).mean()
        ema_slow = close.ewm(span=slow_len, adjust=False).mean()
        ema_trend = close.ewm(span=trend_len, adjust=False).mean()

        fig.add_trace(go.Scatter(
            x=idx, y=ema_fast, mode="lines",
            line=dict(color="#8df688", width=1.5), name=f"EMA {fast_len}",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=idx, y=ema_slow, mode="lines",
            line=dict(color="#f58f7c", width=1.5), name=f"EMA {slow_len}",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=idx, y=ema_trend, mode="lines",
            line=dict(color="#7a7793", width=1, dash="dash"), name=f"EMA {trend_len}",
        ), row=1, col=1)

    # ── BW-ATR / EMA Crossover / MACD: show EMAs ────────────────────────────
    elif "ema" in key_lower or "bw" in key_lower or "butterworth" in key_lower:
        fast_len = int(params.get("ema_fast_len", params.get("fast_len", 9)))
        slow_len = int(params.get("ema_slow_len", params.get("slow_len", 21)))
        ema_fast = close.ewm(span=fast_len, adjust=False).mean()
        ema_slow = close.ewm(span=slow_len, adjust=False).mean()

        fig.add_trace(go.Scatter(
            x=idx, y=ema_fast, mode="lines",
            line=dict(color="#8df688", width=1.5), name=f"EMA {fast_len}",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=idx, y=ema_slow, mode="lines",
            line=dict(color="#f58f7c", width=1.5), name=f"EMA {slow_len}",
        ), row=1, col=1)

    # ── RSI / MACD: show basic EMAs ──────────────────────────────────────────
    elif "rsi" in key_lower or "macd" in key_lower:
        ema_fast = close.ewm(span=9, adjust=False).mean()
        ema_slow = close.ewm(span=21, adjust=False).mean()
        fig.add_trace(go.Scatter(
            x=idx, y=ema_fast, mode="lines",
            line=dict(color="#8df688", width=1), name="EMA 9",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=idx, y=ema_slow, mode="lines",
            line=dict(color="#f58f7c", width=1), name="EMA 21",
        ), row=1, col=1)


def _add_orb_indicators(fig, chart_df, strategy_key, params):
    """Add opening range high/low lines for ORB strategies."""
    import pytz

    key_lower = strategy_key.lower()

    # Determine timezone and opening range window
    if "tokyo" in key_lower:
        tz = pytz.timezone("Asia/Tokyo")
        or_hour = 9
        or_min_start = 0
        or_min_end = 15
    elif "london" in key_lower:
        tz = pytz.timezone("Europe/London")
        or_hour = 8
        or_min_start = 0
        or_min_end = 15
    else:
        # NY default
        tz = pytz.timezone("America/New_York")
        or_hour = 9
        or_min_start = 30
        or_min_end = 45

    # Convert index to local timezone — pre-compute as numpy arrays
    try:
        local_idx = chart_df.index.tz_convert(tz)
    except TypeError:
        try:
            local_idx = chart_df.index.tz_localize("UTC").tz_convert(tz)
        except Exception:
            return

    hours_arr = np.array(local_idx.hour)
    mins_arr = np.array(local_idx.minute)
    dates_arr = np.array(local_idx.date)
    highs = chart_df["high"].values
    lows = chart_df["low"].values
    orig_idx = chart_df.index  # original index for plotting

    # Build opening range mask
    or_mask = (hours_arr == or_hour) & (mins_arr >= or_min_start) & (mins_arr < or_min_end)

    # Find unique local dates
    unique_dates = sorted(set(dates_arr))

    range_count = 0
    for d in unique_dates:
        # Bars in the opening range for this day
        day_or_mask = or_mask & (dates_arr == d)
        or_indices = np.where(day_or_mask)[0]

        if len(or_indices) == 0:
            continue

        range_high = float(highs[or_indices].max())
        range_low = float(lows[or_indices].min())

        # Skip if range is zero or invalid
        if range_high <= range_low:
            continue

        # Find all bars for this local date (for line extent)
        day_indices = np.where(dates_arr == d)[0]
        if len(day_indices) == 0:
            continue

        # Lines start after the opening range ends, not from the beginning of the day
        or_end_idx = or_indices[-1]
        line_start = orig_idx[or_end_idx]
        line_end = orig_idx[day_indices[-1]]

        # Range high line
        fig.add_trace(go.Scatter(
            x=[line_start, line_end], y=[range_high, range_high],
            mode="lines", line=dict(color="#8df688", width=2, dash="dash"),
            showlegend=False,
            hovertemplate=f"Range High: {range_high:.2f}<extra></extra>",
        ), row=1, col=1)

        # Range low line
        fig.add_trace(go.Scatter(
            x=[line_start, line_end], y=[range_low, range_low],
            mode="lines", line=dict(color="#f6506a", width=2, dash="dash"),
            showlegend=False,
            hovertemplate=f"Range Low: {range_low:.2f}<extra></extra>",
        ), row=1, col=1)

        # Light shaded opening range period
        or_start = orig_idx[or_indices[0]]
        or_end = orig_idx[or_indices[-1]]
        fig.add_shape(
            type="rect",
            x0=or_start, x1=or_end, y0=range_low, y1=range_high,
            fillcolor="rgba(138, 134, 166, 0.1)",
            line=dict(width=1, color="rgba(138, 134, 166, 0.3)"),
            row=1, col=1,
        )

        range_count += 1
        # Limit to avoid too many traces (last 10 days)
        if range_count >= 10:
            break

    # Add legend entries once
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color="#8df688", width=2, dash="dash"),
        name="Range High",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color="#f6506a", width=2, dash="dash"),
        name="Range Low",
    ), row=1, col=1)
