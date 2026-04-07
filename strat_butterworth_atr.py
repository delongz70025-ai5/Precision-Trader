"""
Institutional Volatility-Momentum Strategy
═══════════════════════════════════════════
- 2-pole Butterworth low-pass filter (scipy.signal.lfilter, causal only)
  applied to price to strip microstructure noise.
- ATR-based volatility regime detection: only trade when vol is expanding.
- ATR momentum (rate of change of ATR) confirms directional conviction.
- Entries on filtered-price crossovers with vol confirmation.
- ATR-scaled risk management (SL/TP).

CRITICAL: lfilter is strictly causal (forward-only). No future data leaks.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

from strategy import compute_stats, atr as calc_atr, ema


# ──────────────────────────────────────────────────────────────────────────────
# Butterworth filter — causal (lfilter only)
# ──────────────────────────────────────────────────────────────────────────────

def butterworth_lowpass(series: np.ndarray, cutoff_period: float, order: int = 2) -> np.ndarray:
    """
    Apply a causal Butterworth low-pass filter to a 1-D price array.

    Args:
        series:        raw price array
        cutoff_period: cutoff expressed as a bar-period (e.g. 20 = passes
                       frequencies slower than 1/20 bars)
        order:         filter order (2 = 2-pole)

    Returns:
        filtered array (same length, causal — no lookahead)
    """
    # Nyquist frequency = 0.5 cycles/bar
    # Normalised cutoff = (1/cutoff_period) / 0.5 = 2/cutoff_period
    wn = 2.0 / cutoff_period
    wn = np.clip(wn, 1e-6, 0.9999)  # must be in (0, 1) exclusive

    b, a = butter(order, wn, btype="low", analog=False)
    # lfilter = strictly causal IIR filter (forward-only), zero lookahead
    filtered = lfilter(b, a, series)
    return filtered


# ──────────────────────────────────────────────────────────────────────────────
# Strategy runner
# ──────────────────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, params: dict) -> dict:
    """
    Institutional volatility-momentum backtest.

    Params (optimisable):
        bw_fast_period   — Butterworth cutoff for fast (signal) line
        bw_slow_period   — Butterworth cutoff for slow (trend) line
        atr_len          — ATR look-back
        vol_ema_len      — EMA of ATR for regime detection
        vol_expansion    — ATR must be > vol_expansion × ATR_EMA to enter
        atr_roc_len      — look-back for ATR rate-of-change
        sl_atr_mult      — stop-loss distance in ATR multiples
        tp_atr_mult      — take-profit distance in ATR multiples

    Frozen:
        initial_capital, contracts, point_value, fee_pct
    """

    # ── Unpack ───────────────────────────────────────────────────────────────
    bw_fast    = int(params.get("bw_fast_period", 10))
    bw_slow    = int(params.get("bw_slow_period", 40))
    atr_len    = int(params.get("atr_len", 14))
    vol_ema    = int(params.get("vol_ema_len", 20))
    vol_exp    = float(params.get("vol_expansion", 1.0))
    roc_len    = int(params.get("atr_roc_len", 5))
    sl_mult    = float(params.get("sl_atr_mult", 1.5))
    tp_mult    = float(params.get("tp_atr_mult", 3.0))

    capital    = float(params.get("initial_capital", 50000.0))
    contracts  = int(params.get("contracts", 2))
    pt_val     = float(params.get("point_value", 2.0))
    fee        = float(params.get("fee_pct", 0.0015))
    use_sess   = bool(params.get("use_session", True))

    # ── Indicators ───────────────────────────────────────────────────────────
    close_arr = df["close"].values.astype(float)
    high_arr  = df["high"].values.astype(float)
    low_arr   = df["low"].values.astype(float)

    # Butterworth-filtered price curves (causal)
    bw_fast_line = butterworth_lowpass(close_arr, bw_fast, order=2)
    bw_slow_line = butterworth_lowpass(close_arr, bw_slow, order=2)

    # ATR
    atr_series = calc_atr(df["high"], df["low"], df["close"], atr_len)
    atr_arr    = atr_series.values.astype(float)

    # ATR regime: EMA of ATR
    atr_ema_series = ema(atr_series, vol_ema)
    atr_ema_arr    = atr_ema_series.values.astype(float)

    # ATR rate-of-change (momentum of volatility)
    atr_roc = np.full(len(atr_arr), 0.0)
    for i in range(roc_len, len(atr_arr)):
        prev = atr_arr[i - roc_len]
        if prev > 0:
            atr_roc[i] = (atr_arr[i] - prev) / prev

    # ── Warmup ───────────────────────────────────────────────────────────────
    warmup = max(bw_slow + 10, atr_len + vol_ema, roc_len + atr_len, 60)

    # ── Trade loop ───────────────────────────────────────────────────────────
    equity = capital
    trades = []
    equity_curve = []
    dates = []

    pos       = 0
    entry_px  = 0.0
    sl_px     = 0.0
    tp_px     = 0.0
    direction = 0
    entry_time = None

    # ── Pre-compute time arrays (avoid per-bar datetime ops) ───────────────
    _idx = df.index
    hours_arr = _idx.hour.values
    minutes_arr = _idx.minute.values

    n = len(df)
    for i in range(n):
        c = close_arr[i]
        h = high_arr[i]
        l = low_arr[i]

        # ── Session filter (9:30-16:00 ET) ───────────────────────────────────
        if use_sess:
            bar_mins = hours_arr[i] * 60 + minutes_arr[i]
            in_session = 570 <= bar_mins < 960

            # Force close at session end
            if bar_mins >= 960 and pos != 0:
                cost = abs(c * abs(pos) * pt_val) * fee
                pnl  = (c - entry_px) * pos * pt_val - cost
                equity += pnl
                trades.append(_trade(entry_time, _idx[i], entry_px, c, direction,
                                     abs(pos), pnl, "Session Close"))
                pos = 0; direction = 0
        else:
            in_session = True

        if i < warmup:
            equity_curve.append(equity)
            dates.append(_idx[i])
            continue

        # ── Read indicators at bar i ─────────────────────────────────────────
        fast_now  = bw_fast_line[i]
        slow_now  = bw_slow_line[i]
        fast_prev = bw_fast_line[i - 1]
        slow_prev = bw_slow_line[i - 1]

        cur_atr     = atr_arr[i]  if not np.isnan(atr_arr[i])  else 20.0
        cur_atr_ema = atr_ema_arr[i] if not np.isnan(atr_ema_arr[i]) else cur_atr
        cur_roc     = atr_roc[i]

        # Volatility expanding?  ATR > threshold × its own EMA
        vol_expanding = cur_atr > vol_exp * cur_atr_ema
        # Volatility momentum positive (vol accelerating)?
        vol_momentum  = cur_roc > 0

        # Butterworth crossovers
        bull_cross = (fast_prev <= slow_prev) and (fast_now > slow_now)
        bear_cross = (fast_prev >= slow_prev) and (fast_now < slow_now)

        # Trend bias from filtered curves
        trend_bull = fast_now > slow_now
        trend_bear = fast_now < slow_now

        # ── Exit logic ───────────────────────────────────────────────────────
        if pos > 0:
            if h >= tp_px:
                cost = tp_px * pos * pt_val * fee
                pnl  = (tp_px - entry_px) * pos * pt_val - cost
                equity += pnl
                trades.append(_trade(entry_time, _idx[i], entry_px, tp_px, 1,
                                     pos, pnl, "TP"))
                pos = 0; direction = 0
            elif l <= sl_px:
                cost = abs(sl_px * pos * pt_val) * fee
                pnl  = (sl_px - entry_px) * pos * pt_val - cost
                equity += pnl
                trades.append(_trade(entry_time, _idx[i], entry_px, sl_px, 1,
                                     pos, pnl, "SL"))
                pos = 0; direction = 0
            # Trend reversal exit: filtered fast crosses below slow while long
            elif bear_cross:
                cost = abs(c * pos * pt_val) * fee
                pnl  = (c - entry_px) * pos * pt_val - cost
                equity += pnl
                trades.append(_trade(entry_time, _idx[i], entry_px, c, 1,
                                     pos, pnl, "Trend Reversal"))
                pos = 0; direction = 0

        elif pos < 0:
            if l <= tp_px:
                cost = abs(tp_px * abs(pos) * pt_val) * fee
                pnl  = (entry_px - tp_px) * abs(pos) * pt_val - cost
                equity += pnl
                trades.append(_trade(entry_time, _idx[i], entry_px, tp_px, -1,
                                     abs(pos), pnl, "TP"))
                pos = 0; direction = 0
            elif h >= sl_px:
                cost = abs(sl_px * abs(pos) * pt_val) * fee
                pnl  = (entry_px - sl_px) * abs(pos) * pt_val - cost
                equity += pnl
                trades.append(_trade(entry_time, _idx[i], entry_px, sl_px, -1,
                                     abs(pos), pnl, "SL"))
                pos = 0; direction = 0
            elif bull_cross:
                cost = abs(c * abs(pos) * pt_val) * fee
                pnl  = (entry_px - c) * abs(pos) * pt_val - cost
                equity += pnl
                trades.append(_trade(entry_time, _idx[i], entry_px, c, -1,
                                     abs(pos), pnl, "Trend Reversal"))
                pos = 0; direction = 0

        # ── Entry logic ──────────────────────────────────────────────────────
        # Requirements:
        #   1. Butterworth crossover (fast crosses slow)
        #   2. Volatility expanding (ATR > threshold × ATR_EMA)
        #   3. Volatility momentum positive (ATR accelerating)
        if pos == 0 and in_session:
            if bull_cross and vol_expanding and vol_momentum:
                entry_px   = c
                sl_px      = c - cur_atr * sl_mult
                tp_px      = c + cur_atr * tp_mult
                pos        = contracts
                direction  = 1
                entry_time = _idx[i]
                equity    -= c * contracts * pt_val * fee

            elif bear_cross and vol_expanding and vol_momentum:
                entry_px   = c
                sl_px      = c + cur_atr * sl_mult
                tp_px      = c - cur_atr * tp_mult
                pos        = -contracts
                direction  = -1
                entry_time = _idx[i]
                equity    -= c * contracts * pt_val * fee

        equity_curve.append(equity)
        dates.append(_idx[i])

    # ── Close open position at end ───────────────────────────────────────────
    if pos != 0:
        c = close_arr[-1]
        cost = abs(c * abs(pos) * pt_val) * fee
        pnl  = (c - entry_px) * pos * pt_val - cost
        equity += pnl
        trades.append(_trade(entry_time, df.index[-1], entry_px, c, direction,
                             abs(pos), pnl, "End of Data"))
        equity_curve[-1] = equity

    eq = pd.Series(equity_curve, index=dates)
    stats = compute_stats(trades, eq, capital)
    return {"trades": trades, "equity": eq, "stats": stats, "params": params}


def _trade(entry_t, exit_t, entry_px, exit_px, direction, contracts, pnl, reason):
    return {
        "entry_time":  entry_t,
        "exit_time":   exit_t,
        "entry_price": entry_px,
        "exit_price":  exit_px,
        "direction":   direction,
        "contracts":   contracts,
        "pnl":         pnl,
        "exit_reason": reason,
        "entry_type":  "BW-ATR",
        "tp1_hit":     reason == "TP",
        "tp2_hit":     False,
    }
