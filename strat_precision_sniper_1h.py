"""
MNQ Precision Sniper v7.5 (1H) — Signal-Based Exits
════════════════════════════════════════════════════════════════
Pure-Python bar-by-bar backtest. Standalone loop (no StrategyParams).

Core logic:
  - 10-point confluence scoring (EMA, RSI, MACD, VWAP, ADX, volume, HTF)
  - EMA crossover entries + optional pullback entries
  - Supertrend directional filter
  - Exits: opposite signal, session close, max trade loss, daily max loss
  - Trailing stop tracks BE/TP1 for informational purposes
  - Session: NY 09:30-16:00
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytz
from strategy import compute_stats, atr as calc_atr


_NY = pytz.timezone("America/New_York")


# ═══════════════════════════════════════════════════════════════════════════════
# Indicator helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ema(series: np.ndarray, length: int) -> np.ndarray:
    """Exponential moving average (vectorised via pandas)."""
    return pd.Series(series).ewm(span=length, adjust=False).mean().values


def _rsi(close: np.ndarray, length: int) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100.0)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_f = _ema(close, fast)
    ema_s = _ema(close, slow)
    macd_line = ema_f - ema_s
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _adx_dmi(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14):
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        plus_dm[i] = up if (up > dn and up > 0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    alpha = 1.0 / length
    atr_s = pd.Series(tr).ewm(alpha=alpha, min_periods=length, adjust=False).mean().values
    plus_di = 100.0 * pd.Series(plus_dm).ewm(alpha=alpha, min_periods=length, adjust=False).mean().values / np.where(atr_s != 0, atr_s, 1.0)
    minus_di = 100.0 * pd.Series(minus_dm).ewm(alpha=alpha, min_periods=length, adjust=False).mean().values / np.where(atr_s != 0, atr_s, 1.0)
    dx = 100.0 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, plus_di + minus_di, 1.0)
    adx = pd.Series(dx).ewm(alpha=alpha, min_periods=length, adjust=False).mean().values
    return plus_di, minus_di, adx


def _vwap_daily(dates_ny, typical: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Cumulative VWAP, reset on each new NY calendar date."""
    n = len(typical)
    vwap = np.zeros(n)
    cum_tv = 0.0
    cum_v = 0.0
    prev_date = None
    _vwap_dates = dates_ny.date
    for i in range(n):
        d = _vwap_dates[i]
        if prev_date is None or d != prev_date:
            cum_tv = 0.0
            cum_v = 0.0
            prev_date = d
        cum_tv += typical[i] * volume[i]
        cum_v += volume[i]
        vwap[i] = cum_tv / cum_v if cum_v > 0 else typical[i]
    return vwap


def _supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                atr_period: int, factor: float):
    atr_vals = calc_atr(pd.Series(high), pd.Series(low), pd.Series(close), atr_period).values.astype(float)
    hl2 = (high + low) / 2.0
    upper = hl2 + factor * atr_vals
    lower = hl2 - factor * atr_vals
    n = len(close)
    final_upper = upper.copy()
    final_lower = lower.copy()
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1=bullish, -1=bearish
    for i in range(1, n):
        if lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = lower[i]
        else:
            final_lower[i] = final_lower[i - 1]
        if upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = upper[i]
        else:
            final_upper[i] = final_upper[i - 1]
        if direction[i - 1] == 1:
            if close[i] < final_lower[i]:
                direction[i] = -1
                supertrend[i] = final_upper[i]
            else:
                direction[i] = 1
                supertrend[i] = final_lower[i]
        else:
            if close[i] > final_upper[i]:
                direction[i] = 1
                supertrend[i] = final_lower[i]
            else:
                direction[i] = -1
                supertrend[i] = final_upper[i]
    return supertrend, direction


# ═══════════════════════════════════════════════════════════════════════════════
# Confluence scoring
# ═══════════════════════════════════════════════════════════════════════════════

def _score_bull(i, close, ema_fast, ema_slow, ema_trend, rsi, macd_hist,
                macd_line, signal_line, vwap, volume, vol_sma,
                adx, plus_di, minus_di, htf_fast, htf_slow):
    s = 0.0
    if ema_fast[i] > ema_slow[i]:
        s += 1.0
    if close[i] > ema_trend[i]:
        s += 1.0
    if 50 < rsi[i] < 70:
        s += 1.0
    if macd_hist[i] > 0:
        s += 1.0
    if macd_line[i] > signal_line[i]:
        s += 1.0
    if close[i] > vwap[i]:
        s += 1.0
    if vol_sma[i] > 0 and volume[i] > vol_sma[i] * 1.2:
        s += 1.0
    if adx[i] > 20 and plus_di[i] > minus_di[i]:
        s += 1.0
    if htf_fast[i] > htf_slow[i]:
        s += 1.5
    if close[i] > ema_fast[i]:
        s += 0.5
    return s


def _score_bear(i, close, ema_fast, ema_slow, ema_trend, rsi, macd_hist,
                macd_line, signal_line, vwap, volume, vol_sma,
                adx, plus_di, minus_di, htf_fast, htf_slow):
    s = 0.0
    if ema_fast[i] < ema_slow[i]:
        s += 1.0
    if close[i] < ema_trend[i]:
        s += 1.0
    if 30 < rsi[i] < 50:
        s += 1.0
    if macd_hist[i] < 0:
        s += 1.0
    if macd_line[i] < signal_line[i]:
        s += 1.0
    if close[i] < vwap[i]:
        s += 1.0
    if vol_sma[i] > 0 and volume[i] > vol_sma[i] * 1.2:
        s += 1.0
    if adx[i] > 20 and minus_di[i] > plus_di[i]:
        s += 1.0
    if htf_fast[i] < htf_slow[i]:
        s += 1.5
    if close[i] < ema_fast[i]:
        s += 0.5
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# Main backtest
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, params: dict) -> dict:
    # ── Unpack parameters ────────────────────────────────────────────────────
    ema_fast_len        = int(params.get("ema_fast_len", 11))
    ema_slow_len        = int(params.get("ema_slow_len", 21))
    ema_trend_len       = int(params.get("ema_trend_len", 10))
    min_score           = float(params.get("min_score", 5))
    rsi_len             = int(params.get("rsi_len", 25))
    use_pullback        = bool(params.get("use_pullback", 1))
    pullback_min_score  = float(params.get("pullback_min_score", 4))
    fixed_risk_pts      = float(params.get("fixed_risk_pts", 20.0))
    tp1_rr              = float(params.get("tp1_rr", 1.0))
    tp2_rr              = float(params.get("tp2_rr", 3.5))
    contracts           = int(params.get("contracts_per_trade", 2))
    use_supertrend      = bool(params.get("use_supertrend", 1))
    st_atr_len          = int(params.get("st_atr_len", 30))
    st_factor           = float(params.get("st_factor", 4.5))
    use_session         = bool(params.get("use_session", 1))
    use_force_close     = bool(params.get("use_force_close", 1))
    force_close_hour    = int(params.get("force_close_hour", 16))
    use_max_trade_loss  = bool(params.get("use_max_trade_loss", 1))
    max_trade_loss      = float(params.get("max_trade_loss", 350.0))
    use_daily_max_loss  = bool(params.get("use_daily_max_loss", 1))
    daily_max_loss      = float(params.get("daily_max_loss", 1000.0))
    capital             = float(params.get("initial_capital", 50000.0))
    pt_val              = float(params.get("point_value", 2.0))
    fee_per_contract    = float(params.get("fee_per_contract", 0.62))

    # ── Extract arrays ───────────────────────────────────────────────────────
    close  = df["close"].values.astype(float)
    high   = df["high"].values.astype(float)
    low    = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)
    n      = len(df)

    # ── NY datetimes ─────────────────────────────────────────────────────────
    idx = df.index
    # If the index is tz-naive, localise; otherwise convert
    if idx.tz is None:
        ny_times = idx.tz_localize("UTC").tz_convert(_NY)
    else:
        ny_times = idx.tz_convert(_NY)

    # ── Compute indicators ───────────────────────────────────────────────────
    ema_fast  = _ema(close, ema_fast_len)
    ema_slow  = _ema(close, ema_slow_len)
    ema_trend = _ema(close, ema_trend_len)
    rsi       = _rsi(close, rsi_len)
    macd_line, signal_line, macd_hist = _macd(close, 12, 26, 9)
    atr_arr   = calc_atr(df["high"], df["low"], df["close"], 14).values.astype(float)
    plus_di, minus_di, adx = _adx_dmi(high, low, close, 14)
    vol_sma   = _ema(volume, 10)  # SMA approximated; use rolling for exact
    vol_sma   = pd.Series(volume).rolling(10).mean().values

    typical   = (high + low + close) / 3.0
    vwap      = _vwap_daily(ny_times, typical, volume)

    st_vals, st_dir = _supertrend(high, low, close, st_atr_len, st_factor)

    # HTF bias (4x lengths as proxy for 4H on 1H data)
    htf_fast = _ema(close, ema_fast_len * 4)
    htf_slow = _ema(close, ema_slow_len * 4)

    # ── Precompute per-bar scores and crossover flags ────────────────────────
    bull_scores = np.zeros(n)
    bear_scores = np.zeros(n)
    cross_up    = np.zeros(n, dtype=bool)
    cross_down  = np.zeros(n, dtype=bool)

    for i in range(1, n):
        bull_scores[i] = _score_bull(i, close, ema_fast, ema_slow, ema_trend,
                                     rsi, macd_hist, macd_line, signal_line,
                                     vwap, volume, vol_sma, adx, plus_di,
                                     minus_di, htf_fast, htf_slow)
        bear_scores[i] = _score_bear(i, close, ema_fast, ema_slow, ema_trend,
                                     rsi, macd_hist, macd_line, signal_line,
                                     vwap, volume, vol_sma, adx, plus_di,
                                     minus_di, htf_fast, htf_slow)
        cross_up[i]   = ema_fast[i] > ema_slow[i] and ema_fast[i - 1] <= ema_slow[i - 1]
        cross_down[i] = ema_fast[i] < ema_slow[i] and ema_fast[i - 1] >= ema_slow[i - 1]

    # ── Pre-compute NY time arrays (avoid per-bar datetime ops) ────────────
    _idx = idx
    ny_hours_arr = ny_times.hour.values
    ny_minutes_arr = ny_times.minute.values
    ny_dates_arr = ny_times.date

    # ── Trade loop ───────────────────────────────────────────────────────────
    equity       = capital
    trades       = []
    equity_curve = []
    date_list    = []

    # Position state
    pos          = 0       # +contracts = long, -contracts = short
    direction    = 0       # 1 long, -1 short
    entry_px     = 0.0
    entry_time   = None
    sl_px        = 0.0
    tp1_px       = 0.0
    tp2_px       = 0.0
    tp1_hit      = False
    tp2_hit      = False
    trail_px     = np.nan

    # Daily P&L state
    daily_pnl    = 0.0
    daily_blocked = False
    prev_ny_date = None

    for i in range(n):
        ny_hr  = ny_hours_arr[i]
        ny_min = ny_minutes_arr[i]
        ny_date = ny_dates_arr[i]
        c = close[i]
        h = high[i]
        l = low[i]

        # ── New day reset ────────────────────────────────────────────────────
        if prev_ny_date is None or ny_date != prev_ny_date:
            daily_pnl = 0.0
            daily_blocked = False
            prev_ny_date = ny_date

        # ── Session checks ───────────────────────────────────────────────────
        in_session = (ny_hr > 9 or (ny_hr == 9 and ny_min >= 30)) and ny_hr < 16
        should_force_close = use_force_close and ny_hr >= force_close_hour
        outside_holding = ny_hr < 8 or ny_hr >= 16

        # ── Force close at session end ───────────────────────────────────────
        if should_force_close and pos != 0:
            exit_pnl = _calc_pnl(pos, direction, entry_px, c, pt_val, fee_per_contract)
            equity += exit_pnl
            daily_pnl += exit_pnl
            trades.append(_trade(entry_time, _idx[i], entry_px, c, direction,
                                 abs(pos), exit_pnl, "Session Close",
                                 tp1_hit, tp2_hit))
            pos = 0; direction = 0; entry_px = 0.0
            tp1_hit = False; tp2_hit = False; trail_px = np.nan

        # ── Max trade loss (emergency exit) ──────────────────────────────────
        # Use high/low for worst-case intra-bar P&L, not close
        if pos != 0 and use_max_trade_loss:
            if direction == 1:
                worst_px = l  # worst case for long is bar low
                unrealized = (worst_px - entry_px) * abs(pos) * pt_val
            else:
                worst_px = h  # worst case for short is bar high
                unrealized = (entry_px - worst_px) * abs(pos) * pt_val
            if unrealized <= -max_trade_loss:
                # Exit at the stop price, not at close
                stop_delta = max_trade_loss / (abs(pos) * pt_val)
                if direction == 1:
                    exit_px = entry_px - stop_delta
                else:
                    exit_px = entry_px + stop_delta
                exit_pnl = _calc_pnl(pos, direction, entry_px, exit_px, pt_val, fee_per_contract)
                equity += exit_pnl
                daily_pnl += exit_pnl
                trades.append(_trade(entry_time, _idx[i], entry_px, exit_px, direction,
                                     abs(pos), exit_pnl, "Max Trade Loss",
                                     tp1_hit, tp2_hit))
                pos = 0; direction = 0; entry_px = 0.0
                tp1_hit = False; tp2_hit = False; trail_px = np.nan

        # ── Daily max loss check ─────────────────────────────────────────────
        # Include unrealized P&L from open position in daily loss calculation
        if use_daily_max_loss:
            effective_daily = daily_pnl
            if pos != 0:
                if direction == 1:
                    effective_daily += (l - entry_px) * abs(pos) * pt_val
                else:
                    effective_daily += (entry_px - h) * abs(pos) * pt_val
            if effective_daily <= -daily_max_loss:
                daily_blocked = True
                if pos != 0:
                    # Exit at the daily loss limit price
                    remaining = daily_max_loss + daily_pnl  # how much the open trade can lose
                    stop_delta = max(0.0, remaining) / (abs(pos) * pt_val)
                    if direction == 1:
                        exit_px = entry_px - stop_delta
                    else:
                        exit_px = entry_px + stop_delta
                    exit_pnl = _calc_pnl(pos, direction, entry_px, exit_px, pt_val, fee_per_contract)
                    equity += exit_pnl
                    daily_pnl += exit_pnl
                    trades.append(_trade(entry_time, _idx[i], entry_px, exit_px, direction,
                                         abs(pos), exit_pnl, "Daily Max Loss",
                                         tp1_hit, tp2_hit))
                    pos = 0; direction = 0; entry_px = 0.0
                    tp1_hit = False; tp2_hit = False; trail_px = np.nan

        # ── Trailing stop tracking (informational, no exit) ──────────────────
        if pos != 0:
            if direction == 1:
                if not tp1_hit and h >= tp1_px:
                    tp1_hit = True
                    trail_px = entry_px  # move to breakeven
                if not tp2_hit and h >= tp2_px:
                    tp2_hit = True
                    trail_px = tp1_px    # move to TP1
            else:
                if not tp1_hit and l <= tp1_px:
                    tp1_hit = True
                    trail_px = entry_px
                if not tp2_hit and l <= tp2_px:
                    tp2_hit = True
                    trail_px = tp1_px

        # ── Signal generation ────────────────────────────────────────────────
        long_signal = False
        short_signal = False

        if i >= 1 and not daily_blocked and not outside_holding:
            session_ok = (not use_session) or in_session
            st_bull = (not use_supertrend) or st_dir[i] == 1
            st_bear = (not use_supertrend) or st_dir[i] == -1

            # Crossover entries
            if cross_up[i] and rsi[i] < 75 and bull_scores[i] >= min_score and session_ok and st_bull:
                long_signal = True
            if cross_down[i] and rsi[i] > 25 and bear_scores[i] >= min_score and session_ok and st_bear:
                short_signal = True

            # Pullback entries
            if use_pullback and not long_signal and session_ok and st_bull:
                if (ema_fast[i] > ema_slow[i] and close[i] > ema_trend[i]
                        and close[i] > ema_fast[i]):
                    # Check if low touched ema_fast in last 3 bars
                    touched = False
                    for k in range(max(0, i - 2), i + 1):
                        if low[k] <= ema_fast[k]:
                            touched = True
                            break
                    if touched and rsi[i] < 75 and bull_scores[i] >= pullback_min_score:
                        long_signal = True

            if use_pullback and not short_signal and session_ok and st_bear:
                if (ema_fast[i] < ema_slow[i] and close[i] < ema_trend[i]
                        and close[i] < ema_fast[i]):
                    touched = False
                    for k in range(max(0, i - 2), i + 1):
                        if high[k] >= ema_fast[k]:
                            touched = True
                            break
                    if touched and rsi[i] > 25 and bear_scores[i] >= pullback_min_score:
                        short_signal = True

        # ── Exit on opposite signal ──────────────────────────────────────────
        if pos > 0 and short_signal:
            exit_pnl = _calc_pnl(pos, direction, entry_px, c, pt_val, fee_per_contract)
            equity += exit_pnl
            daily_pnl += exit_pnl
            trades.append(_trade(entry_time, _idx[i], entry_px, c, direction,
                                 abs(pos), exit_pnl, "Opposite Signal",
                                 tp1_hit, tp2_hit))
            pos = 0; direction = 0; entry_px = 0.0
            tp1_hit = False; tp2_hit = False; trail_px = np.nan

        if pos < 0 and long_signal:
            exit_pnl = _calc_pnl(pos, direction, entry_px, c, pt_val, fee_per_contract)
            equity += exit_pnl
            daily_pnl += exit_pnl
            trades.append(_trade(entry_time, _idx[i], entry_px, c, direction,
                                 abs(pos), exit_pnl, "Opposite Signal",
                                 tp1_hit, tp2_hit))
            pos = 0; direction = 0; entry_px = 0.0
            tp1_hit = False; tp2_hit = False; trail_px = np.nan

        # ── Entry logic ──────────────────────────────────────────────────────
        if not daily_blocked and not should_force_close and not outside_holding:
            if pos == 0 and long_signal:
                entry_px   = c
                pos        = contracts
                direction  = 1
                entry_time = _idx[i]
                sl_px      = entry_px - fixed_risk_pts
                tp1_px     = entry_px + fixed_risk_pts * tp1_rr
                tp2_px     = entry_px + fixed_risk_pts * tp2_rr
                tp1_hit    = False
                tp2_hit    = False
                trail_px   = np.nan

            elif pos == 0 and short_signal:
                entry_px   = c
                pos        = -contracts
                direction  = -1
                entry_time = _idx[i]
                sl_px      = entry_px + fixed_risk_pts
                tp1_px     = entry_px - fixed_risk_pts * tp1_rr
                tp2_px     = entry_px - fixed_risk_pts * tp2_rr
                tp1_hit    = False
                tp2_hit    = False
                trail_px   = np.nan

        equity_curve.append(equity)
        date_list.append(_idx[i])

    # ── Close open position at end of data ───────────────────────────────────
    if pos != 0:
        c = close[-1]
        exit_pnl = _calc_pnl(pos, direction, entry_px, c, pt_val, fee_per_contract)
        equity += exit_pnl
        trades.append(_trade(entry_time, idx[-1], entry_px, c, direction,
                             abs(pos), exit_pnl, "End of Data",
                             tp1_hit, tp2_hit))
        equity_curve[-1] = equity

    eq = pd.Series(equity_curve, index=date_list)
    stats = compute_stats(trades, eq, capital)
    return {"trades": trades, "equity": eq, "stats": stats, "params": params}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _calc_pnl(pos, direction, entry_px, exit_px, pt_val, fee_per_ct):
    cost = abs(pos) * fee_per_ct
    if direction == 1:
        return (exit_px - entry_px) * abs(pos) * pt_val - cost
    else:
        return (entry_px - exit_px) * abs(pos) * pt_val - cost


def _trade(entry_t, exit_t, entry_px, exit_px, direction, contracts, pnl,
           reason, tp1_hit=False, tp2_hit=False):
    return {
        "entry_time":  entry_t,
        "exit_time":   exit_t,
        "entry_price": entry_px,
        "exit_price":  exit_px,
        "direction":   direction,
        "contracts":   contracts,
        "pnl":         pnl,
        "exit_reason": reason,
        "entry_type":  "Precision Sniper",
        "tp1_hit":     tp1_hit,
        "tp2_hit":     tp2_hit,
    }
