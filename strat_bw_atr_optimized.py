"""
Butterworth ATR Vol-Momentum Strategy [1H] — Optimized — Prop Firm Safe
════════════════════════════════════════════════════════════════════════
Converted from the user's optimized PineScript. Exact same logic:
  - 2-pole causal Butterworth filter (lfilter, zero lookahead)
  - ATR expansion + ATR momentum confirmation
  - 2-contract TP1/TP2 + SL with trail-to-BE after TP1
  - Session filter: Tokyo open (19:00 ET) → NY close (16:00 ET)
  - Force close at 16:00 ET (prop firm safe)
  - Bar-close-only entries (no intrabar)

Optimized parameters baked in as defaults:
  BW Fast=3, BW Slow=5, ATR=10, Vol EMA=75,
  Vol Expansion=0.6, ATR ROC=4, SL=5.0x ATR,
  TP1=2.0R, TP2=5.0R
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

from strategy import compute_stats, atr as calc_atr, ema


# ──────────────────────────────────────────────────────────────────────────────
# Butterworth filter — causal (lfilter only, zero lookahead)
# ──────────────────────────────────────────────────────────────────────────────

def butterworth_lowpass(series: np.ndarray, cutoff_period: float, order: int = 2) -> np.ndarray:
    wn = 2.0 / max(cutoff_period, 3)
    wn = np.clip(wn, 1e-6, 0.9999)
    b, a = butter(order, wn, btype="low", analog=False)
    return lfilter(b, a, series)


# ──────────────────────────────────────────────────────────────────────────────
# Strategy runner — matches PineScript bar-for-bar
# ──────────────────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, params: dict) -> dict:
    # ── Unpack parameters (optimized defaults match PineScript) ──────────────
    bw_fast     = int(params.get("bw_fast_period", 3))
    bw_slow     = int(params.get("bw_slow_period", 5))
    atr_len     = int(params.get("atr_len", 10))
    vol_ema_len = int(params.get("vol_ema_len", 75))
    vol_exp     = float(params.get("vol_expansion", 0.6))
    roc_len     = int(params.get("atr_roc_len", 4))
    sl_mult     = float(params.get("sl_atr_mult", 5.0))
    tp1_rr      = float(params.get("tp1_rr", 2.0))
    tp2_rr      = float(params.get("tp2_rr", 5.0))
    tp1_qty     = int(params.get("tp1_qty", 1))
    tp2_qty     = int(params.get("tp2_qty", 1))
    use_trail   = bool(params.get("use_trail", True))

    capital     = float(params.get("initial_capital", 50000.0))
    contracts   = tp1_qty + tp2_qty
    pt_val      = float(params.get("point_value", 2.0))
    fee         = float(params.get("fee_pct", 0.0015))

    # Max loss per trade (emergency exit)
    use_max_loss    = bool(params.get("use_max_trade_loss", True))
    max_trade_loss  = float(params.get("max_trade_loss", 300.0))

    # Daily max loss
    use_daily_max   = bool(params.get("use_daily_max_loss", True))
    daily_max_loss  = float(params.get("daily_max_loss", 750.0))

    # Session: Tokyo open (19:00 ET) → NY close (16:00 ET)
    sess_open   = int(params.get("session_open_et", 19)) * 60  # 1140
    sess_close  = int(params.get("session_close_et", 16)) * 60  # 960
    use_force   = bool(params.get("use_force_close", True))

    # ── Compute indicators ───────────────────────────────────────────────────
    close_arr = df["close"].values.astype(float)
    high_arr  = df["high"].values.astype(float)
    low_arr   = df["low"].values.astype(float)

    bw_fast_line = butterworth_lowpass(close_arr, bw_fast, order=2)
    bw_slow_line = butterworth_lowpass(close_arr, bw_slow, order=2)

    atr_series   = calc_atr(df["high"], df["low"], df["close"], atr_len)
    atr_arr      = atr_series.values.astype(float)
    atr_ema_arr  = ema(atr_series, vol_ema_len).values.astype(float)

    # ATR rate-of-change
    atr_roc = np.zeros(len(atr_arr))
    for i in range(roc_len, len(atr_arr)):
        prev = atr_arr[i - roc_len]
        if prev > 0:
            atr_roc[i] = (atr_arr[i] - prev) / prev

    warmup = max(bw_slow + 10, atr_len + vol_ema_len, roc_len + atr_len, 60)

    # ── Trade state ──────────────────────────────────────────────────────────
    equity       = capital
    trades       = []
    equity_curve = []
    day_start_eq = capital
    prev_date    = None
    daily_loss_breached = False
    dates        = []

    pos          = 0       # +n long, -n short
    entry_px     = 0.0
    sl_px        = 0.0
    tp1_px       = 0.0
    tp2_px       = 0.0
    trail_stop   = 0.0
    direction    = 0
    tp1_hit      = False
    entry_time   = None
    last_entry_i = -1

    # ── Pre-compute time arrays (avoid per-bar datetime ops) ───────────────
    _idx = df.index
    hours_arr = _idx.hour.values
    minutes_arr = _idx.minute.values
    dates_arr = _idx.date

    n = len(df)
    for i in range(n):
        c = close_arr[i]
        h = high_arr[i]
        l = low_arr[i]

        # ── Daily P&L tracker ────────────────────────────────────────────────
        bar_date = dates_arr[i]
        if prev_date is None or bar_date != prev_date:
            day_start_eq = equity
            daily_loss_breached = False
            prev_date = bar_date

        daily_pnl = equity - day_start_eq
        if use_daily_max and daily_pnl <= -daily_max_loss:
            daily_loss_breached = True
            if pos != 0:
                cost = abs(c * abs(pos) * pt_val) * fee
                pnl  = (c - entry_px) * pos * pt_val - cost
                equity += pnl
                trades.append(_trade(entry_time, _idx[i], entry_px, c, direction,
                                     abs(pos), pnl, "Daily Max Loss"))
                pos = 0; direction = 0; tp1_hit = False

        # ── Max loss per trade (emergency exit) ─────────────────────────────
        if use_max_loss and pos != 0:
            if pos > 0:
                unreal = (c - entry_px) * pos * pt_val
            else:
                unreal = (entry_px - c) * abs(pos) * pt_val
            if unreal <= -max_trade_loss:
                cost = abs(c * abs(pos) * pt_val) * fee
                pnl  = (c - entry_px) * pos * pt_val - cost
                equity += pnl
                trades.append(_trade(entry_time, _idx[i], entry_px, c, direction,
                                     abs(pos), pnl, "Max Trade Loss"))
                pos = 0; direction = 0; tp1_hit = False

        # ── Session logic (Eastern Time) ─────────────────────────────────────
        bar_mins = hours_arr[i] * 60 + minutes_arr[i]

        # Session spans midnight: open=19:00(1140) → close=16:00(960)
        # Active when: bar_mins >= 1140 OR bar_mins < 960
        if sess_open > sess_close:
            in_session = (bar_mins >= sess_open) or (bar_mins < sess_close)
        else:
            in_session = (bar_mins >= sess_open) and (bar_mins < sess_close)

        # Past NY close: between 16:00 and 19:00
        past_ny_close = (bar_mins >= sess_close) and (bar_mins < sess_open)

        # ── Force close at NY close ──────────────────────────────────────────
        if use_force and past_ny_close and pos != 0:
            cost = abs(c * abs(pos) * pt_val) * fee
            pnl  = (c - entry_px) * pos * pt_val - cost
            equity += pnl
            trades.append(_trade(entry_time, _idx[i], entry_px, c, direction,
                                 abs(pos), pnl, "NY Close"))
            pos = 0; direction = 0; tp1_hit = False

        if i < warmup or i < 1:
            equity_curve.append(equity)
            dates.append(_idx[i])
            continue

        # ── Read indicators ──────────────────────────────────────────────────
        fast_now  = bw_fast_line[i]
        slow_now  = bw_slow_line[i]
        fast_prev = bw_fast_line[i - 1]
        slow_prev = bw_slow_line[i - 1]

        cur_atr     = atr_arr[i]     if not np.isnan(atr_arr[i])     else 20.0
        cur_atr_ema = atr_ema_arr[i] if not np.isnan(atr_ema_arr[i]) else cur_atr
        cur_roc     = atr_roc[i]

        vol_expanding = cur_atr > vol_exp * cur_atr_ema
        vol_momentum  = cur_roc > 0

        bull_cross = (fast_prev <= slow_prev) and (fast_now > slow_now)
        bear_cross = (fast_prev >= slow_prev) and (fast_now < slow_now)

        can_enter = (i != last_entry_i)
        risk_pts  = cur_atr * sl_mult if cur_atr * sl_mult > 0 else 25.0

        # ── TP1 / TP2 / SL exit logic ───────────────────────────────────────
        if pos > 0 and direction == 1:
            # Check TP1 (partial exit)
            if not tp1_hit and h >= tp1_px:
                tp1_hit = True
                exit_qty = min(tp1_qty, pos)
                if exit_qty > 0:
                    cost = tp1_px * exit_qty * pt_val * fee
                    pnl  = (tp1_px - entry_px) * exit_qty * pt_val - cost
                    equity += pnl
                    pos -= exit_qty
                    trades.append(_trade(entry_time, _idx[i], entry_px, tp1_px,
                                        1, exit_qty, pnl, "TP1"))
                if use_trail:
                    trail_stop = entry_px  # move stop to BE

            # Check TP2 (runner)
            if pos > 0 and h >= tp2_px:
                exit_qty = min(tp2_qty, pos)
                if exit_qty > 0:
                    cost = tp2_px * exit_qty * pt_val * fee
                    pnl  = (tp2_px - entry_px) * exit_qty * pt_val - cost
                    equity += pnl
                    pos -= exit_qty
                    trades.append(_trade(entry_time, _idx[i], entry_px, tp2_px,
                                        1, exit_qty, pnl, "TP2"))

            # Check SL / trail stop
            if pos > 0:
                stop = trail_stop if tp1_hit else sl_px
                if l <= stop:
                    cost = abs(stop * pos * pt_val) * fee
                    pnl  = (stop - entry_px) * pos * pt_val - cost
                    equity += pnl
                    trades.append(_trade(entry_time, _idx[i], entry_px, stop,
                                        1, pos, pnl, "SL"))
                    pos = 0; direction = 0; tp1_hit = False

        elif pos < 0 and direction == -1:
            # Check TP1 (partial exit)
            if not tp1_hit and l <= tp1_px:
                tp1_hit = True
                exit_qty = min(tp1_qty, abs(pos))
                if exit_qty > 0:
                    cost = tp1_px * exit_qty * pt_val * fee
                    pnl  = (entry_px - tp1_px) * exit_qty * pt_val - cost
                    equity += pnl
                    pos += exit_qty
                    trades.append(_trade(entry_time, _idx[i], entry_px, tp1_px,
                                        -1, exit_qty, pnl, "TP1"))
                if use_trail:
                    trail_stop = entry_px  # move stop to BE

            # Check TP2 (runner)
            if pos < 0 and l <= tp2_px:
                exit_qty = min(tp2_qty, abs(pos))
                if exit_qty > 0:
                    cost = tp2_px * exit_qty * pt_val * fee
                    pnl  = (entry_px - tp2_px) * exit_qty * pt_val - cost
                    equity += pnl
                    pos += exit_qty
                    trades.append(_trade(entry_time, _idx[i], entry_px, tp2_px,
                                        -1, exit_qty, pnl, "TP2"))

            # Check SL / trail stop
            if pos < 0:
                stop = trail_stop if tp1_hit else sl_px
                if h >= stop:
                    cost = abs(stop * abs(pos) * pt_val) * fee
                    pnl  = (entry_px - stop) * abs(pos) * pt_val - cost
                    equity += pnl
                    trades.append(_trade(entry_time, _idx[i], entry_px, stop,
                                        -1, abs(pos), pnl, "SL"))
                    pos = 0; direction = 0; tp1_hit = False

        # ── Entry logic (bar-close confirmed, no intrabar) ───────────────────
        if pos == 0 and in_session and not past_ny_close and can_enter and not daily_loss_breached:
            # Long: BW fast crosses above slow + vol expanding + vol momentum
            if bull_cross and vol_expanding and vol_momentum:
                entry_px   = c
                sl_px      = c - risk_pts
                tp1_px     = c + risk_pts * tp1_rr
                tp2_px     = c + risk_pts * tp2_rr
                trail_stop = sl_px
                pos        = contracts
                direction  = 1
                tp1_hit    = False
                entry_time = _idx[i]
                last_entry_i = i
                equity -= c * contracts * pt_val * fee  # entry cost

            # Short: BW fast crosses below slow + vol expanding + vol momentum
            elif bear_cross and vol_expanding and vol_momentum:
                entry_px   = c
                sl_px      = c + risk_pts
                tp1_px     = c - risk_pts * tp1_rr
                tp2_px     = c - risk_pts * tp2_rr
                trail_stop = sl_px
                pos        = -contracts
                direction  = -1
                tp1_hit    = False
                entry_time = _idx[i]
                last_entry_i = i
                equity -= c * contracts * pt_val * fee  # entry cost

        equity_curve.append(equity)
        dates.append(_idx[i])

    # ── Close any open position at end of data ───────────────────────────────
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
        "tp1_hit":     reason in ("TP1", "TP2"),
        "tp2_hit":     reason == "TP2",
    }
