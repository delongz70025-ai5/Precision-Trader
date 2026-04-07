"""
MNQ Tokyo Rally-Base-Rally (1H) — 2 Contracts: TP1, Runner
════════════════════════════════════════════════════════════
Converted from PineScript. Exact same logic:
  - Tokyo session (09:00-14:00 Asia/Tokyo)
  - EMA 9/21 trend filter (uptrend = fast > slow)
  - Volume spike: volume >= vol_multiplier * SMA(volume, vol_lookback)
  - Rally-Base-Rally (bullish) / Drop-Base-Drop (bearish) pattern detection
  - EMA touch confirmation
  - Fixed risk (90 pts), TP1 at 1.5x R:R (1 contract), Runner at 3.75x R:R (1 contract)
  - 2 contracts split 1/1
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytz
from strategy import compute_stats


TOKYO_TZ = pytz.timezone("Asia/Tokyo")


def run_backtest(df: pd.DataFrame, params: dict) -> dict:
    # ── Unpack parameters ────────────────────────────────────────────────────
    fast_len          = int(params.get("fast_len", 9))
    slow_len          = int(params.get("slow_len", 21))

    vol_lookback      = int(params.get("vol_lookback", 10))
    vol_multiplier    = float(params.get("vol_multiplier", 1.5))

    rbr_body_ratio    = float(params.get("rbr_body_ratio", 0.55))
    base_doji_ratio   = float(params.get("base_doji_ratio", 0.45))
    max_base_bars     = int(params.get("max_base_bars", 3))
    rbr_lookback      = int(params.get("rbr_lookback", 15))

    fixed_risk_pts    = float(params.get("fixed_risk_points", 90.0))
    first_rr          = float(params.get("first_rr_ratio", 1.5))
    runner_rr         = float(params.get("runner_rr_ratio", 3.75))
    contracts         = int(params.get("contracts_per_trade", 2))

    capital           = float(params.get("initial_capital", 50000.0))
    pt_val            = float(params.get("point_value", 2.0))
    fee_per_contract  = float(params.get("fee_per_contract", 0.62))

    # ── Compute indicators ───────────────────────────────────────────────────
    ema_f = df["close"].ewm(span=fast_len, adjust=False).mean().values.astype(float)
    ema_s = df["close"].ewm(span=slow_len, adjust=False).mean().values.astype(float)
    vol_sma = df["volume"].rolling(vol_lookback).mean().values.astype(float)

    close_arr = df["close"].values.astype(float)
    open_arr  = df["open"].values.astype(float)
    high_arr  = df["high"].values.astype(float)
    low_arr   = df["low"].values.astype(float)
    vol_arr   = df["volume"].values.astype(float)

    # ── Precompute body_ratio array ──────────────────────────────────────────
    rng_arr = high_arr - low_arr
    body_arr = np.abs(close_arr - open_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        body_ratio_arr = np.where(rng_arr == 0, 0.0, body_arr / rng_arr)

    # ── Convert index to Tokyo time for session check ────────────────────────
    _idx = df.index
    if _idx.tz is None:
        _tz_index = _idx.tz_localize("UTC").tz_convert(TOKYO_TZ)
    else:
        _tz_index = _idx.tz_convert(TOKYO_TZ)
    hours_arr = _tz_index.hour.values
    minutes_arr = _tz_index.minute.values

    # ── Trade loop ───────────────────────────────────────────────────────────
    equity       = capital
    trades       = []
    equity_curve = []
    dates        = []

    # Position state
    pos          = 0       # remaining contracts (positive=long, negative=short)
    entry_px     = 0.0
    direction    = 0       # 1=long, -1=short
    stop_px      = 0.0
    tp1_px       = 0.0
    runner_px    = 0.0
    tp1_filled   = False
    entry_time   = None

    n = len(df)
    for i in range(n):
        c = close_arr[i]
        o = open_arr[i]
        h = high_arr[i]
        l = low_arr[i]
        v = vol_arr[i]
        t_hr = hours_arr[i]

        # ── Session filter: Tokyo 09:00-14:00 ───────────────────────────────
        in_session = 9 <= t_hr < 14

        # ── Trend ────────────────────────────────────────────────────────────
        ef = ema_f[i]
        es = ema_s[i]
        uptrend   = ef > es
        downtrend = ef < es
        ema_top = max(ef, es)
        ema_bot = min(ef, es)

        # ── Volume spike ─────────────────────────────────────────────────────
        cur_vol_sma = vol_sma[i] if not np.isnan(vol_sma[i]) else 0.0
        vol_spike = cur_vol_sma > 0 and v >= cur_vol_sma * vol_multiplier

        # ── Candle confirm ───────────────────────────────────────────────────
        bull_confirm = c > o
        bear_confirm = c < o

        # ── EMA touch ────────────────────────────────────────────────────────
        ema_touch_bull = l <= ema_top and c > ema_top
        ema_touch_bear = h >= ema_bot and c < ema_bot

        # ── RBR detection (bullish) ──────────────────────────────────────────
        rbr_found = False
        if i >= rbr_lookback:
            for base_count in range(1, max_base_bars + 1):
                rally1_idx = base_count + 1  # bars ago
                if rally1_idx >= rbr_lookback:
                    continue
                r1 = i - rally1_idx
                if r1 < 0:
                    continue
                # Check rally1 candle
                if body_ratio_arr[r1] >= rbr_body_ratio and close_arr[r1] > open_arr[r1]:
                    # Check all base candles (bars 1..base_count ago)
                    all_base = True
                    base_hi = -np.inf
                    base_lo = np.inf
                    for b in range(1, base_count + 1):
                        bi = i - b
                        if bi < 0:
                            all_base = False
                            break
                        if body_ratio_arr[bi] > base_doji_ratio:
                            all_base = False
                            break
                        base_hi = max(base_hi, high_arr[bi])
                        base_lo = min(base_lo, low_arr[bi])
                    if all_base and c > base_hi:
                        rbr_found = True
                        break

        # ── DBD detection (bearish) ──────────────────────────────────────────
        dbd_found = False
        if i >= rbr_lookback:
            for base_count in range(1, max_base_bars + 1):
                drop1_idx = base_count + 1  # bars ago
                if drop1_idx >= rbr_lookback:
                    continue
                d1 = i - drop1_idx
                if d1 < 0:
                    continue
                # Check drop1 candle
                if body_ratio_arr[d1] >= rbr_body_ratio and close_arr[d1] < open_arr[d1]:
                    # Check all base candles (bars 1..base_count ago)
                    all_base = True
                    base_hi = -np.inf
                    base_lo = np.inf
                    for b in range(1, base_count + 1):
                        bi = i - b
                        if bi < 0:
                            all_base = False
                            break
                        if body_ratio_arr[bi] > base_doji_ratio:
                            all_base = False
                            break
                        base_hi = max(base_hi, high_arr[bi])
                        base_lo = min(base_lo, low_arr[bi])
                    if all_base and c < base_lo:
                        dbd_found = True
                        break

        # ── Exit logic ───────────────────────────────────────────────────────
        if pos > 0 and direction == 1:
            # Long: check SL first
            if l <= stop_px:
                # Stop hit — close all remaining contracts
                pnl = _calc_pnl(pos, 1, entry_px, stop_px, pt_val, fee_per_contract)
                equity += pnl
                trades.append(_trade(entry_time, _idx[i], entry_px, stop_px, 1, pos, pnl,
                                     "SL" if not tp1_filled else "Runner SL"))
                pos = 0; direction = 0; tp1_filled = False
            else:
                # TP1
                if not tp1_filled and h >= tp1_px:
                    tp1_filled = True
                    pnl = (tp1_px - entry_px) * 1 * pt_val - fee_per_contract
                    equity += pnl
                    pos -= 1
                    trades.append(_trade(entry_time, _idx[i], entry_px, tp1_px, 1, 1, pnl, "TP1"))

                # Runner TP
                if pos > 0 and h >= runner_px:
                    pnl = (runner_px - entry_px) * 1 * pt_val - fee_per_contract
                    equity += pnl
                    pos -= 1
                    trades.append(_trade(entry_time, _idx[i], entry_px, runner_px, 1, 1, pnl, "Runner TP"))
                    pos = 0; direction = 0; tp1_filled = False

        elif pos < 0 and direction == -1:
            # Short: check SL first
            if h >= stop_px:
                pnl = _calc_pnl(pos, -1, entry_px, stop_px, pt_val, fee_per_contract)
                equity += pnl
                trades.append(_trade(entry_time, _idx[i], entry_px, stop_px, -1, abs(pos), pnl,
                                     "SL" if not tp1_filled else "Runner SL"))
                pos = 0; direction = 0; tp1_filled = False
            else:
                # TP1
                if not tp1_filled and l <= tp1_px:
                    tp1_filled = True
                    pnl = (entry_px - tp1_px) * 1 * pt_val - fee_per_contract
                    equity += pnl
                    pos += 1
                    trades.append(_trade(entry_time, _idx[i], entry_px, tp1_px, -1, 1, pnl, "TP1"))

                # Runner TP
                if pos < 0 and l <= runner_px:
                    pnl = (entry_px - runner_px) * 1 * pt_val - fee_per_contract
                    equity += pnl
                    pos += 1
                    trades.append(_trade(entry_time, _idx[i], entry_px, runner_px, -1, 1, pnl, "Runner TP"))
                    pos = 0; direction = 0; tp1_filled = False

        # ── Entry logic ──────────────────────────────────────────────────────
        if pos == 0:
            long_signal  = (in_session and uptrend and rbr_found
                            and ema_touch_bull and vol_spike and bull_confirm)
            short_signal = (in_session and downtrend and dbd_found
                            and ema_touch_bear and vol_spike and bear_confirm)

            if long_signal:
                entry_px   = c
                pos        = contracts
                direction  = 1
                stop_px    = c - fixed_risk_pts
                tp1_px     = c + fixed_risk_pts * first_rr
                runner_px  = c + fixed_risk_pts * runner_rr
                tp1_filled = False
                entry_time = _idx[i]
                equity -= contracts * fee_per_contract

            elif short_signal:
                entry_px   = c
                pos        = -contracts
                direction  = -1
                stop_px    = c + fixed_risk_pts
                tp1_px     = c - fixed_risk_pts * first_rr
                runner_px  = c - fixed_risk_pts * runner_rr
                tp1_filled = False
                entry_time = _idx[i]
                equity -= contracts * fee_per_contract

        equity_curve.append(equity)
        dates.append(_idx[i])

    # ── Close open position at end of data ───────────────────────────────────
    if pos != 0:
        c = close_arr[-1]
        pnl = _calc_pnl(pos, direction, entry_px, c, pt_val, fee_per_contract)
        equity += pnl
        trades.append(_trade(entry_time, df.index[-1], entry_px, c, direction, abs(pos), pnl, "End of Data"))
        equity_curve[-1] = equity

    eq = pd.Series(equity_curve, index=dates)
    stats = compute_stats(trades, eq, capital)
    return {"trades": trades, "equity": eq, "stats": stats, "params": params}


def _calc_pnl(pos, direction, entry_px, exit_px, pt_val, fee_per_ct):
    cost = abs(pos) * fee_per_ct
    if direction == 1:
        return (exit_px - entry_px) * abs(pos) * pt_val - cost
    else:
        return (entry_px - exit_px) * abs(pos) * pt_val - cost


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
        "entry_type":  "Tokyo RBR",
        "tp1_hit":     reason in ("TP1", "Runner TP", "Runner SL"),
        "tp2_hit":     reason in ("Runner TP", "Runner SL"),
    }
