"""
MNQ London Rally-Base-Rally (1H) — 6 Contracts: TP1 (3), TP2/Runner (3)
════════════════════════════════════════════════════════════════════════
Converted from PineScript. Exact same logic:
  - London session (08:00-13:00 Europe/London)
  - EMA 9/21 trend filter
  - Volume spike filter
  - Rally-Base-Rally (bullish) / Drop-Base-Drop (bearish) pattern
  - EMA touch confirmation
  - Fixed-risk stops and two-tier targets
  - 6 contracts split 3/3 across TP1 and TP2
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytz
from strategy import compute_stats


LONDON_TZ = pytz.timezone("Europe/London")


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

    fixed_risk_pts    = float(params.get("fixed_risk_points", 70.0))
    first_rr          = float(params.get("first_rr_ratio", 1.4))
    runner_rr         = float(params.get("runner_rr_ratio", 3.1))
    contracts         = int(params.get("contracts_per_trade", 6))

    capital           = float(params.get("initial_capital", 50000.0))
    pt_val            = float(params.get("point_value", 2.0))
    fee_per_contract  = float(params.get("fee_per_contract", 0.62))

    # Split contracts: first half TP1, second half TP2/runner
    tp1_qty = contracts // 2
    tp2_qty = contracts - tp1_qty

    # ── Compute indicators ───────────────────────────────────────────────────
    ema_f = df["close"].ewm(span=fast_len, adjust=False).mean().values.astype(float)
    ema_s = df["close"].ewm(span=slow_len, adjust=False).mean().values.astype(float)
    vol_sma = df["volume"].rolling(vol_lookback).mean().values.astype(float)

    close_arr  = df["close"].values.astype(float)
    open_arr   = df["open"].values.astype(float)
    high_arr   = df["high"].values.astype(float)
    low_arr    = df["low"].values.astype(float)
    vol_arr    = df["volume"].values.astype(float)

    # ── Pre-compute London session flags ─────────────────────────────────────
    _idx = df.index
    if _idx.tz is None:
        _tz_index = _idx.tz_localize("UTC").tz_convert(LONDON_TZ)
    else:
        _tz_index = _idx.tz_convert(LONDON_TZ)
    hours_arr = _tz_index.hour.values
    # in_session: hour >= 8 and hour < 13
    in_session = (hours_arr >= 8) & (hours_arr < 13)

    # ── Pre-compute body ratio array ────────────────────────────────────────
    rng_arr = high_arr - low_arr
    body_abs_arr = np.abs(close_arr - open_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        body_ratio_arr = np.where(rng_arr == 0, 0.0, body_abs_arr / rng_arr)

    # ── Trade loop ───────────────────────────────────────────────────────────
    equity       = capital
    trades       = []
    equity_curve = []
    dates        = []

    # Position state
    pos          = 0        # remaining contracts (positive=long, negative=short)
    direction    = 0        # 1=long, -1=short
    entry_px     = 0.0
    entry_time   = None
    sl_px        = 0.0
    tp1_px       = 0.0
    tp2_px       = 0.0
    tp1_filled   = False
    tp2_filled   = False

    n = len(df)
    for i in range(n):
        c = close_arr[i]
        o = open_arr[i]
        h = high_arr[i]
        l = low_arr[i]
        v = vol_arr[i]
        sess = in_session[i]

        # ── Exit logic (process before entry) ────────────────────────────────
        if pos > 0 and direction == 1:
            # TP1: high touches tp1 level
            if not tp1_filled and h >= tp1_px:
                tp1_filled = True
                pnl = (tp1_px - entry_px) * tp1_qty * pt_val - tp1_qty * fee_per_contract
                equity += pnl
                pos -= tp1_qty
                trades.append(_trade(entry_time, _idx[i], entry_px, tp1_px, 1, tp1_qty, pnl, "TP1"))

            # TP2/Runner: high touches tp2 level
            if pos > 0 and not tp2_filled and h >= tp2_px:
                tp2_filled = True
                pnl = (tp2_px - entry_px) * pos * pt_val - pos * fee_per_contract
                equity += pnl
                trades.append(_trade(entry_time, _idx[i], entry_px, tp2_px, 1, pos, pnl, "TP2"))
                pos = 0
                direction = 0

            # Stop loss on remaining contracts
            if pos > 0 and l <= sl_px:
                pnl = (sl_px - entry_px) * pos * pt_val - pos * fee_per_contract
                equity += pnl
                trades.append(_trade(entry_time, _idx[i], entry_px, sl_px, 1, pos, pnl, "SL"))
                pos = 0
                direction = 0

        elif pos < 0 and direction == -1:
            # TP1: low touches tp1 level
            if not tp1_filled and l <= tp1_px:
                tp1_filled = True
                pnl = (entry_px - tp1_px) * tp1_qty * pt_val - tp1_qty * fee_per_contract
                equity += pnl
                pos += tp1_qty
                trades.append(_trade(entry_time, _idx[i], entry_px, tp1_px, -1, tp1_qty, pnl, "TP1"))

            # TP2/Runner: low touches tp2 level
            if pos < 0 and not tp2_filled and l <= tp2_px:
                tp2_filled = True
                remaining = abs(pos)
                pnl = (entry_px - tp2_px) * remaining * pt_val - remaining * fee_per_contract
                equity += pnl
                trades.append(_trade(entry_time, _idx[i], entry_px, tp2_px, -1, remaining, pnl, "TP2"))
                pos = 0
                direction = 0

            # Stop loss on remaining contracts
            if pos < 0 and h >= sl_px:
                remaining = abs(pos)
                pnl = (entry_px - sl_px) * remaining * pt_val - remaining * fee_per_contract
                equity += pnl
                trades.append(_trade(entry_time, _idx[i], entry_px, sl_px, -1, remaining, pnl, "SL"))
                pos = 0
                direction = 0

        # ── Entry logic ──────────────────────────────────────────────────────
        if pos == 0 and sess and i >= rbr_lookback:
            # Trend
            uptrend   = ema_f[i] > ema_s[i]
            downtrend = ema_f[i] < ema_s[i]

            # Volume spike
            cur_vol_sma = vol_sma[i] if not np.isnan(vol_sma[i]) else 0.0
            vol_spike = cur_vol_sma > 0 and v >= cur_vol_sma * vol_multiplier

            # Candle confirm
            bull_confirm = c > o
            bear_confirm = c < o

            # EMA touch
            ema_top = max(ema_f[i], ema_s[i])
            ema_bot = min(ema_f[i], ema_s[i])
            ema_touch_bull = l <= ema_top and c > ema_top
            ema_touch_bear = h >= ema_bot and c < ema_bot

            # ── RBR detection (bullish) ──────────────────────────────────────
            rbr_found = False
            if uptrend and vol_spike and bull_confirm and ema_touch_bull:
                for base_count in range(1, max_base_bars + 1):
                    rally1_idx = i - (base_count + 1)
                    if rally1_idx < 0:
                        continue
                    if (base_count + 1) >= rbr_lookback:
                        continue
                    # Rally1 bar: strong bullish candle
                    if body_ratio_arr[rally1_idx] >= rbr_body_ratio and close_arr[rally1_idx] > open_arr[rally1_idx]:
                        # Check all base candles (bars between rally1 and current)
                        all_base = True
                        base_hi = 0.0
                        base_lo = 1e8
                        for b_offset in range(1, base_count + 1):
                            b_idx = i - b_offset
                            if not (body_ratio_arr[b_idx] <= base_doji_ratio):
                                all_base = False
                            base_hi = max(base_hi, high_arr[b_idx])
                            base_lo = min(base_lo, low_arr[b_idx])

                        if all_base and c > base_hi:
                            rbr_found = True
                            break

                if rbr_found:
                    entry_px   = c
                    pos        = contracts
                    direction  = 1
                    entry_time = _idx[i]
                    sl_px      = c - fixed_risk_pts
                    tp1_px     = c + fixed_risk_pts * first_rr
                    tp2_px     = c + fixed_risk_pts * runner_rr
                    tp1_filled = False
                    tp2_filled = False
                    # Entry commission already included in exit pnl calc
                    equity -= contracts * fee_per_contract

            # ── DBD detection (bearish) ──────────────────────────────────────
            if pos == 0 and downtrend and vol_spike and bear_confirm and ema_touch_bear:
                dbd_found = False
                for base_count in range(1, max_base_bars + 1):
                    drop1_idx = i - (base_count + 1)
                    if drop1_idx < 0:
                        continue
                    if (base_count + 1) >= rbr_lookback:
                        continue
                    # Drop1 bar: strong bearish candle
                    if body_ratio_arr[drop1_idx] >= rbr_body_ratio and close_arr[drop1_idx] < open_arr[drop1_idx]:
                        all_base = True
                        base_hi = 0.0
                        base_lo = 1e8
                        for b_offset in range(1, base_count + 1):
                            b_idx = i - b_offset
                            if not (body_ratio_arr[b_idx] <= base_doji_ratio):
                                all_base = False
                            base_hi = max(base_hi, high_arr[b_idx])
                            base_lo = min(base_lo, low_arr[b_idx])

                        if all_base and c < base_lo:
                            dbd_found = True
                            break

                if dbd_found:
                    entry_px   = c
                    pos        = -contracts
                    direction  = -1
                    entry_time = _idx[i]
                    sl_px      = c + fixed_risk_pts
                    tp1_px     = c - fixed_risk_pts * first_rr
                    tp2_px     = c - fixed_risk_pts * runner_rr
                    tp1_filled = False
                    tp2_filled = False
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
        return (exit_px - entry_px) * pos * pt_val - cost
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
        "entry_type":  "London RBR",
        "tp1_hit":     reason in ("TP1", "TP2", "Runner SL"),
        "tp2_hit":     reason in ("TP2", "Runner SL"),
    }
