"""
MNQ London Rally-Base-Rally (1min) — 6 Contracts: TP1 + Runner
════════════════════════════════════════════════════════════════
  - London session 08:00-10:00 GMT/BST
  - EMA 8/20 trend filter
  - Volume spike filter (bar vol >= multiplier * SMA)
  - Rally-Base-Rally (bullish) / Drop-Base-Drop (bearish) pattern
  - EMA touch confirmation
  - 6 contracts: TP1 3ct (1.4x risk), Runner 3ct (3.1x risk)
  - Fixed risk stop-loss at 30 pts
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytz
from strategy import compute_stats, ema


def run_backtest(df: pd.DataFrame, params: dict) -> dict:
    # ── Unpack parameters ────────────────────────────────────────────────────
    fast_len          = int(params.get("fast_len", 8))
    slow_len          = int(params.get("slow_len", 20))

    vol_lookback      = int(params.get("vol_lookback", 20))
    vol_multiplier    = float(params.get("vol_multiplier", 2.0))

    rbr_body_ratio    = float(params.get("rbr_body_ratio", 0.6))
    base_doji_ratio   = float(params.get("base_doji_ratio", 0.4))
    max_base_bars     = int(params.get("max_base_bars", 3))
    rbr_lookback      = int(params.get("rbr_lookback", 40))

    fixed_risk_points = float(params.get("fixed_risk_points", 30.0))
    first_rr_ratio    = float(params.get("first_rr_ratio", 1.5))
    runner_rr_ratio   = float(params.get("runner_rr_ratio", 3.0))
    contracts         = int(params.get("contracts_per_trade", 6))
    tp1_contracts     = int(params.get("tp1_contracts", 3))

    capital           = float(params.get("initial_capital", 50000.0))
    pt_val            = float(params.get("point_value", 2.0))
    fee_per_contract  = float(params.get("fee_per_contract", 0.62))

    # ── Compute indicators ───────────────────────────────────────────────────
    close_s  = df["close"]
    ema_fast = ema(close_s, fast_len).values.astype(float)
    ema_slow = ema(close_s, slow_len).values.astype(float)

    vol_arr  = df["volume"].values.astype(float)
    vol_sma  = pd.Series(vol_arr).rolling(vol_lookback).mean().values

    open_arr  = df["open"].values.astype(float)
    close_arr = df["close"].values.astype(float)
    high_arr  = df["high"].values.astype(float)
    low_arr   = df["low"].values.astype(float)

    n = len(df)

    # ── Pre-compute body ratio array ────────────────────────────────────────
    rng_arr = high_arr - low_arr
    body_abs_arr = np.abs(close_arr - open_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        body_ratio_arr = np.where(rng_arr == 0, 0.0, body_abs_arr / rng_arr)

    # ── Pre-compute timezone arrays ─────────────────────────────────────────
    _idx = df.index
    london_tz = pytz.timezone("Europe/London")
    if _idx.tz is None:
        _ldn_index = _idx.tz_localize("UTC").tz_convert(london_tz)
    else:
        _ldn_index = _idx.tz_convert(london_tz)
    hours_arr = _ldn_index.hour.values
    minutes_arr = _ldn_index.minute.values

    # ── Trade loop ───────────────────────────────────────────────────────────
    equity       = capital
    trades       = []
    equity_curve = []
    dates        = []

    # Position state
    pos          = 0
    entry_px     = 0.0
    direction    = 0
    stop_px      = 0.0
    tp1_px       = 0.0
    runner_px    = 0.0
    tp1_filled   = False
    entry_time   = None

    for i in range(n):
        c   = close_arr[i]
        h   = high_arr[i]
        l   = low_arr[i]
        o   = open_arr[i]
        v   = vol_arr[i]

        ldn_hour = hours_arr[i]
        ldn_min  = minutes_arr[i]

        # Session: 08:00 - 10:00 London time
        in_session = (8 <= ldn_hour < 10)

        # ── Exit logic (bar-by-bar) ──────────────────────────────────────────
        if pos > 0 and direction == 1:
            if l <= stop_px:
                exit_pnl = _calc_pnl(pos, 1, entry_px, stop_px, pt_val, fee_per_contract)
                equity += exit_pnl
                trades.append(_trade(entry_time, _idx[i], entry_px, stop_px, 1, pos, exit_pnl,
                                     "SL" if not tp1_filled else "Runner SL"))
                pos = 0; direction = 0; tp1_filled = False
            else:
                if not tp1_filled and h >= tp1_px:
                    tp1_filled = True
                    pnl = (tp1_px - entry_px) * tp1_contracts * pt_val - tp1_contracts * fee_per_contract
                    equity += pnl
                    pos -= tp1_contracts
                    trades.append(_trade(entry_time, _idx[i], entry_px, tp1_px, 1, tp1_contracts, pnl, "TP1"))

                if pos > 0 and h >= runner_px:
                    pnl = (runner_px - entry_px) * pos * pt_val - pos * fee_per_contract
                    equity += pnl
                    trades.append(_trade(entry_time, _idx[i], entry_px, runner_px, 1, pos, pnl, "Runner TP"))
                    pos = 0; direction = 0; tp1_filled = False

        elif pos < 0 and direction == -1:
            if h >= stop_px:
                exit_pnl = _calc_pnl(pos, -1, entry_px, stop_px, pt_val, fee_per_contract)
                equity += exit_pnl
                trades.append(_trade(entry_time, _idx[i], entry_px, stop_px, -1, abs(pos), exit_pnl,
                                     "SL" if not tp1_filled else "Runner SL"))
                pos = 0; direction = 0; tp1_filled = False
            else:
                if not tp1_filled and l <= tp1_px:
                    tp1_filled = True
                    pnl = (entry_px - tp1_px) * tp1_contracts * pt_val - tp1_contracts * fee_per_contract
                    equity += pnl
                    pos += tp1_contracts
                    trades.append(_trade(entry_time, _idx[i], entry_px, tp1_px, -1, tp1_contracts, pnl, "TP1"))

                if pos < 0 and l <= runner_px:
                    pnl = (entry_px - runner_px) * abs(pos) * pt_val - abs(pos) * fee_per_contract
                    equity += pnl
                    trades.append(_trade(entry_time, _idx[i], entry_px, runner_px, -1, abs(pos), pnl, "Runner TP"))
                    pos = 0; direction = 0; tp1_filled = False

        # ── Entry logic ──────────────────────────────────────────────────────
        if pos == 0 and in_session and i >= rbr_lookback:
            ef = ema_fast[i]
            es = ema_slow[i]

            if np.isnan(ef) or np.isnan(es):
                equity_curve.append(equity)
                dates.append(_idx[i])
                continue

            uptrend   = ef > es
            downtrend = ef < es

            cur_vol_sma = vol_sma[i] if not np.isnan(vol_sma[i]) else 0.0
            vol_spike   = cur_vol_sma > 0 and v >= cur_vol_sma * vol_multiplier

            bull_confirm = c > o
            bear_confirm = c < o

            ema_top = max(ef, es)
            ema_bot = min(ef, es)

            ema_touch_bull = l <= ema_top and c > ema_top
            ema_touch_bear = h >= ema_bot and c < ema_bot

            # ── RBR detection (bullish) ──────────────────────────────────────
            rbr_found = False
            for base_count in range(1, max_base_bars + 1):
                rally1_idx = i - (base_count + 1)
                if rally1_idx < 0:
                    continue
                if (base_count + 1) >= rbr_lookback:
                    continue

                r1_ratio = body_ratio_arr[rally1_idx]
                if r1_ratio >= rbr_body_ratio and close_arr[rally1_idx] > open_arr[rally1_idx]:
                    all_base = True
                    base_hi  = -np.inf
                    base_lo  = np.inf
                    for b_offset in range(1, base_count + 1):
                        b_idx = i - b_offset
                        if body_ratio_arr[b_idx] > base_doji_ratio:
                            all_base = False
                            break
                        base_hi = max(base_hi, high_arr[b_idx])
                        base_lo = min(base_lo, low_arr[b_idx])

                    if all_base and c > base_hi:
                        rbr_found = True
                        break

            # ── DBD detection (bearish) ──────────────────────────────────────
            dbd_found = False
            for base_count in range(1, max_base_bars + 1):
                drop1_idx = i - (base_count + 1)
                if drop1_idx < 0:
                    continue
                if (base_count + 1) >= rbr_lookback:
                    continue

                d1_ratio = body_ratio_arr[drop1_idx]
                if d1_ratio >= rbr_body_ratio and close_arr[drop1_idx] < open_arr[drop1_idx]:
                    all_base = True
                    base_hi  = -np.inf
                    base_lo  = np.inf
                    for b_offset in range(1, base_count + 1):
                        b_idx = i - b_offset
                        if body_ratio_arr[b_idx] > base_doji_ratio:
                            all_base = False
                            break
                        base_hi = max(base_hi, high_arr[b_idx])
                        base_lo = min(base_lo, low_arr[b_idx])

                    if all_base and c < base_lo:
                        dbd_found = True
                        break

            # ── Long entry ───────────────────────────────────────────────────
            long_signal = (in_session and uptrend and rbr_found
                           and ema_touch_bull and vol_spike and bull_confirm)

            if long_signal:
                entry_px   = c
                pos        = contracts
                direction  = 1
                tp1_filled = False
                entry_time = _idx[i]
                stop_px    = c - fixed_risk_points
                tp1_px     = c + fixed_risk_points * first_rr_ratio
                runner_px  = c + fixed_risk_points * runner_rr_ratio
                equity    -= contracts * fee_per_contract

            # ── Short entry ──────────────────────────────────────────────────
            short_signal = (in_session and downtrend and dbd_found
                            and ema_touch_bear and vol_spike and bear_confirm)

            if not long_signal and short_signal:
                entry_px   = c
                pos        = -contracts
                direction  = -1
                tp1_filled = False
                entry_time = _idx[i]
                stop_px    = c + fixed_risk_points
                tp1_px     = c - fixed_risk_points * first_rr_ratio
                runner_px  = c - fixed_risk_points * runner_rr_ratio
                equity    -= contracts * fee_per_contract

        equity_curve.append(equity)
        dates.append(_idx[i])

    # Close open position at end of data
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
        "entry_type":  "RBR London 1min MNQ",
        "tp1_hit":     reason in ("TP1", "Runner TP", "Runner SL"),
        "tp2_hit":     reason in ("Runner TP",),
    }
