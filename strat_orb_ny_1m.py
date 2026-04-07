"""
MNQ Opening Range Breakout — NY Session 1-MIN — 3 Contracts: TP1, TP2, Runner
════════════════════════════════════════════════════════════════════════════════
  - Opening range: NY 9:30-9:44 (America/New_York)
  - Breakout entry on close above/below range after 9:45
  - 3 contracts split 1/1/1: TP1=25pts, TP2=35pts, Runner trail=30pts
  - BE trigger=35pts, Max stop=145pts, Range width: 90-340pts
  - Volume filter: SMA(20), mult 0.5
  - ATR filter: length 12, avg lookback 40, min mult 0.9, max mult 2.0
  - One trade per day, no re-entry after a loss
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytz
from strategy import compute_stats, atr as calc_atr


def run_backtest(df: pd.DataFrame, params: dict) -> dict:
    # ── Unpack parameters ────────────────────────────────────────────────────
    tp1_pts       = float(params.get("first_tp_points", 25.0))
    tp2_pts       = float(params.get("second_tp_points", 35.0))
    trail_dist    = float(params.get("trail_distance_points", 30.0))
    runner_be_pts = float(params.get("runner_be_trigger_points", 35.0))
    contracts     = int(params.get("contracts_per_trade", 3))
    max_stop_pts  = float(params.get("max_stop_points", 145.0))
    min_range_w   = float(params.get("min_range_width", 90.0))
    max_range_w   = float(params.get("max_range_width", 340.0))

    last_entry_hr = int(params.get("last_entry_hour", 15))
    flat_by_hr    = int(params.get("flatten_hour", 16))

    use_vol_filt  = bool(params.get("use_volume_filter", 1))
    vol_lookback  = int(params.get("volume_lookback", 20))
    vol_mult      = float(params.get("volume_multiplier", 0.5))

    use_atr_filt  = bool(params.get("use_atr_filter", 1))
    atr_len       = int(params.get("atr_length", 12))
    atr_avg_lb    = int(params.get("atr_avg_lookback", 40))
    min_atr_mult  = float(params.get("min_atr_multiplier", 1.0))
    max_atr_mult  = float(params.get("max_atr_multiplier", 2.0))

    capital       = float(params.get("initial_capital", 50000.0))
    pt_val        = float(params.get("point_value", 2.0))
    fee_per_contract = float(params.get("fee_per_contract", 0.62))

    # ── Compute indicators ───────────────────────────────────────────────────
    atr_series = calc_atr(df["high"], df["low"], df["close"], atr_len)
    atr_arr    = atr_series.values.astype(float)
    atr_sma    = atr_series.rolling(atr_avg_lb).mean().values.astype(float)

    vol_arr    = df["volume"].values.astype(float)
    vol_sma    = pd.Series(vol_arr).rolling(vol_lookback).mean().values

    close_arr  = df["close"].values.astype(float)
    high_arr   = df["high"].values.astype(float)
    low_arr    = df["low"].values.astype(float)

    # ── Pre-compute timezone arrays (fast: avoids per-bar datetime calls) ────
    _idx = df.index
    _tz = pytz.timezone("America/New_York")
    _tz_index = _idx.tz_convert(_tz)
    hours_arr   = _tz_index.hour.values
    minutes_arr = _tz_index.minute.values
    dates_arr   = _tz_index.date

    # ── Trade loop ───────────────────────────────────────────────────────────
    equity       = capital
    trades       = []
    equity_curve = []
    date_list    = []

    # Daily state
    range_high      = np.nan
    range_low       = np.nan
    traded_today    = False
    lost_today      = False
    breakout_seen   = False
    prev_date       = None

    # Position state
    pos             = 0
    entry_px        = 0.0
    direction       = 0
    tp1_filled      = False
    tp2_filled      = False
    runner_trail    = np.nan
    capped_stop     = np.nan
    entry_time      = None

    n = len(df)
    for i in range(n):
        bar_date = dates_arr[i]
        c = close_arr[i]
        h = high_arr[i]
        l = low_arr[i]
        v = vol_arr[i]

        ny_hour = hours_arr[i]
        ny_min  = minutes_arr[i]

        # ── New day reset ────────────────────────────────────────────────────
        if prev_date is None or bar_date != prev_date:
            range_high    = np.nan
            range_low     = np.nan
            traded_today  = False
            lost_today    = False
            breakout_seen = False
            runner_trail  = np.nan
            prev_date     = bar_date

        # ── Opening range: NY 9:30-9:44 ─────────────────────────────────────
        in_or = (ny_hour == 9 and 30 <= ny_min <= 44)
        if in_or:
            range_high = h if np.isnan(range_high) else max(range_high, h)
            range_low  = l if np.isnan(range_low)  else min(range_low, l)

        after_range   = (ny_hour > 9) or (ny_hour == 9 and ny_min >= 45)
        entry_allowed = ny_hour < last_entry_hr
        should_flat   = ny_hour >= flat_by_hr

        # ── EOD flatten ──────────────────────────────────────────────────────
        if should_flat and pos != 0:
            exit_pnl = _calc_pnl(pos, direction, entry_px, c, pt_val, fee_per_contract)
            equity += exit_pnl
            trades.append(_trade(entry_time, _idx[i], entry_px, c, direction, abs(pos), exit_pnl, "EOD Flatten"))
            if exit_pnl < 0:
                lost_today = True
            pos = 0; direction = 0; tp1_filled = False; tp2_filled = False
            runner_trail = np.nan

        # ── Filters ──────────────────────────────────────────────────────────
        cur_atr     = atr_arr[i]     if i < len(atr_arr) and not np.isnan(atr_arr[i]) else 0.0
        cur_atr_sma = atr_sma[i]     if i < len(atr_sma) and not np.isnan(atr_sma[i]) else 0.0
        cur_vol_sma = vol_sma[i]     if i < len(vol_sma) and not np.isnan(vol_sma[i]) else 0.0

        vol_ok = (not use_vol_filt) or (cur_vol_sma > 0 and v >= cur_vol_sma * vol_mult)
        atr_ok = (not use_atr_filt) or (cur_atr_sma > 0 and cur_atr >= cur_atr_sma * min_atr_mult and cur_atr <= cur_atr_sma * max_atr_mult)

        range_ready = not np.isnan(range_high) and not np.isnan(range_low)
        range_width = (range_high - range_low) if range_ready else 0.0
        range_w_ok  = range_ready and min_range_w <= range_width <= max_range_w

        filters_ok  = vol_ok and atr_ok and range_w_ok

        # ── Exit logic (long) ────────────────────────────────────────────────
        if pos > 0 and direction == 1 and not should_flat:
            long_stop = max(range_low, entry_px - max_stop_pts) if not np.isnan(range_low) else entry_px - max_stop_pts
            runner_stop = long_stop

            # TP1
            if not tp1_filled and h >= entry_px + tp1_pts:
                tp1_filled = True
                pnl = (tp1_pts) * 1 * pt_val - fee_per_contract
                equity += pnl
                pos -= 1
                trades.append(_trade(entry_time, _idx[i], entry_px, entry_px + tp1_pts, 1, 1, pnl, "TP1"))

            # TP2
            if pos > 0 and not tp2_filled and h >= entry_px + tp2_pts:
                tp2_filled = True
                pnl = (tp2_pts) * 1 * pt_val - fee_per_contract
                equity += pnl
                pos -= 1
                trades.append(_trade(entry_time, _idx[i], entry_px, entry_px + tp2_pts, 1, 1, pnl, "TP2"))

            # Runner trailing stop
            if pos > 0 and tp2_filled:
                new_trail = h - trail_dist
                runner_trail = new_trail if np.isnan(runner_trail) else max(runner_trail, new_trail)

            if pos > 0:
                if h >= entry_px + runner_be_pts:
                    runner_stop = max(runner_stop, entry_px)
                if not np.isnan(runner_trail):
                    runner_stop = max(runner_stop, runner_trail)

                stop_to_use = runner_stop if tp2_filled else long_stop
                if l <= stop_to_use:
                    pnl = (stop_to_use - entry_px) * pos * pt_val - abs(pos) * fee_per_contract
                    equity += pnl
                    trades.append(_trade(entry_time, _idx[i], entry_px, stop_to_use, 1, pos, pnl, "SL" if not tp2_filled else "Runner SL"))
                    if pnl < 0:
                        lost_today = True
                    pos = 0; direction = 0; tp1_filled = False; tp2_filled = False; runner_trail = np.nan

        # ── Exit logic (short) ───────────────────────────────────────────────
        elif pos < 0 and direction == -1 and not should_flat:
            short_stop = min(range_high, entry_px + max_stop_pts) if not np.isnan(range_high) else entry_px + max_stop_pts
            runner_stop = short_stop

            # TP1
            if not tp1_filled and l <= entry_px - tp1_pts:
                tp1_filled = True
                pnl = (tp1_pts) * 1 * pt_val - fee_per_contract
                equity += pnl
                pos += 1
                trades.append(_trade(entry_time, _idx[i], entry_px, entry_px - tp1_pts, -1, 1, pnl, "TP1"))

            # TP2
            if pos < 0 and not tp2_filled and l <= entry_px - tp2_pts:
                tp2_filled = True
                pnl = (tp2_pts) * 1 * pt_val - fee_per_contract
                equity += pnl
                pos += 1
                trades.append(_trade(entry_time, _idx[i], entry_px, entry_px - tp2_pts, -1, 1, pnl, "TP2"))

            # Runner trailing stop
            if pos < 0 and tp2_filled:
                new_trail = l + trail_dist
                runner_trail = new_trail if np.isnan(runner_trail) else min(runner_trail, new_trail)

            if pos < 0:
                if l <= entry_px - runner_be_pts:
                    runner_stop = min(runner_stop, entry_px)
                if not np.isnan(runner_trail):
                    runner_stop = min(runner_stop, runner_trail)

                stop_to_use = runner_stop if tp2_filled else short_stop
                if h >= stop_to_use:
                    pnl = (entry_px - stop_to_use) * abs(pos) * pt_val - abs(pos) * fee_per_contract
                    equity += pnl
                    trades.append(_trade(entry_time, _idx[i], entry_px, stop_to_use, -1, abs(pos), pnl, "SL" if not tp2_filled else "Runner SL"))
                    if pnl < 0:
                        lost_today = True
                    pos = 0; direction = 0; tp1_filled = False; tp2_filled = False; runner_trail = np.nan

        # ── Entry logic ──────────────────────────────────────────────────────
        if pos == 0 and after_range and entry_allowed and not should_flat and filters_ok and not traded_today and not lost_today:
            long_break  = c > range_high
            short_break = c < range_low

            if (long_break or short_break) and not breakout_seen:
                breakout_seen = True

                if long_break:
                    entry_px   = c
                    pos        = contracts
                    direction  = 1
                    tp1_filled = False
                    tp2_filled = False
                    runner_trail = np.nan
                    entry_time = _idx[i]
                    traded_today = True
                    equity -= contracts * fee_per_contract

                elif short_break:
                    entry_px   = c
                    pos        = -contracts
                    direction  = -1
                    tp1_filled = False
                    tp2_filled = False
                    runner_trail = np.nan
                    entry_time = _idx[i]
                    traded_today = True
                    equity -= contracts * fee_per_contract

        equity_curve.append(equity)
        date_list.append(_idx[i])

    # Close open position at end
    if pos != 0:
        c = close_arr[-1]
        pnl = _calc_pnl(pos, direction, entry_px, c, pt_val, fee_per_contract)
        equity += pnl
        trades.append(_trade(entry_time, df.index[-1], entry_px, c, direction, abs(pos), pnl, "End of Data"))
        equity_curve[-1] = equity

    eq = pd.Series(equity_curve, index=date_list)
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
        "entry_type":  "ORB NY 1min MNQ",
        "tp1_hit":     reason in ("TP1", "TP2", "Runner SL"),
        "tp2_hit":     reason in ("TP2", "Runner SL"),
    }
