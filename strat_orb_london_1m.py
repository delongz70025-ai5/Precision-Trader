"""
MNQ Opening Range Breakout — London Session 1-MIN — 3 Contracts: TP1, TP2, Runner
════════════════════════════════════════════════════════════════════════════════
  - Opening range: London 8:00-8:14 (Europe/London)
  - Breakout entry on close above/below range after 8:15
  - 3 contracts split 1/1/1: TP1=19pts, TP2=20pts, Runner trail=2pts
  - BE trigger=19pts, Max stop=130pts, Range width: 49-275pts
  - No volume/ATR filters
  - One trade per day, no re-entry after a loss
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytz
from strategy import compute_stats, atr as calc_atr


def run_backtest(df: pd.DataFrame, params: dict) -> dict:
    # ── Unpack parameters ────────────────────────────────────────────────────
    tp1_pts       = float(params.get("first_tp_points", 19.0))
    tp2_pts       = float(params.get("second_tp_points", 20.0))
    trail_dist    = float(params.get("trail_distance_points", 2.0))
    runner_be_pts = float(params.get("runner_be_trigger_points", 15.0))
    contracts     = int(params.get("contracts_per_trade", 3))
    max_stop_pts  = float(params.get("max_stop_points", 130.0))
    min_range_w   = float(params.get("min_range_width", 50.0))
    max_range_w   = float(params.get("max_range_width", 275.0))

    flat_by_hr    = int(params.get("flatten_hour", 16))   # London 16:00

    capital       = float(params.get("initial_capital", 50000.0))
    pt_val        = float(params.get("point_value", 2.0))
    fee_per_contract = float(params.get("fee_per_contract", 0.62))

    # ── Compute indicators ───────────────────────────────────────────────────
    close_arr  = df["close"].values.astype(float)
    high_arr   = df["high"].values.astype(float)
    low_arr    = df["low"].values.astype(float)

    # ── Pre-compute timezone arrays (fast: avoids per-bar datetime calls) ────
    _idx = df.index
    _tz = pytz.timezone("Europe/London")
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

        ldn_hour = hours_arr[i]
        ldn_min  = minutes_arr[i]

        # ── New day reset ────────────────────────────────────────────────────
        if prev_date is None or bar_date != prev_date:
            range_high    = np.nan
            range_low     = np.nan
            traded_today  = False
            lost_today    = False
            breakout_seen = False
            runner_trail  = np.nan
            prev_date     = bar_date

        # ── Opening range: London 8:00-8:14 ─────────────────────────────────
        in_or = (ldn_hour == 8 and 0 <= ldn_min <= 14)
        if in_or:
            range_high = h if np.isnan(range_high) else max(range_high, h)
            range_low  = l if np.isnan(range_low)  else min(range_low, l)

        after_range   = (ldn_hour > 8) or (ldn_hour == 8 and ldn_min >= 15)
        entry_allowed = ldn_hour < flat_by_hr
        should_flat   = ldn_hour >= flat_by_hr

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
        range_ready = not np.isnan(range_high) and not np.isnan(range_low)
        range_width = (range_high - range_low) if range_ready else 0.0
        range_w_ok  = range_ready and min_range_w <= range_width <= max_range_w

        filters_ok  = range_w_ok

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
        "entry_type":  "ORB London 1min MNQ",
        "tp1_hit":     reason in ("TP1", "TP2", "Runner SL"),
        "tp2_hit":     reason in ("TP2", "Runner SL"),
    }
