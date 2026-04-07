"""
MNQ Combined ALL 6 Strategies (1-min) with Direction Lock
==========================================================
  ORB NY   | ORB Tokyo  | ORB London
  RBR NY   | RBR Tokyo  | RBR London

  Direction lock: once any sub-strategy enters, all subsequent entries
  must be in the same direction until every sub-strategy is flat.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytz
from strategy import compute_stats, atr as calc_atr, ema


# ── helpers ──────────────────────────────────────────────────────────────────

def _calc_pnl(pos, direction, entry_px, exit_px, pt_val, fee_per_ct):
    cost = abs(pos) * fee_per_ct
    if direction == 1:
        return (exit_px - entry_px) * abs(pos) * pt_val - cost
    else:
        return (entry_px - exit_px) * abs(pos) * pt_val - cost


def _trade(entry_t, exit_t, entry_px, exit_px, direction, contracts,
           pnl, reason, entry_type):
    return {
        "entry_time":  entry_t,
        "exit_time":   exit_t,
        "entry_price": entry_px,
        "exit_price":  exit_px,
        "direction":   direction,
        "contracts":   contracts,
        "pnl":         pnl,
        "exit_reason": reason,
        "entry_type":  entry_type,
        "tp1_hit":     reason in ("TP1", "TP2", "Runner SL", "Runner TP"),
        "tp2_hit":     reason in ("TP2", "Runner SL", "Runner TP"),
    }


# ── main backtest ────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, params: dict) -> dict:

    # =====================================================================
    #  UNPACK PARAMETERS
    # =====================================================================

    # -- general --
    capital          = float(params.get("initial_capital", 50000.0))
    pt_val           = float(params.get("point_value", 2.0))
    fee              = float(params.get("fee_per_contract", 0.62))

    # -- ORB NY --
    orb_ny_tp1        = float(params.get("orb_ny_first_tp_points", 25.0))
    orb_ny_tp2        = float(params.get("orb_ny_second_tp_points", 35.0))
    orb_ny_trail      = float(params.get("orb_ny_trail_distance_points", 30.0))
    orb_ny_be         = float(params.get("orb_ny_runner_be_trigger_points", 35.0))
    orb_ny_cts        = int(params.get("orb_ny_contracts_per_trade", 3))
    orb_ny_max_stop   = float(params.get("orb_ny_max_stop_points", 145.0))
    orb_ny_min_rw     = float(params.get("orb_ny_min_range_width", 90.0))
    orb_ny_max_rw     = float(params.get("orb_ny_max_range_width", 340.0))
    orb_ny_last_hr    = int(params.get("orb_ny_last_entry_hour", 15))
    orb_ny_flat_hr    = int(params.get("orb_ny_flatten_hour", 16))
    orb_ny_use_vol    = bool(params.get("orb_ny_use_volume_filter", 1))
    orb_ny_vol_lb     = int(params.get("orb_ny_volume_lookback", 20))
    orb_ny_vol_mult   = float(params.get("orb_ny_volume_multiplier", 0.5))
    orb_ny_use_atr    = bool(params.get("orb_ny_use_atr_filter", 1))
    orb_ny_atr_len    = int(params.get("orb_ny_atr_length", 12))
    orb_ny_atr_avg_lb = int(params.get("orb_ny_atr_avg_lookback", 40))
    orb_ny_min_atr_m  = float(params.get("orb_ny_min_atr_multiplier", 1.0))
    orb_ny_max_atr_m  = float(params.get("orb_ny_max_atr_multiplier", 2.0))

    # -- ORB Tokyo --
    orb_tk_tp1        = float(params.get("orb_tk_first_tp_points", 20.0))
    orb_tk_tp2        = float(params.get("orb_tk_second_tp_points", 30.0))
    orb_tk_trail      = float(params.get("orb_tk_trail_distance_points", 1.0))
    orb_tk_be         = float(params.get("orb_tk_runner_be_trigger_points", 6.0))
    orb_tk_cts        = int(params.get("orb_tk_contracts_per_trade", 6))
    orb_tk_c1         = int(params.get("orb_tk_tp1_contracts", 2))
    orb_tk_c2         = int(params.get("orb_tk_tp2_contracts", 2))
    orb_tk_max_stop   = float(params.get("orb_tk_max_stop_points", 70.0))
    orb_tk_min_rw     = float(params.get("orb_tk_min_range_width", 40.0))
    orb_tk_max_rw     = float(params.get("orb_tk_max_range_width", 100.0))
    orb_tk_flat_hr    = int(params.get("orb_tk_flatten_hour", 15))

    # -- ORB London --
    orb_ld_tp1        = float(params.get("orb_ld_first_tp_points", 19.0))
    orb_ld_tp2        = float(params.get("orb_ld_second_tp_points", 20.0))
    orb_ld_trail      = float(params.get("orb_ld_trail_distance_points", 2.0))
    orb_ld_be         = float(params.get("orb_ld_runner_be_trigger_points", 15.0))
    orb_ld_cts        = int(params.get("orb_ld_contracts_per_trade", 3))
    orb_ld_max_stop   = float(params.get("orb_ld_max_stop_points", 130.0))
    orb_ld_min_rw     = float(params.get("orb_ld_min_range_width", 50.0))
    orb_ld_max_rw     = float(params.get("orb_ld_max_range_width", 275.0))
    orb_ld_flat_hr    = int(params.get("orb_ld_flatten_hour", 16))

    # -- RBR NY --
    rbr_ny_fast       = int(params.get("rbr_ny_fast_len", 8))
    rbr_ny_slow       = int(params.get("rbr_ny_slow_len", 20))
    rbr_ny_vol_lb     = int(params.get("rbr_ny_vol_lookback", 20))
    rbr_ny_vol_mult   = float(params.get("rbr_ny_vol_multiplier", 2.0))
    rbr_ny_body_r     = float(params.get("rbr_ny_rbr_body_ratio", 0.6))
    rbr_ny_doji_r     = float(params.get("rbr_ny_base_doji_ratio", 0.4))
    rbr_ny_max_bb     = int(params.get("rbr_ny_max_base_bars", 3))
    rbr_ny_lookback   = int(params.get("rbr_ny_rbr_lookback", 40))
    rbr_ny_risk       = float(params.get("rbr_ny_fixed_risk_points", 90.0))
    rbr_ny_rr1        = float(params.get("rbr_ny_first_rr_ratio", 2.0))
    rbr_ny_rr_run     = float(params.get("rbr_ny_runner_rr_ratio", 2.5))
    rbr_ny_cts        = int(params.get("rbr_ny_contracts_per_trade", 2))
    rbr_ny_flat_hr    = int(params.get("rbr_ny_flatten_hour", 16))

    # -- RBR Tokyo --
    rbr_tk_fast       = int(params.get("rbr_tk_fast_len", 8))
    rbr_tk_slow       = int(params.get("rbr_tk_slow_len", 20))
    rbr_tk_vol_lb     = int(params.get("rbr_tk_vol_lookback", 20))
    rbr_tk_vol_mult   = float(params.get("rbr_tk_vol_multiplier", 2.5))
    rbr_tk_body_r     = float(params.get("rbr_tk_rbr_body_ratio", 0.6))
    rbr_tk_doji_r     = float(params.get("rbr_tk_base_doji_ratio", 0.4))
    rbr_tk_max_bb     = int(params.get("rbr_tk_max_base_bars", 3))
    rbr_tk_lookback   = int(params.get("rbr_tk_rbr_lookback", 40))
    rbr_tk_risk       = float(params.get("rbr_tk_fixed_risk_points", 40.0))
    rbr_tk_rr1        = float(params.get("rbr_tk_first_rr_ratio", 2.5))
    rbr_tk_rr_run     = float(params.get("rbr_tk_runner_rr_ratio", 2.5))
    rbr_tk_cts        = int(params.get("rbr_tk_contracts_per_trade", 2))
    rbr_tk_flat_hr    = int(params.get("rbr_tk_flatten_hour", 15))

    # -- RBR London --
    rbr_ld_fast       = int(params.get("rbr_ld_fast_len", 8))
    rbr_ld_slow       = int(params.get("rbr_ld_slow_len", 20))
    rbr_ld_vol_lb     = int(params.get("rbr_ld_vol_lookback", 20))
    rbr_ld_vol_mult   = float(params.get("rbr_ld_vol_multiplier", 2.5))
    rbr_ld_body_r     = float(params.get("rbr_ld_rbr_body_ratio", 0.6))
    rbr_ld_doji_r     = float(params.get("rbr_ld_base_doji_ratio", 0.4))
    rbr_ld_max_bb     = int(params.get("rbr_ld_max_base_bars", 3))
    rbr_ld_lookback   = int(params.get("rbr_ld_rbr_lookback", 40))
    rbr_ld_risk       = float(params.get("rbr_ld_fixed_risk_points", 30.0))
    rbr_ld_rr1        = float(params.get("rbr_ld_first_rr_ratio", 1.5))
    rbr_ld_rr_run     = float(params.get("rbr_ld_runner_rr_ratio", 3.0))
    rbr_ld_cts        = int(params.get("rbr_ld_contracts_per_trade", 6))
    rbr_ld_tp1_cts    = int(params.get("rbr_ld_tp1_contracts", 3))
    rbr_ld_flat_hr    = int(params.get("rbr_ld_flatten_hour", 16))

    # =====================================================================
    #  PRE-COMPUTE PRICE / VOLUME / INDICATOR ARRAYS
    # =====================================================================
    _idx      = df.index
    open_arr  = df["open"].values.astype(float)
    close_arr = df["close"].values.astype(float)
    high_arr  = df["high"].values.astype(float)
    low_arr   = df["low"].values.astype(float)
    vol_arr   = df["volume"].values.astype(float)
    n         = len(df)

    # ATR for ORB NY
    atr_series    = calc_atr(df["high"], df["low"], df["close"], orb_ny_atr_len)
    atr_arr       = atr_series.values.astype(float)
    atr_sma       = atr_series.rolling(orb_ny_atr_avg_lb).mean().values.astype(float)

    # Volume SMA for ORB NY
    orb_ny_vol_sma = pd.Series(vol_arr).rolling(orb_ny_vol_lb).mean().values

    # RBR EMAs (shared close series)
    close_s = df["close"]
    rbr_ny_ema_f  = ema(close_s, rbr_ny_fast).values.astype(float)
    rbr_ny_ema_s  = ema(close_s, rbr_ny_slow).values.astype(float)
    rbr_ny_vol_sma = pd.Series(vol_arr).rolling(rbr_ny_vol_lb).mean().values

    rbr_tk_ema_f  = ema(close_s, rbr_tk_fast).values.astype(float)
    rbr_tk_ema_s  = ema(close_s, rbr_tk_slow).values.astype(float)
    rbr_tk_vol_sma = pd.Series(vol_arr).rolling(rbr_tk_vol_lb).mean().values

    rbr_ld_ema_f  = ema(close_s, rbr_ld_fast).values.astype(float)
    rbr_ld_ema_s  = ema(close_s, rbr_ld_slow).values.astype(float)
    rbr_ld_vol_sma = pd.Series(vol_arr).rolling(rbr_ld_vol_lb).mean().values

    # Body ratio for all RBR strategies
    rng_arr      = high_arr - low_arr
    body_abs_arr = np.abs(close_arr - open_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        body_ratio_arr = np.where(rng_arr == 0, 0.0, body_abs_arr / rng_arr)

    # =====================================================================
    #  PRE-COMPUTE TIMEZONE ARRAYS
    # =====================================================================
    ny_tz    = pytz.timezone("America/New_York")
    tokyo_tz = pytz.timezone("Asia/Tokyo")
    london_tz = pytz.timezone("Europe/London")

    _ny_idx = _idx.tz_convert(ny_tz)
    ny_hours = _ny_idx.hour.values
    ny_mins  = _ny_idx.minute.values
    ny_dates = _ny_idx.date

    _tk_idx = _idx.tz_convert(tokyo_tz)
    tk_hours = _tk_idx.hour.values
    tk_mins  = _tk_idx.minute.values
    tk_dates = _tk_idx.date

    _ld_idx = _idx.tz_convert(london_tz)
    ld_hours = _ld_idx.hour.values
    ld_mins  = _ld_idx.minute.values
    ld_dates = _ld_idx.date

    # =====================================================================
    #  STATE VARIABLES
    # =====================================================================
    equity       = capital
    trades       = []
    equity_curve = []
    date_list    = []

    # Direction lock
    locked_direction = 0   # 0=none, 1=long, -1=short

    # -- ORB NY state --
    orb_ny_pos = 0; orb_ny_entry_px = 0.0; orb_ny_dir = 0
    orb_ny_tp1_filled = False; orb_ny_tp2_filled = False
    orb_ny_runner_trail = np.nan; orb_ny_entry_time = None
    orb_ny_range_high = np.nan; orb_ny_range_low = np.nan
    orb_ny_traded = False; orb_ny_lost = False; orb_ny_breakout = False
    orb_ny_prev_date = None

    # -- ORB Tokyo state --
    orb_tk_pos = 0; orb_tk_entry_px = 0.0; orb_tk_dir = 0
    orb_tk_tp1_filled = False; orb_tk_tp2_filled = False
    orb_tk_runner_trail = np.nan; orb_tk_entry_time = None
    orb_tk_range_high = np.nan; orb_tk_range_low = np.nan
    orb_tk_traded = False; orb_tk_lost = False; orb_tk_breakout = False
    orb_tk_prev_date = None

    # -- ORB London state --
    orb_ld_pos = 0; orb_ld_entry_px = 0.0; orb_ld_dir = 0
    orb_ld_tp1_filled = False; orb_ld_tp2_filled = False
    orb_ld_runner_trail = np.nan; orb_ld_entry_time = None
    orb_ld_range_high = np.nan; orb_ld_range_low = np.nan
    orb_ld_traded = False; orb_ld_lost = False; orb_ld_breakout = False
    orb_ld_prev_date = None

    # -- RBR NY state --
    rbr_ny_pos = 0; rbr_ny_entry_px = 0.0; rbr_ny_dir = 0
    rbr_ny_stop = 0.0; rbr_ny_tp1_px = 0.0; rbr_ny_runner_px = 0.0
    rbr_ny_tp1_filled = False; rbr_ny_entry_time = None

    # -- RBR Tokyo state --
    rbr_tk_pos = 0; rbr_tk_entry_px = 0.0; rbr_tk_dir = 0
    rbr_tk_stop = 0.0; rbr_tk_tp1_px = 0.0; rbr_tk_runner_px = 0.0
    rbr_tk_tp1_filled = False; rbr_tk_entry_time = None

    # -- RBR London state --
    rbr_ld_pos = 0; rbr_ld_entry_px = 0.0; rbr_ld_dir = 0
    rbr_ld_stop = 0.0; rbr_ld_tp1_px = 0.0; rbr_ld_runner_px = 0.0
    rbr_ld_tp1_filled = False; rbr_ld_entry_time = None

    # min rbr lookback across all 3
    min_rbr_lb = max(rbr_ny_lookback, rbr_tk_lookback, rbr_ld_lookback)

    # =====================================================================
    #  MAIN BAR LOOP
    # =====================================================================
    for i in range(n):
        c = close_arr[i]
        h = high_arr[i]
        l = low_arr[i]
        o = open_arr[i]
        v = vol_arr[i]

        ny_h = ny_hours[i];  ny_m = ny_mins[i];  ny_d = ny_dates[i]
        tk_h = tk_hours[i];  tk_m = tk_mins[i];  tk_d = tk_dates[i]
        ld_h = ld_hours[i];  ld_m = ld_mins[i];  ld_d = ld_dates[i]

        # =================================================================
        #  ORB NY — daily reset
        # =================================================================
        if orb_ny_prev_date is None or ny_d != orb_ny_prev_date:
            orb_ny_range_high = np.nan; orb_ny_range_low = np.nan
            orb_ny_traded = False; orb_ny_lost = False; orb_ny_breakout = False
            orb_ny_runner_trail = np.nan; orb_ny_prev_date = ny_d

        # Opening range 9:30-9:44 NY
        if ny_h == 9 and 30 <= ny_m <= 44:
            orb_ny_range_high = h if np.isnan(orb_ny_range_high) else max(orb_ny_range_high, h)
            orb_ny_range_low  = l if np.isnan(orb_ny_range_low)  else min(orb_ny_range_low, l)

        orb_ny_after  = (ny_h > 9) or (ny_h == 9 and ny_m >= 45)
        orb_ny_eok    = ny_h < orb_ny_last_hr
        orb_ny_flat   = ny_h >= orb_ny_flat_hr

        # EOD flatten
        if orb_ny_flat and orb_ny_pos != 0:
            pnl = _calc_pnl(orb_ny_pos, orb_ny_dir, orb_ny_entry_px, c, pt_val, fee)
            equity += pnl
            trades.append(_trade(orb_ny_entry_time, _idx[i], orb_ny_entry_px, c,
                                 orb_ny_dir, abs(orb_ny_pos), pnl, "EOD Flatten", "ORB NY"))
            if pnl < 0: orb_ny_lost = True
            orb_ny_pos = 0; orb_ny_dir = 0; orb_ny_tp1_filled = False
            orb_ny_tp2_filled = False; orb_ny_runner_trail = np.nan

        # Filters
        cur_atr     = atr_arr[i] if not np.isnan(atr_arr[i]) else 0.0
        cur_atr_sma = atr_sma[i] if not np.isnan(atr_sma[i]) else 0.0
        cur_vol_sma_ny = orb_ny_vol_sma[i] if not np.isnan(orb_ny_vol_sma[i]) else 0.0
        vol_ok_ny = (not orb_ny_use_vol) or (cur_vol_sma_ny > 0 and v >= cur_vol_sma_ny * orb_ny_vol_mult)
        atr_ok_ny = (not orb_ny_use_atr) or (cur_atr_sma > 0 and cur_atr >= cur_atr_sma * orb_ny_min_atr_m and cur_atr <= cur_atr_sma * orb_ny_max_atr_m)
        rng_rdy_ny = not np.isnan(orb_ny_range_high) and not np.isnan(orb_ny_range_low)
        rng_w_ny   = (orb_ny_range_high - orb_ny_range_low) if rng_rdy_ny else 0.0
        rng_ok_ny  = rng_rdy_ny and orb_ny_min_rw <= rng_w_ny <= orb_ny_max_rw
        filt_ny    = vol_ok_ny and atr_ok_ny and rng_ok_ny

        # Exit long ORB NY
        if orb_ny_pos > 0 and orb_ny_dir == 1 and not orb_ny_flat:
            long_stop = max(orb_ny_range_low, orb_ny_entry_px - orb_ny_max_stop) if not np.isnan(orb_ny_range_low) else orb_ny_entry_px - orb_ny_max_stop
            runner_stop = long_stop
            if not orb_ny_tp1_filled and h >= orb_ny_entry_px + orb_ny_tp1:
                orb_ny_tp1_filled = True
                pnl = orb_ny_tp1 * 1 * pt_val - fee
                equity += pnl; orb_ny_pos -= 1
                trades.append(_trade(orb_ny_entry_time, _idx[i], orb_ny_entry_px, orb_ny_entry_px + orb_ny_tp1, 1, 1, pnl, "TP1", "ORB NY"))
            if orb_ny_pos > 0 and not orb_ny_tp2_filled and h >= orb_ny_entry_px + orb_ny_tp2:
                orb_ny_tp2_filled = True
                pnl = orb_ny_tp2 * 1 * pt_val - fee
                equity += pnl; orb_ny_pos -= 1
                trades.append(_trade(orb_ny_entry_time, _idx[i], orb_ny_entry_px, orb_ny_entry_px + orb_ny_tp2, 1, 1, pnl, "TP2", "ORB NY"))
            if orb_ny_pos > 0 and orb_ny_tp2_filled:
                nt = h - orb_ny_trail
                orb_ny_runner_trail = nt if np.isnan(orb_ny_runner_trail) else max(orb_ny_runner_trail, nt)
            if orb_ny_pos > 0:
                if h >= orb_ny_entry_px + orb_ny_be: runner_stop = max(runner_stop, orb_ny_entry_px)
                if not np.isnan(orb_ny_runner_trail): runner_stop = max(runner_stop, orb_ny_runner_trail)
                stu = runner_stop if orb_ny_tp2_filled else long_stop
                if l <= stu:
                    pnl = (stu - orb_ny_entry_px) * orb_ny_pos * pt_val - abs(orb_ny_pos) * fee
                    equity += pnl
                    trades.append(_trade(orb_ny_entry_time, _idx[i], orb_ny_entry_px, stu, 1, orb_ny_pos, pnl,
                                         "SL" if not orb_ny_tp2_filled else "Runner SL", "ORB NY"))
                    if pnl < 0: orb_ny_lost = True
                    orb_ny_pos = 0; orb_ny_dir = 0; orb_ny_tp1_filled = False; orb_ny_tp2_filled = False; orb_ny_runner_trail = np.nan

        # Exit short ORB NY
        elif orb_ny_pos < 0 and orb_ny_dir == -1 and not orb_ny_flat:
            short_stop = min(orb_ny_range_high, orb_ny_entry_px + orb_ny_max_stop) if not np.isnan(orb_ny_range_high) else orb_ny_entry_px + orb_ny_max_stop
            runner_stop = short_stop
            if not orb_ny_tp1_filled and l <= orb_ny_entry_px - orb_ny_tp1:
                orb_ny_tp1_filled = True
                pnl = orb_ny_tp1 * 1 * pt_val - fee
                equity += pnl; orb_ny_pos += 1
                trades.append(_trade(orb_ny_entry_time, _idx[i], orb_ny_entry_px, orb_ny_entry_px - orb_ny_tp1, -1, 1, pnl, "TP1", "ORB NY"))
            if orb_ny_pos < 0 and not orb_ny_tp2_filled and l <= orb_ny_entry_px - orb_ny_tp2:
                orb_ny_tp2_filled = True
                pnl = orb_ny_tp2 * 1 * pt_val - fee
                equity += pnl; orb_ny_pos += 1
                trades.append(_trade(orb_ny_entry_time, _idx[i], orb_ny_entry_px, orb_ny_entry_px - orb_ny_tp2, -1, 1, pnl, "TP2", "ORB NY"))
            if orb_ny_pos < 0 and orb_ny_tp2_filled:
                nt = l + orb_ny_trail
                orb_ny_runner_trail = nt if np.isnan(orb_ny_runner_trail) else min(orb_ny_runner_trail, nt)
            if orb_ny_pos < 0:
                if l <= orb_ny_entry_px - orb_ny_be: runner_stop = min(runner_stop, orb_ny_entry_px)
                if not np.isnan(orb_ny_runner_trail): runner_stop = min(runner_stop, orb_ny_runner_trail)
                stu = runner_stop if orb_ny_tp2_filled else short_stop
                if h >= stu:
                    pnl = (orb_ny_entry_px - stu) * abs(orb_ny_pos) * pt_val - abs(orb_ny_pos) * fee
                    equity += pnl
                    trades.append(_trade(orb_ny_entry_time, _idx[i], orb_ny_entry_px, stu, -1, abs(orb_ny_pos), pnl,
                                         "SL" if not orb_ny_tp2_filled else "Runner SL", "ORB NY"))
                    if pnl < 0: orb_ny_lost = True
                    orb_ny_pos = 0; orb_ny_dir = 0; orb_ny_tp1_filled = False; orb_ny_tp2_filled = False; orb_ny_runner_trail = np.nan

        # =================================================================
        #  ORB TOKYO — daily reset
        # =================================================================
        if orb_tk_prev_date is None or tk_d != orb_tk_prev_date:
            orb_tk_range_high = np.nan; orb_tk_range_low = np.nan
            orb_tk_traded = False; orb_tk_lost = False; orb_tk_breakout = False
            orb_tk_runner_trail = np.nan; orb_tk_prev_date = tk_d

        # Opening range 9:00-9:14 Tokyo
        if tk_h == 9 and 0 <= tk_m <= 14:
            orb_tk_range_high = h if np.isnan(orb_tk_range_high) else max(orb_tk_range_high, h)
            orb_tk_range_low  = l if np.isnan(orb_tk_range_low)  else min(orb_tk_range_low, l)

        orb_tk_after = (tk_h > 9) or (tk_h == 9 and tk_m >= 15)
        orb_tk_eok   = tk_h < orb_tk_flat_hr
        orb_tk_flat  = tk_h >= orb_tk_flat_hr

        # EOD flatten
        if orb_tk_flat and orb_tk_pos != 0:
            pnl = _calc_pnl(orb_tk_pos, orb_tk_dir, orb_tk_entry_px, c, pt_val, fee)
            equity += pnl
            trades.append(_trade(orb_tk_entry_time, _idx[i], orb_tk_entry_px, c,
                                 orb_tk_dir, abs(orb_tk_pos), pnl, "EOD Flatten", "ORB Tokyo"))
            if pnl < 0: orb_tk_lost = True
            orb_tk_pos = 0; orb_tk_dir = 0; orb_tk_tp1_filled = False
            orb_tk_tp2_filled = False; orb_tk_runner_trail = np.nan

        rng_rdy_tk = not np.isnan(orb_tk_range_high) and not np.isnan(orb_tk_range_low)
        rng_w_tk   = (orb_tk_range_high - orb_tk_range_low) if rng_rdy_tk else 0.0
        rng_ok_tk  = rng_rdy_tk and orb_tk_min_rw <= rng_w_tk <= orb_tk_max_rw
        filt_tk    = rng_ok_tk

        # Exit long ORB Tokyo
        if orb_tk_pos > 0 and orb_tk_dir == 1 and not orb_tk_flat:
            long_stop = max(orb_tk_range_low, orb_tk_entry_px - orb_tk_max_stop) if not np.isnan(orb_tk_range_low) else orb_tk_entry_px - orb_tk_max_stop
            runner_stop = long_stop
            if not orb_tk_tp1_filled and h >= orb_tk_entry_px + orb_tk_tp1:
                orb_tk_tp1_filled = True
                pnl = orb_tk_tp1 * orb_tk_c1 * pt_val - orb_tk_c1 * fee
                equity += pnl; orb_tk_pos -= orb_tk_c1
                trades.append(_trade(orb_tk_entry_time, _idx[i], orb_tk_entry_px, orb_tk_entry_px + orb_tk_tp1, 1, orb_tk_c1, pnl, "TP1", "ORB Tokyo"))
            if orb_tk_pos > 0 and not orb_tk_tp2_filled and h >= orb_tk_entry_px + orb_tk_tp2:
                orb_tk_tp2_filled = True
                pnl = orb_tk_tp2 * orb_tk_c2 * pt_val - orb_tk_c2 * fee
                equity += pnl; orb_tk_pos -= orb_tk_c2
                trades.append(_trade(orb_tk_entry_time, _idx[i], orb_tk_entry_px, orb_tk_entry_px + orb_tk_tp2, 1, orb_tk_c2, pnl, "TP2", "ORB Tokyo"))
            if orb_tk_pos > 0 and orb_tk_tp2_filled:
                nt = h - orb_tk_trail
                orb_tk_runner_trail = nt if np.isnan(orb_tk_runner_trail) else max(orb_tk_runner_trail, nt)
            if orb_tk_pos > 0:
                if h >= orb_tk_entry_px + orb_tk_be: runner_stop = max(runner_stop, orb_tk_entry_px)
                if not np.isnan(orb_tk_runner_trail): runner_stop = max(runner_stop, orb_tk_runner_trail)
                stu = runner_stop if orb_tk_tp2_filled else long_stop
                if l <= stu:
                    pnl = (stu - orb_tk_entry_px) * orb_tk_pos * pt_val - abs(orb_tk_pos) * fee
                    equity += pnl
                    trades.append(_trade(orb_tk_entry_time, _idx[i], orb_tk_entry_px, stu, 1, orb_tk_pos, pnl,
                                         "SL" if not orb_tk_tp2_filled else "Runner SL", "ORB Tokyo"))
                    if pnl < 0: orb_tk_lost = True
                    orb_tk_pos = 0; orb_tk_dir = 0; orb_tk_tp1_filled = False; orb_tk_tp2_filled = False; orb_tk_runner_trail = np.nan

        # Exit short ORB Tokyo
        elif orb_tk_pos < 0 and orb_tk_dir == -1 and not orb_tk_flat:
            short_stop = min(orb_tk_range_high, orb_tk_entry_px + orb_tk_max_stop) if not np.isnan(orb_tk_range_high) else orb_tk_entry_px + orb_tk_max_stop
            runner_stop = short_stop
            if not orb_tk_tp1_filled and l <= orb_tk_entry_px - orb_tk_tp1:
                orb_tk_tp1_filled = True
                pnl = orb_tk_tp1 * orb_tk_c1 * pt_val - orb_tk_c1 * fee
                equity += pnl; orb_tk_pos += orb_tk_c1
                trades.append(_trade(orb_tk_entry_time, _idx[i], orb_tk_entry_px, orb_tk_entry_px - orb_tk_tp1, -1, orb_tk_c1, pnl, "TP1", "ORB Tokyo"))
            if orb_tk_pos < 0 and not orb_tk_tp2_filled and l <= orb_tk_entry_px - orb_tk_tp2:
                orb_tk_tp2_filled = True
                pnl = orb_tk_tp2 * orb_tk_c2 * pt_val - orb_tk_c2 * fee
                equity += pnl; orb_tk_pos += orb_tk_c2
                trades.append(_trade(orb_tk_entry_time, _idx[i], orb_tk_entry_px, orb_tk_entry_px - orb_tk_tp2, -1, orb_tk_c2, pnl, "TP2", "ORB Tokyo"))
            if orb_tk_pos < 0 and orb_tk_tp2_filled:
                nt = l + orb_tk_trail
                orb_tk_runner_trail = nt if np.isnan(orb_tk_runner_trail) else min(orb_tk_runner_trail, nt)
            if orb_tk_pos < 0:
                if l <= orb_tk_entry_px - orb_tk_be: runner_stop = min(runner_stop, orb_tk_entry_px)
                if not np.isnan(orb_tk_runner_trail): runner_stop = min(runner_stop, orb_tk_runner_trail)
                stu = runner_stop if orb_tk_tp2_filled else short_stop
                if h >= stu:
                    pnl = (orb_tk_entry_px - stu) * abs(orb_tk_pos) * pt_val - abs(orb_tk_pos) * fee
                    equity += pnl
                    trades.append(_trade(orb_tk_entry_time, _idx[i], orb_tk_entry_px, stu, -1, abs(orb_tk_pos), pnl,
                                         "SL" if not orb_tk_tp2_filled else "Runner SL", "ORB Tokyo"))
                    if pnl < 0: orb_tk_lost = True
                    orb_tk_pos = 0; orb_tk_dir = 0; orb_tk_tp1_filled = False; orb_tk_tp2_filled = False; orb_tk_runner_trail = np.nan

        # =================================================================
        #  ORB LONDON — daily reset
        # =================================================================
        if orb_ld_prev_date is None or ld_d != orb_ld_prev_date:
            orb_ld_range_high = np.nan; orb_ld_range_low = np.nan
            orb_ld_traded = False; orb_ld_lost = False; orb_ld_breakout = False
            orb_ld_runner_trail = np.nan; orb_ld_prev_date = ld_d

        # Opening range 8:00-8:14 London
        if ld_h == 8 and 0 <= ld_m <= 14:
            orb_ld_range_high = h if np.isnan(orb_ld_range_high) else max(orb_ld_range_high, h)
            orb_ld_range_low  = l if np.isnan(orb_ld_range_low)  else min(orb_ld_range_low, l)

        orb_ld_after = (ld_h > 8) or (ld_h == 8 and ld_m >= 15)
        orb_ld_eok   = ld_h < orb_ld_flat_hr
        orb_ld_flat  = ld_h >= orb_ld_flat_hr

        # EOD flatten
        if orb_ld_flat and orb_ld_pos != 0:
            pnl = _calc_pnl(orb_ld_pos, orb_ld_dir, orb_ld_entry_px, c, pt_val, fee)
            equity += pnl
            trades.append(_trade(orb_ld_entry_time, _idx[i], orb_ld_entry_px, c,
                                 orb_ld_dir, abs(orb_ld_pos), pnl, "EOD Flatten", "ORB London"))
            if pnl < 0: orb_ld_lost = True
            orb_ld_pos = 0; orb_ld_dir = 0; orb_ld_tp1_filled = False
            orb_ld_tp2_filled = False; orb_ld_runner_trail = np.nan

        rng_rdy_ld = not np.isnan(orb_ld_range_high) and not np.isnan(orb_ld_range_low)
        rng_w_ld   = (orb_ld_range_high - orb_ld_range_low) if rng_rdy_ld else 0.0
        rng_ok_ld  = rng_rdy_ld and orb_ld_min_rw <= rng_w_ld <= orb_ld_max_rw
        filt_ld    = rng_ok_ld

        # Exit long ORB London
        if orb_ld_pos > 0 and orb_ld_dir == 1 and not orb_ld_flat:
            long_stop = max(orb_ld_range_low, orb_ld_entry_px - orb_ld_max_stop) if not np.isnan(orb_ld_range_low) else orb_ld_entry_px - orb_ld_max_stop
            runner_stop = long_stop
            if not orb_ld_tp1_filled and h >= orb_ld_entry_px + orb_ld_tp1:
                orb_ld_tp1_filled = True
                pnl = orb_ld_tp1 * 1 * pt_val - fee
                equity += pnl; orb_ld_pos -= 1
                trades.append(_trade(orb_ld_entry_time, _idx[i], orb_ld_entry_px, orb_ld_entry_px + orb_ld_tp1, 1, 1, pnl, "TP1", "ORB London"))
            if orb_ld_pos > 0 and not orb_ld_tp2_filled and h >= orb_ld_entry_px + orb_ld_tp2:
                orb_ld_tp2_filled = True
                pnl = orb_ld_tp2 * 1 * pt_val - fee
                equity += pnl; orb_ld_pos -= 1
                trades.append(_trade(orb_ld_entry_time, _idx[i], orb_ld_entry_px, orb_ld_entry_px + orb_ld_tp2, 1, 1, pnl, "TP2", "ORB London"))
            if orb_ld_pos > 0 and orb_ld_tp2_filled:
                nt = h - orb_ld_trail
                orb_ld_runner_trail = nt if np.isnan(orb_ld_runner_trail) else max(orb_ld_runner_trail, nt)
            if orb_ld_pos > 0:
                if h >= orb_ld_entry_px + orb_ld_be: runner_stop = max(runner_stop, orb_ld_entry_px)
                if not np.isnan(orb_ld_runner_trail): runner_stop = max(runner_stop, orb_ld_runner_trail)
                stu = runner_stop if orb_ld_tp2_filled else long_stop
                if l <= stu:
                    pnl = (stu - orb_ld_entry_px) * orb_ld_pos * pt_val - abs(orb_ld_pos) * fee
                    equity += pnl
                    trades.append(_trade(orb_ld_entry_time, _idx[i], orb_ld_entry_px, stu, 1, orb_ld_pos, pnl,
                                         "SL" if not orb_ld_tp2_filled else "Runner SL", "ORB London"))
                    if pnl < 0: orb_ld_lost = True
                    orb_ld_pos = 0; orb_ld_dir = 0; orb_ld_tp1_filled = False; orb_ld_tp2_filled = False; orb_ld_runner_trail = np.nan

        # Exit short ORB London
        elif orb_ld_pos < 0 and orb_ld_dir == -1 and not orb_ld_flat:
            short_stop = min(orb_ld_range_high, orb_ld_entry_px + orb_ld_max_stop) if not np.isnan(orb_ld_range_high) else orb_ld_entry_px + orb_ld_max_stop
            runner_stop = short_stop
            if not orb_ld_tp1_filled and l <= orb_ld_entry_px - orb_ld_tp1:
                orb_ld_tp1_filled = True
                pnl = orb_ld_tp1 * 1 * pt_val - fee
                equity += pnl; orb_ld_pos += 1
                trades.append(_trade(orb_ld_entry_time, _idx[i], orb_ld_entry_px, orb_ld_entry_px - orb_ld_tp1, -1, 1, pnl, "TP1", "ORB London"))
            if orb_ld_pos < 0 and not orb_ld_tp2_filled and l <= orb_ld_entry_px - orb_ld_tp2:
                orb_ld_tp2_filled = True
                pnl = orb_ld_tp2 * 1 * pt_val - fee
                equity += pnl; orb_ld_pos += 1
                trades.append(_trade(orb_ld_entry_time, _idx[i], orb_ld_entry_px, orb_ld_entry_px - orb_ld_tp2, -1, 1, pnl, "TP2", "ORB London"))
            if orb_ld_pos < 0 and orb_ld_tp2_filled:
                nt = l + orb_ld_trail
                orb_ld_runner_trail = nt if np.isnan(orb_ld_runner_trail) else min(orb_ld_runner_trail, nt)
            if orb_ld_pos < 0:
                if l <= orb_ld_entry_px - orb_ld_be: runner_stop = min(runner_stop, orb_ld_entry_px)
                if not np.isnan(orb_ld_runner_trail): runner_stop = min(runner_stop, orb_ld_runner_trail)
                stu = runner_stop if orb_ld_tp2_filled else short_stop
                if h >= stu:
                    pnl = (orb_ld_entry_px - stu) * abs(orb_ld_pos) * pt_val - abs(orb_ld_pos) * fee
                    equity += pnl
                    trades.append(_trade(orb_ld_entry_time, _idx[i], orb_ld_entry_px, stu, -1, abs(orb_ld_pos), pnl,
                                         "SL" if not orb_ld_tp2_filled else "Runner SL", "ORB London"))
                    if pnl < 0: orb_ld_lost = True
                    orb_ld_pos = 0; orb_ld_dir = 0; orb_ld_tp1_filled = False; orb_ld_tp2_filled = False; orb_ld_runner_trail = np.nan

        # =================================================================
        #  RBR NY — session flatten
        # =================================================================
        rbr_ny_flat = ny_h >= rbr_ny_flat_hr
        if rbr_ny_flat and rbr_ny_pos != 0:
            pnl = _calc_pnl(rbr_ny_pos, rbr_ny_dir, rbr_ny_entry_px, c, pt_val, fee)
            equity += pnl
            trades.append(_trade(rbr_ny_entry_time, _idx[i], rbr_ny_entry_px, c,
                                 rbr_ny_dir, abs(rbr_ny_pos), pnl, "EOD Flatten", "RBR NY"))
            rbr_ny_pos = 0; rbr_ny_dir = 0; rbr_ny_tp1_filled = False

        # =================================================================
        #  RBR NY — exit logic
        # =================================================================
        if rbr_ny_pos > 0 and rbr_ny_dir == 1:
            if l <= rbr_ny_stop:
                pnl = _calc_pnl(rbr_ny_pos, 1, rbr_ny_entry_px, rbr_ny_stop, pt_val, fee)
                equity += pnl
                trades.append(_trade(rbr_ny_entry_time, _idx[i], rbr_ny_entry_px, rbr_ny_stop, 1, rbr_ny_pos, pnl,
                                     "SL" if not rbr_ny_tp1_filled else "Runner SL", "RBR NY"))
                rbr_ny_pos = 0; rbr_ny_dir = 0; rbr_ny_tp1_filled = False
            else:
                if not rbr_ny_tp1_filled and h >= rbr_ny_tp1_px:
                    rbr_ny_tp1_filled = True
                    pnl = (rbr_ny_tp1_px - rbr_ny_entry_px) * 1 * pt_val - fee
                    equity += pnl; rbr_ny_pos -= 1
                    trades.append(_trade(rbr_ny_entry_time, _idx[i], rbr_ny_entry_px, rbr_ny_tp1_px, 1, 1, pnl, "TP1", "RBR NY"))
                if rbr_ny_pos > 0 and h >= rbr_ny_runner_px:
                    pnl = (rbr_ny_runner_px - rbr_ny_entry_px) * rbr_ny_pos * pt_val - rbr_ny_pos * fee
                    equity += pnl
                    trades.append(_trade(rbr_ny_entry_time, _idx[i], rbr_ny_entry_px, rbr_ny_runner_px, 1, rbr_ny_pos, pnl, "Runner TP", "RBR NY"))
                    rbr_ny_pos = 0; rbr_ny_dir = 0; rbr_ny_tp1_filled = False
        elif rbr_ny_pos < 0 and rbr_ny_dir == -1:
            if h >= rbr_ny_stop:
                pnl = _calc_pnl(rbr_ny_pos, -1, rbr_ny_entry_px, rbr_ny_stop, pt_val, fee)
                equity += pnl
                trades.append(_trade(rbr_ny_entry_time, _idx[i], rbr_ny_entry_px, rbr_ny_stop, -1, abs(rbr_ny_pos), pnl,
                                     "SL" if not rbr_ny_tp1_filled else "Runner SL", "RBR NY"))
                rbr_ny_pos = 0; rbr_ny_dir = 0; rbr_ny_tp1_filled = False
            else:
                if not rbr_ny_tp1_filled and l <= rbr_ny_tp1_px:
                    rbr_ny_tp1_filled = True
                    pnl = (rbr_ny_entry_px - rbr_ny_tp1_px) * 1 * pt_val - fee
                    equity += pnl; rbr_ny_pos += 1
                    trades.append(_trade(rbr_ny_entry_time, _idx[i], rbr_ny_entry_px, rbr_ny_tp1_px, -1, 1, pnl, "TP1", "RBR NY"))
                if rbr_ny_pos < 0 and l <= rbr_ny_runner_px:
                    pnl = (rbr_ny_entry_px - rbr_ny_runner_px) * abs(rbr_ny_pos) * pt_val - abs(rbr_ny_pos) * fee
                    equity += pnl
                    trades.append(_trade(rbr_ny_entry_time, _idx[i], rbr_ny_entry_px, rbr_ny_runner_px, -1, abs(rbr_ny_pos), pnl, "Runner TP", "RBR NY"))
                    rbr_ny_pos = 0; rbr_ny_dir = 0; rbr_ny_tp1_filled = False

        # =================================================================
        #  RBR TOKYO — session flatten
        # =================================================================
        rbr_tk_flat = tk_h >= rbr_tk_flat_hr
        if rbr_tk_flat and rbr_tk_pos != 0:
            pnl = _calc_pnl(rbr_tk_pos, rbr_tk_dir, rbr_tk_entry_px, c, pt_val, fee)
            equity += pnl
            trades.append(_trade(rbr_tk_entry_time, _idx[i], rbr_tk_entry_px, c,
                                 rbr_tk_dir, abs(rbr_tk_pos), pnl, "EOD Flatten", "RBR Tokyo"))
            rbr_tk_pos = 0; rbr_tk_dir = 0; rbr_tk_tp1_filled = False

        # =================================================================
        #  RBR TOKYO — exit logic
        # =================================================================
        if rbr_tk_pos > 0 and rbr_tk_dir == 1:
            if l <= rbr_tk_stop:
                pnl = _calc_pnl(rbr_tk_pos, 1, rbr_tk_entry_px, rbr_tk_stop, pt_val, fee)
                equity += pnl
                trades.append(_trade(rbr_tk_entry_time, _idx[i], rbr_tk_entry_px, rbr_tk_stop, 1, rbr_tk_pos, pnl,
                                     "SL" if not rbr_tk_tp1_filled else "Runner SL", "RBR Tokyo"))
                rbr_tk_pos = 0; rbr_tk_dir = 0; rbr_tk_tp1_filled = False
            else:
                if not rbr_tk_tp1_filled and h >= rbr_tk_tp1_px:
                    rbr_tk_tp1_filled = True
                    pnl = (rbr_tk_tp1_px - rbr_tk_entry_px) * 1 * pt_val - fee
                    equity += pnl; rbr_tk_pos -= 1
                    trades.append(_trade(rbr_tk_entry_time, _idx[i], rbr_tk_entry_px, rbr_tk_tp1_px, 1, 1, pnl, "TP1", "RBR Tokyo"))
                if rbr_tk_pos > 0 and h >= rbr_tk_runner_px:
                    pnl = (rbr_tk_runner_px - rbr_tk_entry_px) * rbr_tk_pos * pt_val - rbr_tk_pos * fee
                    equity += pnl
                    trades.append(_trade(rbr_tk_entry_time, _idx[i], rbr_tk_entry_px, rbr_tk_runner_px, 1, rbr_tk_pos, pnl, "Runner TP", "RBR Tokyo"))
                    rbr_tk_pos = 0; rbr_tk_dir = 0; rbr_tk_tp1_filled = False
        elif rbr_tk_pos < 0 and rbr_tk_dir == -1:
            if h >= rbr_tk_stop:
                pnl = _calc_pnl(rbr_tk_pos, -1, rbr_tk_entry_px, rbr_tk_stop, pt_val, fee)
                equity += pnl
                trades.append(_trade(rbr_tk_entry_time, _idx[i], rbr_tk_entry_px, rbr_tk_stop, -1, abs(rbr_tk_pos), pnl,
                                     "SL" if not rbr_tk_tp1_filled else "Runner SL", "RBR Tokyo"))
                rbr_tk_pos = 0; rbr_tk_dir = 0; rbr_tk_tp1_filled = False
            else:
                if not rbr_tk_tp1_filled and l <= rbr_tk_tp1_px:
                    rbr_tk_tp1_filled = True
                    pnl = (rbr_tk_entry_px - rbr_tk_tp1_px) * 1 * pt_val - fee
                    equity += pnl; rbr_tk_pos += 1
                    trades.append(_trade(rbr_tk_entry_time, _idx[i], rbr_tk_entry_px, rbr_tk_tp1_px, -1, 1, pnl, "TP1", "RBR Tokyo"))
                if rbr_tk_pos < 0 and l <= rbr_tk_runner_px:
                    pnl = (rbr_tk_entry_px - rbr_tk_runner_px) * abs(rbr_tk_pos) * pt_val - abs(rbr_tk_pos) * fee
                    equity += pnl
                    trades.append(_trade(rbr_tk_entry_time, _idx[i], rbr_tk_entry_px, rbr_tk_runner_px, -1, abs(rbr_tk_pos), pnl, "Runner TP", "RBR Tokyo"))
                    rbr_tk_pos = 0; rbr_tk_dir = 0; rbr_tk_tp1_filled = False

        # =================================================================
        #  RBR LONDON — session flatten
        # =================================================================
        rbr_ld_flat = ld_h >= rbr_ld_flat_hr
        if rbr_ld_flat and rbr_ld_pos != 0:
            pnl = _calc_pnl(rbr_ld_pos, rbr_ld_dir, rbr_ld_entry_px, c, pt_val, fee)
            equity += pnl
            trades.append(_trade(rbr_ld_entry_time, _idx[i], rbr_ld_entry_px, c,
                                 rbr_ld_dir, abs(rbr_ld_pos), pnl, "EOD Flatten", "RBR London"))
            rbr_ld_pos = 0; rbr_ld_dir = 0; rbr_ld_tp1_filled = False

        # =================================================================
        #  RBR LONDON — exit logic
        # =================================================================
        if rbr_ld_pos > 0 and rbr_ld_dir == 1:
            if l <= rbr_ld_stop:
                pnl = _calc_pnl(rbr_ld_pos, 1, rbr_ld_entry_px, rbr_ld_stop, pt_val, fee)
                equity += pnl
                trades.append(_trade(rbr_ld_entry_time, _idx[i], rbr_ld_entry_px, rbr_ld_stop, 1, rbr_ld_pos, pnl,
                                     "SL" if not rbr_ld_tp1_filled else "Runner SL", "RBR London"))
                rbr_ld_pos = 0; rbr_ld_dir = 0; rbr_ld_tp1_filled = False
            else:
                if not rbr_ld_tp1_filled and h >= rbr_ld_tp1_px:
                    rbr_ld_tp1_filled = True
                    pnl = (rbr_ld_tp1_px - rbr_ld_entry_px) * rbr_ld_tp1_cts * pt_val - rbr_ld_tp1_cts * fee
                    equity += pnl; rbr_ld_pos -= rbr_ld_tp1_cts
                    trades.append(_trade(rbr_ld_entry_time, _idx[i], rbr_ld_entry_px, rbr_ld_tp1_px, 1, rbr_ld_tp1_cts, pnl, "TP1", "RBR London"))
                if rbr_ld_pos > 0 and h >= rbr_ld_runner_px:
                    pnl = (rbr_ld_runner_px - rbr_ld_entry_px) * rbr_ld_pos * pt_val - rbr_ld_pos * fee
                    equity += pnl
                    trades.append(_trade(rbr_ld_entry_time, _idx[i], rbr_ld_entry_px, rbr_ld_runner_px, 1, rbr_ld_pos, pnl, "Runner TP", "RBR London"))
                    rbr_ld_pos = 0; rbr_ld_dir = 0; rbr_ld_tp1_filled = False
        elif rbr_ld_pos < 0 and rbr_ld_dir == -1:
            if h >= rbr_ld_stop:
                pnl = _calc_pnl(rbr_ld_pos, -1, rbr_ld_entry_px, rbr_ld_stop, pt_val, fee)
                equity += pnl
                trades.append(_trade(rbr_ld_entry_time, _idx[i], rbr_ld_entry_px, rbr_ld_stop, -1, abs(rbr_ld_pos), pnl,
                                     "SL" if not rbr_ld_tp1_filled else "Runner SL", "RBR London"))
                rbr_ld_pos = 0; rbr_ld_dir = 0; rbr_ld_tp1_filled = False
            else:
                if not rbr_ld_tp1_filled and l <= rbr_ld_tp1_px:
                    rbr_ld_tp1_filled = True
                    pnl = (rbr_ld_entry_px - rbr_ld_tp1_px) * rbr_ld_tp1_cts * pt_val - rbr_ld_tp1_cts * fee
                    equity += pnl; rbr_ld_pos += rbr_ld_tp1_cts
                    trades.append(_trade(rbr_ld_entry_time, _idx[i], rbr_ld_entry_px, rbr_ld_tp1_px, -1, rbr_ld_tp1_cts, pnl, "TP1", "RBR London"))
                if rbr_ld_pos < 0 and l <= rbr_ld_runner_px:
                    pnl = (rbr_ld_entry_px - rbr_ld_runner_px) * abs(rbr_ld_pos) * pt_val - abs(rbr_ld_pos) * fee
                    equity += pnl
                    trades.append(_trade(rbr_ld_entry_time, _idx[i], rbr_ld_entry_px, rbr_ld_runner_px, -1, abs(rbr_ld_pos), pnl, "Runner TP", "RBR London"))
                    rbr_ld_pos = 0; rbr_ld_dir = 0; rbr_ld_tp1_filled = False

        # =================================================================
        #  DIRECTION LOCK — reset when no open positions exist
        # =================================================================
        # Check if ANY strategy currently has an open position
        any_open = (orb_ny_pos != 0 or orb_tk_pos != 0 or orb_ld_pos != 0
                    or rbr_ny_pos != 0 or rbr_tk_pos != 0 or rbr_ld_pos != 0)
        if not any_open:
            locked_direction = 0

        # =================================================================
        #  ENTRY — ORB NY
        # =================================================================
        if (orb_ny_pos == 0 and orb_ny_after and orb_ny_eok and not orb_ny_flat
                and filt_ny and not orb_ny_traded and not orb_ny_lost and not orb_ny_breakout):
            long_brk  = c > orb_ny_range_high if not np.isnan(orb_ny_range_high) else False
            short_brk = c < orb_ny_range_low if not np.isnan(orb_ny_range_low) else False
            rw = (orb_ny_range_high - orb_ny_range_low) if (not np.isnan(orb_ny_range_high) and not np.isnan(orb_ny_range_low)) else 0
            rw_ok = orb_ny_min_rw <= rw <= orb_ny_max_rw
            if rw_ok and long_brk and (locked_direction == 0 or locked_direction == 1):
                orb_ny_breakout = True
                orb_ny_entry_px = c; orb_ny_pos = orb_ny_cts; orb_ny_dir = 1
                orb_ny_tp1_filled = False; orb_ny_tp2_filled = False
                orb_ny_runner_trail = np.nan; orb_ny_entry_time = _idx[i]
                orb_ny_traded = True; equity -= orb_ny_cts * fee
                locked_direction = 1
            elif rw_ok and short_brk and (locked_direction == 0 or locked_direction == -1):
                orb_ny_breakout = True
                orb_ny_entry_px = c; orb_ny_pos = -orb_ny_cts; orb_ny_dir = -1
                orb_ny_tp1_filled = False; orb_ny_tp2_filled = False
                orb_ny_runner_trail = np.nan; orb_ny_entry_time = _idx[i]
                orb_ny_traded = True; equity -= orb_ny_cts * fee
                locked_direction = -1

        # =================================================================
        #  ENTRY — ORB TOKYO
        # =================================================================
        if (orb_tk_pos == 0 and orb_tk_after and orb_tk_eok and not orb_tk_flat
                and filt_tk and not orb_tk_traded and not orb_tk_lost and not orb_tk_breakout):
            long_brk  = c > orb_tk_range_high if not np.isnan(orb_tk_range_high) else False
            short_brk = c < orb_tk_range_low if not np.isnan(orb_tk_range_low) else False
            rw = (orb_tk_range_high - orb_tk_range_low) if (not np.isnan(orb_tk_range_high) and not np.isnan(orb_tk_range_low)) else 0
            rw_ok = orb_tk_min_rw <= rw <= orb_tk_max_rw
            if rw_ok and long_brk and (locked_direction == 0 or locked_direction == 1):
                orb_tk_breakout = True
                orb_tk_entry_px = c; orb_tk_pos = orb_tk_cts; orb_tk_dir = 1
                orb_tk_tp1_filled = False; orb_tk_tp2_filled = False
                orb_tk_runner_trail = np.nan; orb_tk_entry_time = _idx[i]
                orb_tk_traded = True; equity -= orb_tk_cts * fee
                locked_direction = 1
            elif rw_ok and short_brk and (locked_direction == 0 or locked_direction == -1):
                orb_tk_breakout = True
                orb_tk_entry_px = c; orb_tk_pos = -orb_tk_cts; orb_tk_dir = -1
                orb_tk_tp1_filled = False; orb_tk_tp2_filled = False
                orb_tk_runner_trail = np.nan; orb_tk_entry_time = _idx[i]
                orb_tk_traded = True; equity -= orb_tk_cts * fee
                locked_direction = -1

        # =================================================================
        #  ENTRY — ORB LONDON
        # =================================================================
        if (orb_ld_pos == 0 and orb_ld_after and orb_ld_eok and not orb_ld_flat
                and filt_ld and not orb_ld_traded and not orb_ld_lost and not orb_ld_breakout):
            long_brk  = c > orb_ld_range_high if not np.isnan(orb_ld_range_high) else False
            short_brk = c < orb_ld_range_low if not np.isnan(orb_ld_range_low) else False
            rw = (orb_ld_range_high - orb_ld_range_low) if (not np.isnan(orb_ld_range_high) and not np.isnan(orb_ld_range_low)) else 0
            rw_ok = orb_ld_min_rw <= rw <= orb_ld_max_rw
            if rw_ok and long_brk and (locked_direction == 0 or locked_direction == 1):
                orb_ld_breakout = True
                orb_ld_entry_px = c; orb_ld_pos = orb_ld_cts; orb_ld_dir = 1
                orb_ld_tp1_filled = False; orb_ld_tp2_filled = False
                orb_ld_runner_trail = np.nan; orb_ld_entry_time = _idx[i]
                orb_ld_traded = True; equity -= orb_ld_cts * fee
                locked_direction = 1
            elif rw_ok and short_brk and (locked_direction == 0 or locked_direction == -1):
                orb_ld_breakout = True
                orb_ld_entry_px = c; orb_ld_pos = -orb_ld_cts; orb_ld_dir = -1
                orb_ld_tp1_filled = False; orb_ld_tp2_filled = False
                orb_ld_runner_trail = np.nan; orb_ld_entry_time = _idx[i]
                orb_ld_traded = True; equity -= orb_ld_cts * fee
                locked_direction = -1

        # =================================================================
        #  ENTRY — RBR NY
        # =================================================================
        rbr_ny_in_sess = ((ny_h == 9 and ny_m >= 30) or ny_h == 10 or (ny_h == 11 and ny_m < 30))
        if rbr_ny_pos == 0 and rbr_ny_in_sess and not rbr_ny_flat and i >= rbr_ny_lookback:
            ef = rbr_ny_ema_f[i]; es = rbr_ny_ema_s[i]
            if not (np.isnan(ef) or np.isnan(es)):
                uptrend = ef > es; downtrend = ef < es
                cvs = rbr_ny_vol_sma[i] if not np.isnan(rbr_ny_vol_sma[i]) else 0.0
                vol_spike = cvs > 0 and v >= cvs * rbr_ny_vol_mult
                bull_confirm = c > o; bear_confirm = c < o
                ema_top = max(ef, es); ema_bot = min(ef, es)
                ema_touch_bull = l <= ema_top and c > ema_top
                ema_touch_bear = h >= ema_bot and c < ema_bot

                rbr_found = False
                for bc in range(1, rbr_ny_max_bb + 1):
                    r1i = i - (bc + 1)
                    if r1i < 0 or (bc + 1) >= rbr_ny_lookback: continue
                    if body_ratio_arr[r1i] >= rbr_ny_body_r and close_arr[r1i] > open_arr[r1i]:
                        ab = True; bh = -np.inf; bl_ = np.inf
                        for bo in range(1, bc + 1):
                            bi = i - bo
                            if body_ratio_arr[bi] > rbr_ny_doji_r: ab = False; break
                            bh = max(bh, high_arr[bi]); bl_ = min(bl_, low_arr[bi])
                        if ab and c > bh: rbr_found = True; break

                dbd_found = False
                for bc in range(1, rbr_ny_max_bb + 1):
                    d1i = i - (bc + 1)
                    if d1i < 0 or (bc + 1) >= rbr_ny_lookback: continue
                    if body_ratio_arr[d1i] >= rbr_ny_body_r and close_arr[d1i] < open_arr[d1i]:
                        ab = True; bh = -np.inf; bl_ = np.inf
                        for bo in range(1, bc + 1):
                            bi = i - bo
                            if body_ratio_arr[bi] > rbr_ny_doji_r: ab = False; break
                            bh = max(bh, high_arr[bi]); bl_ = min(bl_, low_arr[bi])
                        if ab and c < bl_: dbd_found = True; break

                long_sig = rbr_ny_in_sess and uptrend and rbr_found and ema_touch_bull and vol_spike and bull_confirm
                if long_sig and (locked_direction == 0 or locked_direction == 1):
                    rbr_ny_entry_px = c; rbr_ny_pos = rbr_ny_cts; rbr_ny_dir = 1
                    rbr_ny_tp1_filled = False; rbr_ny_entry_time = _idx[i]
                    rbr_ny_stop = c - rbr_ny_risk
                    rbr_ny_tp1_px = c + rbr_ny_risk * rbr_ny_rr1
                    rbr_ny_runner_px = c + rbr_ny_risk * rbr_ny_rr_run
                    equity -= rbr_ny_cts * fee; locked_direction = 1
                else:
                    short_sig = rbr_ny_in_sess and downtrend and dbd_found and ema_touch_bear and vol_spike and bear_confirm
                    if short_sig and (locked_direction == 0 or locked_direction == -1):
                        rbr_ny_entry_px = c; rbr_ny_pos = -rbr_ny_cts; rbr_ny_dir = -1
                        rbr_ny_tp1_filled = False; rbr_ny_entry_time = _idx[i]
                        rbr_ny_stop = c + rbr_ny_risk
                        rbr_ny_tp1_px = c - rbr_ny_risk * rbr_ny_rr1
                        rbr_ny_runner_px = c - rbr_ny_risk * rbr_ny_rr_run
                        equity -= rbr_ny_cts * fee; locked_direction = -1

        # =================================================================
        #  ENTRY — RBR TOKYO
        # =================================================================
        rbr_tk_in_sess = (9 <= tk_h < 11)
        if rbr_tk_pos == 0 and rbr_tk_in_sess and not rbr_tk_flat and i >= rbr_tk_lookback:
            ef = rbr_tk_ema_f[i]; es = rbr_tk_ema_s[i]
            if not (np.isnan(ef) or np.isnan(es)):
                uptrend = ef > es; downtrend = ef < es
                cvs = rbr_tk_vol_sma[i] if not np.isnan(rbr_tk_vol_sma[i]) else 0.0
                vol_spike = cvs > 0 and v >= cvs * rbr_tk_vol_mult
                bull_confirm = c > o; bear_confirm = c < o
                ema_top = max(ef, es); ema_bot = min(ef, es)
                ema_touch_bull = l <= ema_top and c > ema_top
                ema_touch_bear = h >= ema_bot and c < ema_bot

                rbr_found = False
                for bc in range(1, rbr_tk_max_bb + 1):
                    r1i = i - (bc + 1)
                    if r1i < 0 or (bc + 1) >= rbr_tk_lookback: continue
                    if body_ratio_arr[r1i] >= rbr_tk_body_r and close_arr[r1i] > open_arr[r1i]:
                        ab = True; bh = -np.inf; bl_ = np.inf
                        for bo in range(1, bc + 1):
                            bi = i - bo
                            if body_ratio_arr[bi] > rbr_tk_doji_r: ab = False; break
                            bh = max(bh, high_arr[bi]); bl_ = min(bl_, low_arr[bi])
                        if ab and c > bh: rbr_found = True; break

                dbd_found = False
                for bc in range(1, rbr_tk_max_bb + 1):
                    d1i = i - (bc + 1)
                    if d1i < 0 or (bc + 1) >= rbr_tk_lookback: continue
                    if body_ratio_arr[d1i] >= rbr_tk_body_r and close_arr[d1i] < open_arr[d1i]:
                        ab = True; bh = -np.inf; bl_ = np.inf
                        for bo in range(1, bc + 1):
                            bi = i - bo
                            if body_ratio_arr[bi] > rbr_tk_doji_r: ab = False; break
                            bh = max(bh, high_arr[bi]); bl_ = min(bl_, low_arr[bi])
                        if ab and c < bl_: dbd_found = True; break

                long_sig = rbr_tk_in_sess and uptrend and rbr_found and ema_touch_bull and vol_spike and bull_confirm
                if long_sig and (locked_direction == 0 or locked_direction == 1):
                    rbr_tk_entry_px = c; rbr_tk_pos = rbr_tk_cts; rbr_tk_dir = 1
                    rbr_tk_tp1_filled = False; rbr_tk_entry_time = _idx[i]
                    rbr_tk_stop = c - rbr_tk_risk
                    rbr_tk_tp1_px = c + rbr_tk_risk * rbr_tk_rr1
                    rbr_tk_runner_px = c + rbr_tk_risk * rbr_tk_rr_run
                    equity -= rbr_tk_cts * fee; locked_direction = 1
                else:
                    short_sig = rbr_tk_in_sess and downtrend and dbd_found and ema_touch_bear and vol_spike and bear_confirm
                    if short_sig and (locked_direction == 0 or locked_direction == -1):
                        rbr_tk_entry_px = c; rbr_tk_pos = -rbr_tk_cts; rbr_tk_dir = -1
                        rbr_tk_tp1_filled = False; rbr_tk_entry_time = _idx[i]
                        rbr_tk_stop = c + rbr_tk_risk
                        rbr_tk_tp1_px = c - rbr_tk_risk * rbr_tk_rr1
                        rbr_tk_runner_px = c - rbr_tk_risk * rbr_tk_rr_run
                        equity -= rbr_tk_cts * fee; locked_direction = -1

        # =================================================================
        #  ENTRY — RBR LONDON
        # =================================================================
        rbr_ld_in_sess = (8 <= ld_h < 10)
        if rbr_ld_pos == 0 and rbr_ld_in_sess and not rbr_ld_flat and i >= rbr_ld_lookback:
            ef = rbr_ld_ema_f[i]; es = rbr_ld_ema_s[i]
            if not (np.isnan(ef) or np.isnan(es)):
                uptrend = ef > es; downtrend = ef < es
                cvs = rbr_ld_vol_sma[i] if not np.isnan(rbr_ld_vol_sma[i]) else 0.0
                vol_spike = cvs > 0 and v >= cvs * rbr_ld_vol_mult
                bull_confirm = c > o; bear_confirm = c < o
                ema_top = max(ef, es); ema_bot = min(ef, es)
                ema_touch_bull = l <= ema_top and c > ema_top
                ema_touch_bear = h >= ema_bot and c < ema_bot

                rbr_found = False
                for bc in range(1, rbr_ld_max_bb + 1):
                    r1i = i - (bc + 1)
                    if r1i < 0 or (bc + 1) >= rbr_ld_lookback: continue
                    if body_ratio_arr[r1i] >= rbr_ld_body_r and close_arr[r1i] > open_arr[r1i]:
                        ab = True; bh = -np.inf; bl_ = np.inf
                        for bo in range(1, bc + 1):
                            bi = i - bo
                            if body_ratio_arr[bi] > rbr_ld_doji_r: ab = False; break
                            bh = max(bh, high_arr[bi]); bl_ = min(bl_, low_arr[bi])
                        if ab and c > bh: rbr_found = True; break

                dbd_found = False
                for bc in range(1, rbr_ld_max_bb + 1):
                    d1i = i - (bc + 1)
                    if d1i < 0 or (bc + 1) >= rbr_ld_lookback: continue
                    if body_ratio_arr[d1i] >= rbr_ld_body_r and close_arr[d1i] < open_arr[d1i]:
                        ab = True; bh = -np.inf; bl_ = np.inf
                        for bo in range(1, bc + 1):
                            bi = i - bo
                            if body_ratio_arr[bi] > rbr_ld_doji_r: ab = False; break
                            bh = max(bh, high_arr[bi]); bl_ = min(bl_, low_arr[bi])
                        if ab and c < bl_: dbd_found = True; break

                long_sig = rbr_ld_in_sess and uptrend and rbr_found and ema_touch_bull and vol_spike and bull_confirm
                if long_sig and (locked_direction == 0 or locked_direction == 1):
                    rbr_ld_entry_px = c; rbr_ld_pos = rbr_ld_cts; rbr_ld_dir = 1
                    rbr_ld_tp1_filled = False; rbr_ld_entry_time = _idx[i]
                    rbr_ld_stop = c - rbr_ld_risk
                    rbr_ld_tp1_px = c + rbr_ld_risk * rbr_ld_rr1
                    rbr_ld_runner_px = c + rbr_ld_risk * rbr_ld_rr_run
                    equity -= rbr_ld_cts * fee; locked_direction = 1
                else:
                    short_sig = rbr_ld_in_sess and downtrend and dbd_found and ema_touch_bear and vol_spike and bear_confirm
                    if short_sig and (locked_direction == 0 or locked_direction == -1):
                        rbr_ld_entry_px = c; rbr_ld_pos = -rbr_ld_cts; rbr_ld_dir = -1
                        rbr_ld_tp1_filled = False; rbr_ld_entry_time = _idx[i]
                        rbr_ld_stop = c + rbr_ld_risk
                        rbr_ld_tp1_px = c - rbr_ld_risk * rbr_ld_rr1
                        rbr_ld_runner_px = c - rbr_ld_risk * rbr_ld_rr_run
                        equity -= rbr_ld_cts * fee; locked_direction = -1

        # =================================================================
        #  Record equity
        # =================================================================
        equity_curve.append(equity)
        date_list.append(_idx[i])

    # =====================================================================
    #  CLOSE ANY OPEN POSITIONS AT END OF DATA
    # =====================================================================
    last_c = close_arr[-1]
    for tag, s_pos, s_dir, s_epx, s_et in [
        ("ORB NY",      orb_ny_pos, orb_ny_dir, orb_ny_entry_px, orb_ny_entry_time),
        ("ORB Tokyo",   orb_tk_pos, orb_tk_dir, orb_tk_entry_px, orb_tk_entry_time),
        ("ORB London",  orb_ld_pos, orb_ld_dir, orb_ld_entry_px, orb_ld_entry_time),
        ("RBR NY",      rbr_ny_pos, rbr_ny_dir, rbr_ny_entry_px, rbr_ny_entry_time),
        ("RBR Tokyo",   rbr_tk_pos, rbr_tk_dir, rbr_tk_entry_px, rbr_tk_entry_time),
        ("RBR London",  rbr_ld_pos, rbr_ld_dir, rbr_ld_entry_px, rbr_ld_entry_time),
    ]:
        if s_pos != 0:
            pnl = _calc_pnl(s_pos, s_dir, s_epx, last_c, pt_val, fee)
            equity += pnl
            trades.append(_trade(s_et, _idx[-1], s_epx, last_c, s_dir, abs(s_pos), pnl, "End of Data", tag))
    if equity_curve:
        equity_curve[-1] = equity

    eq = pd.Series(equity_curve, index=date_list)
    stats = compute_stats(trades, eq, capital)
    return {"trades": trades, "equity": eq, "stats": stats, "params": params}
