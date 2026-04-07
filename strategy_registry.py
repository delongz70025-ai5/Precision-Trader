"""
Strategy Registry — Plugin system for multiple trading strategies.
Each strategy defines its own parameters, backtest logic, optimization grid,
and frozen params. The walk-forward engine and app are strategy-agnostic.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from strategy import (
    StrategyParams, run_backtest, compute_stats,
    ema, rsi, macd, atr, supertrend, dmi, vwap,
)


# ══════════════════════════════════════════════════════════════════════════════
# Base class — all strategies must implement this interface
# ══════════════════════════════════════════════════════════════════════════════

class BaseStrategy(ABC):
    """Interface that every strategy plugin must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name shown in the dropdown."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description."""

    @abstractmethod
    def default_grid(self) -> Dict[str, List]:
        """Optimizable parameter grid for the sidebar."""

    @abstractmethod
    def frozen_params(self) -> Dict[str, Any]:
        """Parameters that stay fixed (not optimized)."""

    @abstractmethod
    def run(self, df: pd.DataFrame, params: dict) -> dict:
        """
        Run a single backtest. Must return:
            { "trades": [...], "equity": pd.Series, "stats": dict, "params": dict }
        """

    def sidebar_grid_options(self) -> Dict[str, Dict]:
        """
        Define what the sidebar shows for each optimizable parameter.
        Returns: { param_name: { "options": [...], "default": [...], "label": str } }
        Falls back to default_grid() values if not overridden.
        """
        grid = self.default_grid()
        return {
            k: {"options": v, "default": v, "label": k.replace("_", " ").title()}
            for k, v in grid.items()
        }


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 1: Precision Sniper v7.5 (original)
# ══════════════════════════════════════════════════════════════════════════════

class PrecisionSniperStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "Precision Sniper NY 15min MNQ"

    @property
    def description(self) -> str:
        return "EMA crossover + pullback entries with 10-point confluence scoring, Supertrend filter, prop-firm safety."

    def default_grid(self) -> Dict[str, List]:
        return {
            "ema_fast_len":   [8, 10, 13],
            "ema_slow_len":   [18, 21, 26],
            "ema_trend_len":  [50, 55, 60],
            "min_score":      [4, 5, 6],
            "rsi_len":        [21, 26],
            "fixed_risk_pts": [20.0, 25.0, 30.0],
            "tp1_rr":         [1.5, 2.0],
            "tp2_rr":         [3.0, 3.5],
            "st_factor":      [3.5, 4.0],
        }

    def frozen_params(self) -> Dict[str, Any]:
        return {
            "tp3_qty": 0, "use_supertrend": True, "use_trail": True,
            "trail_after_tp": "TP2", "tp1_qty": 1, "tp2_qty": 1,
            "use_force_close": True, "use_max_trade_loss": True,
            "use_daily_max_loss": True, "max_trade_loss": 200.0,
            "daily_max_loss": 650.0, "exchange_fee_pct": 0.0010,
            "slippage_pct": 0.0005, "point_value": 2.0,
            "initial_capital": 50000.0, "allow_longs": True,
            "allow_shorts": True, "use_session": True,
            "use_pullback": True, "pullback_score": 4, "use_atr_risk": False,
        }

    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "ema_fast_len":   {"options": [5, 8, 10, 13, 15],         "default": [8, 10, 13],       "label": "EMA Fast"},
            "ema_slow_len":   {"options": [15, 18, 21, 26, 30],       "default": [18, 21, 26],      "label": "EMA Slow"},
            "ema_trend_len":  {"options": [40, 50, 55, 60, 75],       "default": [50, 55, 60],      "label": "EMA Trend"},
            "min_score":      {"options": [3, 4, 5, 6, 7],            "default": [4, 5, 6],         "label": "Min Score"},
            "rsi_len":        {"options": [14, 21, 26, 30],           "default": [21, 26],           "label": "RSI Length"},
            "fixed_risk_pts": {"options": [15.0, 20.0, 25.0, 30.0, 35.0], "default": [20.0, 25.0, 30.0], "label": "Fixed Risk Pts"},
            "tp1_rr":         {"options": [1.5, 2.0, 2.5],            "default": [1.5, 2.0],        "label": "TP1 R:R"},
            "tp2_rr":         {"options": [3.0, 3.5, 4.0],            "default": [3.0, 3.5],        "label": "TP2 R:R"},
            "st_factor":      {"options": [3.0, 3.5, 4.0, 4.5],       "default": [3.5, 4.0],        "label": "ST Factor"},
        }

    def run(self, df: pd.DataFrame, params: dict) -> dict:
        p = StrategyParams(**params)
        return run_backtest(df, p)


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 2: Simple EMA Crossover
# ══════════════════════════════════════════════════════════════════════════════

class EMACrossoverStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "EMA Crossover All 15min MNQ"

    @property
    def description(self) -> str:
        return "Classic fast/slow EMA crossover. Long when fast > slow, short when fast < slow. ATR-based stops."

    def default_grid(self) -> Dict[str, List]:
        return {
            "ema_fast":   [8, 10, 13, 15],
            "ema_slow":   [21, 26, 30, 50],
            "atr_len":    [14, 20],
            "sl_atr_mult": [1.5, 2.0, 2.5],
            "tp_atr_mult": [2.0, 3.0, 4.0],
        }

    def frozen_params(self) -> Dict[str, Any]:
        return {
            "initial_capital": 50000.0,
            "contracts": 2,
            "exchange_fee_pct": 0.0010,
            "slippage_pct": 0.0005,
            "point_value": 2.0,
            "use_session": True,
        }

    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "ema_fast":    {"options": [5, 8, 10, 13, 15, 20],       "default": [8, 10, 13, 15],   "label": "EMA Fast"},
            "ema_slow":    {"options": [20, 21, 26, 30, 50, 55],     "default": [21, 26, 30, 50],  "label": "EMA Slow"},
            "atr_len":     {"options": [10, 14, 20, 26],             "default": [14, 20],           "label": "ATR Length"},
            "sl_atr_mult": {"options": [1.0, 1.5, 2.0, 2.5, 3.0],  "default": [1.5, 2.0, 2.5],   "label": "SL ATR Mult"},
            "tp_atr_mult": {"options": [1.5, 2.0, 3.0, 4.0, 5.0],  "default": [2.0, 3.0, 4.0],   "label": "TP ATR Mult"},
        }

    def run(self, df: pd.DataFrame, params: dict) -> dict:
        frozen = self.frozen_params()
        merged = {**frozen, **params}

        ema_fast_len = int(merged["ema_fast"])
        ema_slow_len = int(merged["ema_slow"])
        atr_len      = int(merged["atr_len"])
        sl_mult      = float(merged["sl_atr_mult"])
        tp_mult      = float(merged["tp_atr_mult"])
        contracts    = int(merged.get("contracts", 2))
        capital      = float(merged["initial_capital"])
        cost_pct     = float(merged["exchange_fee_pct"]) + float(merged["slippage_pct"])
        point_val    = float(merged["point_value"])
        use_session  = bool(merged.get("use_session", True))

        src = df["close"]
        ef  = ema(src, ema_fast_len)
        es  = ema(src, ema_slow_len)
        atr_val = atr(df["high"], df["low"], src, atr_len)

        warmup = max(ema_slow_len, atr_len, 50)
        equity = capital
        trades = []
        equity_curve = []
        dates = []

        pos_size    = 0
        entry_price = np.nan
        sl_price    = np.nan
        tp_price    = np.nan
        trade_dir   = 0
        entry_time  = None

        n = len(df)
        for i in range(n):
            bar_time = df.index[i]
            c = df["close"].iloc[i]
            h = df["high"].iloc[i]
            l = df["low"].iloc[i]

            # Session filter (9:30-16:00 ET)
            if use_session:
                bar_mins = bar_time.hour * 60 + bar_time.minute
                in_session = (bar_mins >= 570) and (bar_mins < 960)
            else:
                in_session = True

            # Force close at 16:00
            if use_session and bar_time.hour * 60 + bar_time.minute >= 960 and pos_size != 0:
                exit_p = c
                cost = abs(exit_p * abs(pos_size) * point_val) * cost_pct
                pnl = (exit_p - entry_price) * pos_size * point_val - cost
                equity += pnl
                trades.append({"entry_time": entry_time, "exit_time": bar_time,
                               "entry_price": entry_price, "exit_price": exit_p,
                               "direction": trade_dir, "contracts": abs(pos_size),
                               "pnl": pnl, "exit_reason": "Session Close",
                               "entry_type": "Cross", "tp1_hit": False, "tp2_hit": False})
                pos_size = 0
                trade_dir = 0

            if i < warmup or i < 1:
                equity_curve.append(equity)
                dates.append(bar_time)
                continue

            ef_now  = float(ef.iloc[i])
            es_now  = float(es.iloc[i])
            ef_prev = float(ef.iloc[i - 1])
            es_prev = float(es.iloc[i - 1])
            cur_atr = float(atr_val.iloc[i]) if not np.isnan(atr_val.iloc[i]) else 20.0

            bull_cross = (ef_prev <= es_prev) and (ef_now > es_now)
            bear_cross = (ef_prev >= es_prev) and (ef_now < es_now)

            # TP/SL check
            if pos_size > 0:
                if h >= tp_price:
                    cost = tp_price * pos_size * point_val * cost_pct
                    pnl = (tp_price - entry_price) * pos_size * point_val - cost
                    equity += pnl
                    trades.append({"entry_time": entry_time, "exit_time": bar_time,
                                   "entry_price": entry_price, "exit_price": tp_price,
                                   "direction": 1, "contracts": pos_size,
                                   "pnl": pnl, "exit_reason": "TP",
                                   "entry_type": "Cross", "tp1_hit": True, "tp2_hit": False})
                    pos_size = 0; trade_dir = 0
                elif l <= sl_price:
                    cost = sl_price * pos_size * point_val * cost_pct
                    pnl = (sl_price - entry_price) * pos_size * point_val - cost
                    equity += pnl
                    trades.append({"entry_time": entry_time, "exit_time": bar_time,
                                   "entry_price": entry_price, "exit_price": sl_price,
                                   "direction": 1, "contracts": pos_size,
                                   "pnl": pnl, "exit_reason": "SL",
                                   "entry_type": "Cross", "tp1_hit": False, "tp2_hit": False})
                    pos_size = 0; trade_dir = 0

            elif pos_size < 0:
                if l <= tp_price:
                    cost = tp_price * abs(pos_size) * point_val * cost_pct
                    pnl = (entry_price - tp_price) * abs(pos_size) * point_val - cost
                    equity += pnl
                    trades.append({"entry_time": entry_time, "exit_time": bar_time,
                                   "entry_price": entry_price, "exit_price": tp_price,
                                   "direction": -1, "contracts": abs(pos_size),
                                   "pnl": pnl, "exit_reason": "TP",
                                   "entry_type": "Cross", "tp1_hit": True, "tp2_hit": False})
                    pos_size = 0; trade_dir = 0
                elif h >= sl_price:
                    cost = sl_price * abs(pos_size) * point_val * cost_pct
                    pnl = (entry_price - sl_price) * abs(pos_size) * point_val - cost
                    equity += pnl
                    trades.append({"entry_time": entry_time, "exit_time": bar_time,
                                   "entry_price": entry_price, "exit_price": sl_price,
                                   "direction": -1, "contracts": abs(pos_size),
                                   "pnl": pnl, "exit_reason": "SL",
                                   "entry_type": "Cross", "tp1_hit": False, "tp2_hit": False})
                    pos_size = 0; trade_dir = 0

            # Entry
            if pos_size == 0 and in_session:
                if bull_cross:
                    entry_price = c
                    sl_price = c - cur_atr * sl_mult
                    tp_price = c + cur_atr * tp_mult
                    pos_size = contracts
                    trade_dir = 1
                    entry_time = bar_time
                    equity -= c * contracts * point_val * cost_pct
                elif bear_cross:
                    entry_price = c
                    sl_price = c + cur_atr * sl_mult
                    tp_price = c - cur_atr * tp_mult
                    pos_size = -contracts
                    trade_dir = -1
                    entry_time = bar_time
                    equity -= c * contracts * point_val * cost_pct

            equity_curve.append(equity)
            dates.append(bar_time)

        # Close open position
        if pos_size != 0:
            exit_p = df["close"].iloc[-1]
            cost = exit_p * abs(pos_size) * point_val * cost_pct
            pnl = (exit_p - entry_price) * pos_size * point_val - cost
            equity += pnl
            trades.append({"entry_time": entry_time, "exit_time": df.index[-1],
                           "entry_price": entry_price, "exit_price": exit_p,
                           "direction": trade_dir, "contracts": abs(pos_size),
                           "pnl": pnl, "exit_reason": "End of Data",
                           "entry_type": "Cross", "tp1_hit": False, "tp2_hit": False})
            equity_curve[-1] = equity

        eq_series = pd.Series(equity_curve, index=dates)
        stats = compute_stats(trades, eq_series, capital)
        return {"trades": trades, "equity": eq_series, "stats": stats, "params": params}


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 3: RSI Mean Reversion
# ══════════════════════════════════════════════════════════════════════════════

class RSIMeanReversionStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "RSI Reversion All 15min MNQ"

    @property
    def description(self) -> str:
        return "Buy oversold (RSI < threshold), sell overbought (RSI > threshold). Trend filter with EMA."

    def default_grid(self) -> Dict[str, List]:
        return {
            "rsi_len":       [10, 14, 21],
            "rsi_oversold":  [25, 30, 35],
            "rsi_overbought": [65, 70, 75],
            "trend_ema_len": [50, 100, 200],
            "atr_len":       [14, 20],
            "sl_atr_mult":   [1.5, 2.0],
            "tp_atr_mult":   [1.5, 2.0, 3.0],
        }

    def frozen_params(self) -> Dict[str, Any]:
        return {
            "initial_capital": 50000.0, "contracts": 2,
            "exchange_fee_pct": 0.0010, "slippage_pct": 0.0005,
            "point_value": 2.0, "use_session": True,
        }

    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "rsi_len":        {"options": [7, 10, 14, 21, 26],       "default": [10, 14, 21],      "label": "RSI Length"},
            "rsi_oversold":   {"options": [20, 25, 30, 35],          "default": [25, 30, 35],      "label": "RSI Oversold"},
            "rsi_overbought": {"options": [65, 70, 75, 80],          "default": [65, 70, 75],      "label": "RSI Overbought"},
            "trend_ema_len":  {"options": [50, 100, 150, 200],       "default": [50, 100, 200],    "label": "Trend EMA"},
            "atr_len":        {"options": [10, 14, 20],              "default": [14, 20],           "label": "ATR Length"},
            "sl_atr_mult":    {"options": [1.0, 1.5, 2.0, 2.5],     "default": [1.5, 2.0],        "label": "SL ATR Mult"},
            "tp_atr_mult":    {"options": [1.0, 1.5, 2.0, 3.0],     "default": [1.5, 2.0, 3.0],   "label": "TP ATR Mult"},
        }

    def run(self, df: pd.DataFrame, params: dict) -> dict:
        frozen = self.frozen_params()
        merged = {**frozen, **params}

        rsi_len       = int(merged["rsi_len"])
        rsi_os        = float(merged["rsi_oversold"])
        rsi_ob        = float(merged["rsi_overbought"])
        trend_len     = int(merged["trend_ema_len"])
        atr_len_val   = int(merged["atr_len"])
        sl_mult       = float(merged["sl_atr_mult"])
        tp_mult       = float(merged["tp_atr_mult"])
        contracts     = int(merged.get("contracts", 2))
        capital       = float(merged["initial_capital"])
        cost_pct      = float(merged["exchange_fee_pct"]) + float(merged["slippage_pct"])
        point_val     = float(merged["point_value"])
        use_session   = bool(merged.get("use_session", True))

        src = df["close"]
        rsi_val = rsi(src, rsi_len)
        trend_ema = ema(src, trend_len)
        atr_val = atr(df["high"], df["low"], src, atr_len_val)

        warmup = max(trend_len, rsi_len, 50)
        equity = capital
        trades = []
        equity_curve = []
        dates = []

        pos_size = 0
        entry_price = np.nan
        sl_price = np.nan
        tp_price = np.nan
        trade_dir = 0
        entry_time = None

        n = len(df)
        for i in range(n):
            bar_time = df.index[i]
            c = df["close"].iloc[i]
            h = df["high"].iloc[i]
            l = df["low"].iloc[i]

            if use_session:
                bar_mins = bar_time.hour * 60 + bar_time.minute
                in_session = (bar_mins >= 570) and (bar_mins < 960)
                if bar_mins >= 960 and pos_size != 0:
                    cost = abs(c * abs(pos_size) * point_val) * cost_pct
                    pnl = (c - entry_price) * pos_size * point_val - cost
                    equity += pnl
                    trades.append({"entry_time": entry_time, "exit_time": bar_time,
                                   "entry_price": entry_price, "exit_price": c,
                                   "direction": trade_dir, "contracts": abs(pos_size),
                                   "pnl": pnl, "exit_reason": "Session Close",
                                   "entry_type": "RSI", "tp1_hit": False, "tp2_hit": False})
                    pos_size = 0; trade_dir = 0
            else:
                in_session = True

            if i < warmup:
                equity_curve.append(equity)
                dates.append(bar_time)
                continue

            cur_rsi = float(rsi_val.iloc[i]) if not np.isnan(rsi_val.iloc[i]) else 50.0
            cur_trend = float(trend_ema.iloc[i])
            cur_atr = float(atr_val.iloc[i]) if not np.isnan(atr_val.iloc[i]) else 20.0

            # TP/SL
            if pos_size > 0:
                if h >= tp_price:
                    cost = tp_price * pos_size * point_val * cost_pct
                    pnl = (tp_price - entry_price) * pos_size * point_val - cost
                    equity += pnl
                    trades.append({"entry_time": entry_time, "exit_time": bar_time,
                                   "entry_price": entry_price, "exit_price": tp_price,
                                   "direction": 1, "contracts": pos_size, "pnl": pnl,
                                   "exit_reason": "TP", "entry_type": "RSI",
                                   "tp1_hit": True, "tp2_hit": False})
                    pos_size = 0; trade_dir = 0
                elif l <= sl_price:
                    cost = sl_price * pos_size * point_val * cost_pct
                    pnl = (sl_price - entry_price) * pos_size * point_val - cost
                    equity += pnl
                    trades.append({"entry_time": entry_time, "exit_time": bar_time,
                                   "entry_price": entry_price, "exit_price": sl_price,
                                   "direction": 1, "contracts": pos_size, "pnl": pnl,
                                   "exit_reason": "SL", "entry_type": "RSI",
                                   "tp1_hit": False, "tp2_hit": False})
                    pos_size = 0; trade_dir = 0
            elif pos_size < 0:
                if l <= tp_price:
                    cost = tp_price * abs(pos_size) * point_val * cost_pct
                    pnl = (entry_price - tp_price) * abs(pos_size) * point_val - cost
                    equity += pnl
                    trades.append({"entry_time": entry_time, "exit_time": bar_time,
                                   "entry_price": entry_price, "exit_price": tp_price,
                                   "direction": -1, "contracts": abs(pos_size), "pnl": pnl,
                                   "exit_reason": "TP", "entry_type": "RSI",
                                   "tp1_hit": True, "tp2_hit": False})
                    pos_size = 0; trade_dir = 0
                elif h >= sl_price:
                    cost = sl_price * abs(pos_size) * point_val * cost_pct
                    pnl = (entry_price - sl_price) * abs(pos_size) * point_val - cost
                    equity += pnl
                    trades.append({"entry_time": entry_time, "exit_time": bar_time,
                                   "entry_price": entry_price, "exit_price": sl_price,
                                   "direction": -1, "contracts": abs(pos_size), "pnl": pnl,
                                   "exit_reason": "SL", "entry_type": "RSI",
                                   "tp1_hit": False, "tp2_hit": False})
                    pos_size = 0; trade_dir = 0

            # Entry: RSI reversal with trend filter
            if pos_size == 0 and in_session:
                if cur_rsi < rsi_os and c > cur_trend:  # oversold + uptrend = buy
                    entry_price = c
                    sl_price = c - cur_atr * sl_mult
                    tp_price = c + cur_atr * tp_mult
                    pos_size = contracts
                    trade_dir = 1
                    entry_time = bar_time
                    equity -= c * contracts * point_val * cost_pct
                elif cur_rsi > rsi_ob and c < cur_trend:  # overbought + downtrend = sell
                    entry_price = c
                    sl_price = c + cur_atr * sl_mult
                    tp_price = c - cur_atr * tp_mult
                    pos_size = -contracts
                    trade_dir = -1
                    entry_time = bar_time
                    equity -= c * contracts * point_val * cost_pct

            equity_curve.append(equity)
            dates.append(bar_time)

        if pos_size != 0:
            exit_p = df["close"].iloc[-1]
            cost = exit_p * abs(pos_size) * point_val * cost_pct
            pnl = (exit_p - entry_price) * pos_size * point_val - cost
            equity += pnl
            trades.append({"entry_time": entry_time, "exit_time": df.index[-1],
                           "entry_price": entry_price, "exit_price": exit_p,
                           "direction": trade_dir, "contracts": abs(pos_size),
                           "pnl": pnl, "exit_reason": "End of Data",
                           "entry_type": "RSI", "tp1_hit": False, "tp2_hit": False})
            equity_curve[-1] = equity

        eq_series = pd.Series(equity_curve, index=dates)
        stats = compute_stats(trades, eq_series, capital)
        return {"trades": trades, "equity": eq_series, "stats": stats, "params": params}


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 4: MACD + Supertrend
# ══════════════════════════════════════════════════════════════════════════════

class MACDSupertrendStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "MACD Supertrend All 15min MNQ"

    @property
    def description(self) -> str:
        return "MACD histogram crossover filtered by Supertrend direction. ATR-based risk."

    def default_grid(self) -> Dict[str, List]:
        return {
            "macd_fast":   [8, 12],
            "macd_slow":   [21, 26],
            "macd_signal": [7, 9],
            "st_factor":   [2.0, 3.0, 4.0],
            "st_atr_len":  [10, 14],
            "sl_atr_mult": [1.5, 2.0],
            "tp_atr_mult": [2.0, 3.0, 4.0],
        }

    def frozen_params(self) -> Dict[str, Any]:
        return {
            "initial_capital": 50000.0, "contracts": 2,
            "exchange_fee_pct": 0.0010, "slippage_pct": 0.0005,
            "point_value": 2.0, "use_session": True,
        }

    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "macd_fast":   {"options": [6, 8, 10, 12],              "default": [8, 12],           "label": "MACD Fast"},
            "macd_slow":   {"options": [20, 21, 26, 30],            "default": [21, 26],          "label": "MACD Slow"},
            "macd_signal": {"options": [5, 7, 9, 12],               "default": [7, 9],            "label": "MACD Signal"},
            "st_factor":   {"options": [1.5, 2.0, 3.0, 4.0, 5.0],  "default": [2.0, 3.0, 4.0],  "label": "ST Factor"},
            "st_atr_len":  {"options": [7, 10, 14, 20],             "default": [10, 14],          "label": "ST ATR Len"},
            "sl_atr_mult": {"options": [1.0, 1.5, 2.0, 2.5],       "default": [1.5, 2.0],        "label": "SL ATR Mult"},
            "tp_atr_mult": {"options": [1.5, 2.0, 3.0, 4.0, 5.0],  "default": [2.0, 3.0, 4.0],  "label": "TP ATR Mult"},
        }

    def run(self, df: pd.DataFrame, params: dict) -> dict:
        frozen = self.frozen_params()
        merged = {**frozen, **params}

        macd_f   = int(merged["macd_fast"])
        macd_s   = int(merged["macd_slow"])
        macd_sig = int(merged["macd_signal"])
        st_fac   = float(merged["st_factor"])
        st_atr   = int(merged["st_atr_len"])
        sl_mult  = float(merged["sl_atr_mult"])
        tp_mult  = float(merged["tp_atr_mult"])
        contracts = int(merged.get("contracts", 2))
        capital  = float(merged["initial_capital"])
        cost_pct = float(merged["exchange_fee_pct"]) + float(merged["slippage_pct"])
        point_val = float(merged["point_value"])
        use_session = bool(merged.get("use_session", True))

        src = df["close"]
        macd_line, macd_signal, macd_hist = macd(src, macd_f, macd_s, macd_sig)
        st_line, st_dir = supertrend(df["high"], df["low"], src, st_fac, st_atr)
        atr_val = atr(df["high"], df["low"], src, st_atr)

        warmup = max(macd_s, st_atr, 50)
        equity = capital
        trades = []
        equity_curve = []
        dates = []

        pos_size = 0
        entry_price = np.nan
        sl_price = np.nan
        tp_price = np.nan
        trade_dir = 0
        entry_time = None

        n = len(df)
        for i in range(n):
            bar_time = df.index[i]
            c = df["close"].iloc[i]
            h = df["high"].iloc[i]
            l = df["low"].iloc[i]

            if use_session:
                bar_mins = bar_time.hour * 60 + bar_time.minute
                in_session = (bar_mins >= 570) and (bar_mins < 960)
                if bar_mins >= 960 and pos_size != 0:
                    cost = abs(c * abs(pos_size) * point_val) * cost_pct
                    pnl = (c - entry_price) * pos_size * point_val - cost
                    equity += pnl
                    trades.append({"entry_time": entry_time, "exit_time": bar_time,
                                   "entry_price": entry_price, "exit_price": c,
                                   "direction": trade_dir, "contracts": abs(pos_size),
                                   "pnl": pnl, "exit_reason": "Session Close",
                                   "entry_type": "MACD", "tp1_hit": False, "tp2_hit": False})
                    pos_size = 0; trade_dir = 0
            else:
                in_session = True

            if i < warmup or i < 1:
                equity_curve.append(equity)
                dates.append(bar_time)
                continue

            cur_hist = float(macd_hist.iloc[i]) if not np.isnan(macd_hist.iloc[i]) else 0.0
            prev_hist = float(macd_hist.iloc[i-1]) if not np.isnan(macd_hist.iloc[i-1]) else 0.0
            st_bull = float(st_dir.iloc[i]) == 1
            st_bear = float(st_dir.iloc[i]) == -1
            cur_atr = float(atr_val.iloc[i]) if not np.isnan(atr_val.iloc[i]) else 20.0

            hist_cross_up   = (prev_hist <= 0) and (cur_hist > 0)
            hist_cross_down = (prev_hist >= 0) and (cur_hist < 0)

            # TP/SL
            if pos_size > 0:
                if h >= tp_price:
                    cost = tp_price * pos_size * point_val * cost_pct
                    pnl = (tp_price - entry_price) * pos_size * point_val - cost
                    equity += pnl
                    trades.append({"entry_time": entry_time, "exit_time": bar_time,
                                   "entry_price": entry_price, "exit_price": tp_price,
                                   "direction": 1, "contracts": pos_size, "pnl": pnl,
                                   "exit_reason": "TP", "entry_type": "MACD",
                                   "tp1_hit": True, "tp2_hit": False})
                    pos_size = 0; trade_dir = 0
                elif l <= sl_price:
                    cost = sl_price * pos_size * point_val * cost_pct
                    pnl = (sl_price - entry_price) * pos_size * point_val - cost
                    equity += pnl
                    trades.append({"entry_time": entry_time, "exit_time": bar_time,
                                   "entry_price": entry_price, "exit_price": sl_price,
                                   "direction": 1, "contracts": pos_size, "pnl": pnl,
                                   "exit_reason": "SL", "entry_type": "MACD",
                                   "tp1_hit": False, "tp2_hit": False})
                    pos_size = 0; trade_dir = 0
            elif pos_size < 0:
                if l <= tp_price:
                    cost = tp_price * abs(pos_size) * point_val * cost_pct
                    pnl = (entry_price - tp_price) * abs(pos_size) * point_val - cost
                    equity += pnl
                    trades.append({"entry_time": entry_time, "exit_time": bar_time,
                                   "entry_price": entry_price, "exit_price": tp_price,
                                   "direction": -1, "contracts": abs(pos_size), "pnl": pnl,
                                   "exit_reason": "TP", "entry_type": "MACD",
                                   "tp1_hit": True, "tp2_hit": False})
                    pos_size = 0; trade_dir = 0
                elif h >= sl_price:
                    cost = sl_price * abs(pos_size) * point_val * cost_pct
                    pnl = (entry_price - sl_price) * abs(pos_size) * point_val - cost
                    equity += pnl
                    trades.append({"entry_time": entry_time, "exit_time": bar_time,
                                   "entry_price": entry_price, "exit_price": sl_price,
                                   "direction": -1, "contracts": abs(pos_size), "pnl": pnl,
                                   "exit_reason": "SL", "entry_type": "MACD",
                                   "tp1_hit": False, "tp2_hit": False})
                    pos_size = 0; trade_dir = 0

            # Entry: MACD histogram cross + Supertrend confirmation
            if pos_size == 0 and in_session:
                if hist_cross_up and st_bull:
                    entry_price = c
                    sl_price = c - cur_atr * sl_mult
                    tp_price = c + cur_atr * tp_mult
                    pos_size = contracts; trade_dir = 1; entry_time = bar_time
                    equity -= c * contracts * point_val * cost_pct
                elif hist_cross_down and st_bear:
                    entry_price = c
                    sl_price = c + cur_atr * sl_mult
                    tp_price = c - cur_atr * tp_mult
                    pos_size = -contracts; trade_dir = -1; entry_time = bar_time
                    equity -= c * contracts * point_val * cost_pct

            equity_curve.append(equity)
            dates.append(bar_time)

        if pos_size != 0:
            exit_p = df["close"].iloc[-1]
            cost = exit_p * abs(pos_size) * point_val * cost_pct
            pnl = (exit_p - entry_price) * pos_size * point_val - cost
            equity += pnl
            trades.append({"entry_time": entry_time, "exit_time": df.index[-1],
                           "entry_price": entry_price, "exit_price": exit_p,
                           "direction": trade_dir, "contracts": abs(pos_size),
                           "pnl": pnl, "exit_reason": "End of Data",
                           "entry_type": "MACD", "tp1_hit": False, "tp2_hit": False})
            equity_curve[-1] = equity

        eq_series = pd.Series(equity_curve, index=dates)
        stats = compute_stats(trades, eq_series, capital)
        return {"trades": trades, "equity": eq_series, "stats": stats, "params": params}


# ══════════════════════════════════════════════════════════════════════════════
# Registry — single source of truth for all available strategies
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Strategy 5: Butterworth + ATR Volatility-Momentum (Institutional)
# ══════════════════════════════════════════════════════════════════════════════

class ButterworthATRStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "BW-ATR All 15min MNQ"

    @property
    def description(self) -> str:
        return (
            "Institutional quant: 2-pole Butterworth low-pass filter (causal lfilter) "
            "strips noise. ATR expansion + ATR momentum confirm entries. Zero lookahead."
        )

    def default_grid(self) -> Dict[str, List]:
        return {
            "bw_fast_period": [8, 12, 16],
            "bw_slow_period": [30, 40, 55],
            "atr_len":        [10, 14, 20],
            "vol_ema_len":    [15, 20, 30],
            "vol_expansion":  [0.8, 1.0, 1.2],
            "atr_roc_len":    [3, 5, 8],
            "sl_atr_mult":    [1.5, 2.0],
            "tp_atr_mult":    [2.5, 3.5, 5.0],
        }

    def frozen_params(self) -> Dict[str, Any]:
        return {
            "initial_capital": 50000.0,
            "contracts": 2,
            "point_value": 2.0,
            "fee_pct": 0.0015,
            "use_session": True,
        }

    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "bw_fast_period": {"options": [6, 8, 10, 12, 16, 20],        "default": [8, 12, 16],        "label": "BW Fast Period"},
            "bw_slow_period": {"options": [25, 30, 40, 55, 70],          "default": [30, 40, 55],       "label": "BW Slow Period"},
            "atr_len":        {"options": [7, 10, 14, 20, 26],           "default": [10, 14, 20],       "label": "ATR Length"},
            "vol_ema_len":    {"options": [10, 15, 20, 30, 40],          "default": [15, 20, 30],       "label": "Vol EMA Length"},
            "vol_expansion":  {"options": [0.6, 0.8, 1.0, 1.2, 1.5],    "default": [0.8, 1.0, 1.2],   "label": "Vol Expansion Threshold"},
            "atr_roc_len":    {"options": [3, 5, 8, 10],                 "default": [3, 5, 8],          "label": "ATR ROC Length"},
            "sl_atr_mult":    {"options": [1.0, 1.5, 2.0, 2.5],         "default": [1.5, 2.0],         "label": "SL ATR Mult"},
            "tp_atr_mult":    {"options": [2.0, 2.5, 3.0, 3.5, 5.0],    "default": [2.5, 3.5, 5.0],   "label": "TP ATR Mult"},
        }

    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_butterworth_atr import run_backtest as bw_backtest
        merged = {**self.frozen_params(), **params}
        return bw_backtest(df, merged)


class ButterworthATR_1H_Strategy(BaseStrategy):
    """
    Same institutional math as ButterworthATRStrategy but with parameter
    ranges scaled for 1-hour bars.

    On 15m you need ~40 bars to see a 10-hour cycle.
    On 1h  you only need ~10 bars for the same cycle.
    So filter periods, ATR lengths, and ROC windows all shrink by roughly 4x,
    then we widen the search range around those scaled values.
    """

    @property
    def name(self) -> str:
        return "BW-ATR All 1h MNQ"

    @property
    def description(self) -> str:
        return (
            "1-hour optimised variant. Causal Butterworth filter + ATR vol-momentum. "
            "Shorter filter periods and wider TP targets suited for hourly swing trades."
        )

    def default_grid(self) -> Dict[str, List]:
        return {
            "bw_fast_period": [4, 6, 8, 10],
            "bw_slow_period": [14, 20, 28, 36],
            "atr_len":        [8, 14, 20],
            "vol_ema_len":    [10, 15, 24],
            "vol_expansion":  [0.8, 1.0, 1.2],
            "atr_roc_len":    [2, 3, 5],
            "sl_atr_mult":    [1.0, 1.5, 2.0],
            "tp_atr_mult":    [2.0, 3.0, 4.0, 6.0],
        }

    def frozen_params(self) -> Dict[str, Any]:
        return {
            "initial_capital": 50000.0,
            "contracts": 2,
            "point_value": 2.0,
            "fee_pct": 0.0015,
            "use_session": False,   # 1H trades can hold through sessions
        }

    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "bw_fast_period": {"options": [3, 4, 5, 6, 8, 10, 12],       "default": [4, 6, 8, 10],      "label": "BW Fast Period"},
            "bw_slow_period": {"options": [10, 14, 18, 20, 28, 36, 48],  "default": [14, 20, 28, 36],   "label": "BW Slow Period"},
            "atr_len":        {"options": [5, 8, 10, 14, 20],            "default": [8, 14, 20],         "label": "ATR Length"},
            "vol_ema_len":    {"options": [8, 10, 15, 20, 24, 30],       "default": [10, 15, 24],        "label": "Vol EMA Length"},
            "vol_expansion":  {"options": [0.6, 0.8, 1.0, 1.2, 1.5],    "default": [0.8, 1.0, 1.2],    "label": "Vol Expansion"},
            "atr_roc_len":    {"options": [2, 3, 4, 5, 8],               "default": [2, 3, 5],           "label": "ATR ROC Length"},
            "sl_atr_mult":    {"options": [0.75, 1.0, 1.5, 2.0, 2.5],   "default": [1.0, 1.5, 2.0],    "label": "SL ATR Mult"},
            "tp_atr_mult":    {"options": [1.5, 2.0, 3.0, 4.0, 6.0, 8.0], "default": [2.0, 3.0, 4.0, 6.0], "label": "TP ATR Mult"},
        }

    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_butterworth_atr import run_backtest as bw_backtest
        merged = {**self.frozen_params(), **params}
        return bw_backtest(df, merged)


class ButterworthATR_Optimized_Strategy(BaseStrategy):
    """User's optimized BW-ATR with TP1/TP2, session filter, prop firm safety."""

    @property
    def name(self) -> str:
        return "BW-ATR Optimized All 1h MNQ"

    @property
    def description(self) -> str:
        return (
            "Optimized: BW 3/5, ATR 10, Vol EMA 75, Expansion 0.6, SL 5x ATR, "
            "TP1 2R + TP2 5R, trail to BE. Session: Tokyo→NY close. Prop firm safe."
        )

    def default_grid(self) -> Dict[str, List]:
        return {
            "bw_fast_period": [3, 4, 5],
            "bw_slow_period": [5, 7, 10],
            "atr_len":        [8, 10, 14],
            "vol_ema_len":    [50, 75, 100],
            "vol_expansion":  [0.4, 0.6, 0.8],
            "atr_roc_len":    [3, 4, 5],
            "sl_atr_mult":    [3.0, 5.0],
            "tp1_rr":         [1.5, 2.0],
            "tp2_rr":         [4.0, 5.0, 6.0],
        }

    def frozen_params(self) -> Dict[str, Any]:
        return {
            "initial_capital": 50000.0,
            "tp1_qty": 1, "tp2_qty": 1,
            "point_value": 2.0, "fee_pct": 0.0015,
            "use_trail": True, "use_force_close": True,
            "session_open_et": 19, "session_close_et": 16,
        }

    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "bw_fast_period": {"options": [3, 4, 5, 6],                "default": [3, 4, 5],         "label": "BW Fast Period"},
            "bw_slow_period": {"options": [5, 7, 10, 14],              "default": [5, 7, 10],        "label": "BW Slow Period"},
            "atr_len":        {"options": [7, 8, 10, 14],              "default": [8, 10, 14],       "label": "ATR Length"},
            "vol_ema_len":    {"options": [40, 50, 75, 100],           "default": [50, 75, 100],     "label": "Vol EMA Length"},
            "vol_expansion":  {"options": [0.3, 0.4, 0.6, 0.8, 1.0],  "default": [0.4, 0.6, 0.8],  "label": "Vol Expansion"},
            "atr_roc_len":    {"options": [2, 3, 4, 5],                "default": [3, 4, 5],         "label": "ATR ROC Length"},
            "sl_atr_mult":    {"options": [2.0, 3.0, 4.0, 5.0],       "default": [3.0, 5.0],        "label": "SL ATR Mult"},
            "tp1_rr":         {"options": [1.5, 2.0, 2.5],             "default": [1.5, 2.0],        "label": "TP1 R:R"},
            "tp2_rr":         {"options": [3.0, 4.0, 5.0, 6.0],       "default": [4.0, 5.0, 6.0],   "label": "TP2 R:R"},
        }

    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_bw_atr_optimized import run_backtest as opt_backtest
        merged = {**self.frozen_params(), **params}
        return opt_backtest(df, merged)


class ORB_1H_Strategy(BaseStrategy):
    """MNQ Opening Range Breakout — 3 contracts: TP1, TP2, trailing runner."""

    @property
    def name(self) -> str:
        return "ORB NY 1h MNQ"

    @property
    def description(self) -> str:
        return (
            "First-hour range breakout. 3 contracts: TP1 (fixed pts), TP2 (fixed pts), "
            "Runner (trailing stop). Volume + ATR filters. One trade/day, EOD flatten."
        )

    def default_grid(self) -> Dict[str, List]:
        return {
            "first_tp_points":          [40.0, 60.0, 80.0],
            "second_tp_points":         [80.0, 110.0, 140.0],
            "trail_distance_points":    [50.0, 75.0, 100.0],
            "runner_be_trigger_points": [75.0, 100.0],
            "max_stop_points":          [150.0, 200.0, 250.0],
            "min_range_width":          [15.0, 25.0],
            "max_range_width":          [150.0, 200.0],
            "volume_multiplier":        [0.5, 0.6, 0.8],
            "min_atr_multiplier":       [0.5, 0.7],
            "max_atr_multiplier":       [2.0, 2.5, 3.0],
        }

    def frozen_params(self) -> Dict[str, Any]:
        return {
            "contracts_per_trade": 3,
            "last_entry_hour": 15,
            "flatten_hour": 16,
            "use_volume_filter": True,
            "volume_lookback": 20,
            "use_atr_filter": True,
            "atr_length": 14,
            "atr_avg_lookback": 20,
            "initial_capital": 50000.0,
            "point_value": 2.0,
            "fee_per_contract": 0.62,
        }

    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "first_tp_points":          {"options": [30.0, 40.0, 50.0, 60.0, 80.0, 100.0],    "default": [40.0, 60.0, 80.0],       "label": "TP1 Points"},
            "second_tp_points":         {"options": [60.0, 80.0, 100.0, 110.0, 140.0, 170.0], "default": [80.0, 110.0, 140.0],     "label": "TP2 Points"},
            "trail_distance_points":    {"options": [40.0, 50.0, 75.0, 100.0, 125.0],         "default": [50.0, 75.0, 100.0],      "label": "Trail Distance"},
            "runner_be_trigger_points": {"options": [50.0, 75.0, 100.0, 125.0],               "default": [75.0, 100.0],             "label": "Runner BE Trigger"},
            "max_stop_points":          {"options": [100.0, 150.0, 200.0, 250.0, 300.0],      "default": [150.0, 200.0, 250.0],    "label": "Max Stop Points"},
            "min_range_width":          {"options": [10.0, 15.0, 20.0, 25.0, 30.0],           "default": [15.0, 25.0],              "label": "Min Range Width"},
            "max_range_width":          {"options": [100.0, 150.0, 200.0, 250.0],             "default": [150.0, 200.0],            "label": "Max Range Width"},
            "volume_multiplier":        {"options": [0.4, 0.5, 0.6, 0.7, 0.8, 1.0],          "default": [0.5, 0.6, 0.8],          "label": "Vol Multiplier"},
            "min_atr_multiplier":       {"options": [0.4, 0.5, 0.6, 0.7, 0.8],               "default": [0.5, 0.7],                "label": "Min ATR Mult"},
            "max_atr_multiplier":       {"options": [1.5, 2.0, 2.5, 3.0],                    "default": [2.0, 2.5, 3.0],           "label": "Max ATR Mult"},
        }

    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_orb_1h import run_backtest as orb_backtest
        merged = {**self.frozen_params(), **params}
        return orb_backtest(df, merged)


# ══════════════════════════════════════════════════════════════════════════════
# Strategy: Tokyo ORB 1H
# ══════════════════════════════════════════════════════════════════════════════

class TokyoORB_1H_Strategy(BaseStrategy):
    """MNQ Tokyo Opening Range Breakout — 6 contracts: TP1(2), TP2(2), Runner(2)."""

    @property
    def name(self) -> str:
        return "ORB Tokyo 1h MNQ"

    @property
    def description(self) -> str:
        return (
            "Tokyo session first-hour range breakout. 6 contracts (2/2/2): TP1, TP2, "
            "Runner (trailing stop). No vol/ATR filters. One trade/day, EOD flatten."
        )

    def default_grid(self) -> Dict[str, List]:
        return {
            "first_tp_points":          [10.0, 15.0, 20.0],
            "second_tp_points":         [18.0, 22.0, 28.0],
            "trail_distance_points":    [10.0, 15.0, 20.0],
            "runner_be_trigger_points": [12.0, 18.0, 24.0],
            "max_stop_points":          [80.0, 100.0, 120.0],
            "min_range_width":          [5.0, 8.0, 12.0],
            "max_range_width":          [50.0, 80.0, 100.0],
        }

    def frozen_params(self) -> Dict[str, Any]:
        return {
            "contracts_per_trade": 6,
            "last_entry_hour": 14,
            "flatten_hour": 15,
            "initial_capital": 50000.0,
            "point_value": 2.0,
            "fee_per_contract": 0.62,
        }

    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "first_tp_points":          {"options": [8.0, 10.0, 12.0, 15.0, 18.0, 20.0],   "default": [10.0, 15.0, 20.0],    "label": "TP1 Points"},
            "second_tp_points":         {"options": [15.0, 18.0, 22.0, 28.0, 35.0],         "default": [18.0, 22.0, 28.0],    "label": "TP2 Points"},
            "trail_distance_points":    {"options": [8.0, 10.0, 12.0, 15.0, 20.0],          "default": [10.0, 15.0, 20.0],    "label": "Trail Distance"},
            "runner_be_trigger_points": {"options": [10.0, 12.0, 15.0, 18.0, 24.0],         "default": [12.0, 18.0],           "label": "Runner BE Trigger"},
            "max_stop_points":          {"options": [60.0, 80.0, 100.0, 120.0, 150.0],      "default": [80.0, 100.0, 120.0],  "label": "Max Stop Points"},
            "min_range_width":          {"options": [3.0, 5.0, 8.0, 12.0, 15.0],            "default": [5.0, 8.0],             "label": "Min Range Width"},
            "max_range_width":          {"options": [40.0, 60.0, 80.0, 100.0],              "default": [60.0, 80.0],           "label": "Max Range Width"},
        }

    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_tokyo_orb_1h import run_backtest as tokyo_orb_backtest
        merged = {**self.frozen_params(), **params}
        return tokyo_orb_backtest(df, merged)


# ══════════════════════════════════════════════════════════════════════════════
# Strategy: London ORB 1H
# ══════════════════════════════════════════════════════════════════════════════

class LondonORB_1H_Strategy(BaseStrategy):
    """MNQ London Opening Range Breakout — 3 contracts: TP1, TP2, Runner."""

    @property
    def name(self) -> str:
        return "ORB London 1h MNQ"

    @property
    def description(self) -> str:
        return (
            "London session first-hour range breakout. 3 contracts (1/1/1): TP1, TP2, "
            "Runner (trailing stop). No vol/ATR filters. One trade/day, EOD flatten."
        )

    def default_grid(self) -> Dict[str, List]:
        return {
            "first_tp_points":          [30.0, 40.0, 50.0],
            "second_tp_points":         [40.0, 50.0, 60.0],
            "trail_distance_points":    [3.0, 5.0, 8.0],
            "runner_be_trigger_points": [30.0, 40.0, 50.0],
            "max_stop_points":          [120.0, 160.0, 200.0],
            "min_range_width":          [10.0, 15.0, 20.0],
            "max_range_width":          [80.0, 100.0, 120.0],
        }

    def frozen_params(self) -> Dict[str, Any]:
        return {
            "contracts_per_trade": 3,
            "last_entry_hour": 12,
            "flatten_hour": 13,
            "initial_capital": 50000.0,
            "point_value": 2.0,
            "fee_per_contract": 0.62,
        }

    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "first_tp_points":          {"options": [20.0, 30.0, 40.0, 50.0, 60.0],         "default": [30.0, 40.0, 50.0],    "label": "TP1 Points"},
            "second_tp_points":         {"options": [30.0, 40.0, 50.0, 60.0, 75.0],         "default": [40.0, 50.0, 60.0],    "label": "TP2 Points"},
            "trail_distance_points":    {"options": [2.0, 3.0, 5.0, 8.0, 10.0],             "default": [3.0, 5.0, 8.0],       "label": "Trail Distance"},
            "runner_be_trigger_points": {"options": [25.0, 30.0, 40.0, 50.0],               "default": [30.0, 50.0],           "label": "Runner BE Trigger"},
            "max_stop_points":          {"options": [100.0, 120.0, 160.0, 200.0, 250.0],    "default": [120.0, 160.0, 200.0], "label": "Max Stop Points"},
            "min_range_width":          {"options": [8.0, 10.0, 15.0, 20.0],                "default": [10.0, 15.0],           "label": "Min Range Width"},
            "max_range_width":          {"options": [60.0, 80.0, 100.0, 120.0],             "default": [80.0, 120.0],          "label": "Max Range Width"},
        }

    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_london_orb_1h import run_backtest as london_orb_backtest
        merged = {**self.frozen_params(), **params}
        return london_orb_backtest(df, merged)


# ══════════════════════════════════════════════════════════════════════════════
# Strategy: NY RBR 1H (Rally Base Rally)
# ══════════════════════════════════════════════════════════════════════════════

class NY_RBR_1H_Strategy(BaseStrategy):
    """MNQ NY Session Rally-Base-Rally — 2 contracts: TP1 + Runner."""

    @property
    def name(self) -> str:
        return "RBR NY 1h MNQ"

    @property
    def description(self) -> str:
        return (
            "EMA 9/21 trend + Rally-Base-Rally pattern + volume spike + EMA touch. "
            "NY session 09:30-15:00. 2 contracts: TP1 (1.5R), Runner (3.6R). Fixed 120pt risk."
        )

    def default_grid(self) -> Dict[str, List]:
        return {
            "fast_len":           [7, 9, 12],
            "slow_len":           [18, 21, 26],
            "vol_multiplier":     [1.0, 1.5, 2.0],
            "rbr_body_ratio":     [0.5, 0.55, 0.6],
            "base_doji_ratio":    [0.4, 0.45, 0.5],
            "fixed_risk_points":  [80.0, 100.0, 120.0, 150.0],
            "first_rr_ratio":     [1.2, 1.5, 2.0],
            "runner_rr_ratio":    [3.0, 3.6, 4.5],
        }

    def frozen_params(self) -> Dict[str, Any]:
        return {
            "contracts_per_trade": 2,
            "vol_lookback": 10,
            "max_base_bars": 3,
            "rbr_lookback": 15,
            "initial_capital": 50000.0,
            "point_value": 2.0,
            "fee_per_contract": 0.62,
        }

    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "fast_len":           {"options": [5, 7, 8, 9, 12],                  "default": [7, 9, 12],              "label": "Fast EMA"},
            "slow_len":           {"options": [15, 18, 21, 26, 30],              "default": [18, 21, 26],            "label": "Slow EMA"},
            "vol_multiplier":     {"options": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5],   "default": [1.0, 1.5, 2.0],        "label": "Vol Multiplier"},
            "rbr_body_ratio":     {"options": [0.45, 0.5, 0.55, 0.6, 0.65],     "default": [0.5, 0.55, 0.6],       "label": "Rally Body Ratio"},
            "base_doji_ratio":    {"options": [0.35, 0.4, 0.45, 0.5],           "default": [0.4, 0.45],             "label": "Base Body Ratio"},
            "fixed_risk_points":  {"options": [60.0, 80.0, 100.0, 120.0, 150.0],"default": [80.0, 100.0, 120.0],   "label": "Fixed Risk Pts"},
            "first_rr_ratio":     {"options": [1.0, 1.2, 1.5, 2.0],             "default": [1.2, 1.5],              "label": "TP1 R:R"},
            "runner_rr_ratio":    {"options": [2.5, 3.0, 3.6, 4.0, 4.5],        "default": [3.0, 3.6],              "label": "Runner R:R"},
        }

    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_ny_rbr_1h import run_backtest as ny_rbr_backtest
        merged = {**self.frozen_params(), **params}
        return ny_rbr_backtest(df, merged)


# ══════════════════════════════════════════════════════════════════════════════
# Strategy: Tokyo RBR 1H (Rally Base Rally)
# ══════════════════════════════════════════════════════════════════════════════

class Tokyo_RBR_1H_Strategy(BaseStrategy):
    """MNQ Tokyo Session Rally-Base-Rally — 2 contracts: TP1 + Runner."""

    @property
    def name(self) -> str:
        return "RBR Tokyo 1h MNQ"

    @property
    def description(self) -> str:
        return (
            "EMA 9/21 trend + Rally-Base-Rally pattern + volume spike + EMA touch. "
            "Tokyo session 09:00-14:00. 2 contracts: TP1 (1.5R), Runner (3.75R). Fixed 90pt risk."
        )

    def default_grid(self) -> Dict[str, List]:
        return {
            "fast_len":           [7, 9, 12],
            "slow_len":           [18, 21, 26],
            "vol_multiplier":     [1.0, 1.5, 2.0],
            "rbr_body_ratio":     [0.5, 0.55, 0.6],
            "base_doji_ratio":    [0.4, 0.45, 0.5],
            "fixed_risk_points":  [60.0, 75.0, 90.0, 110.0],
            "first_rr_ratio":     [1.2, 1.5, 2.0],
            "runner_rr_ratio":    [3.0, 3.75, 4.5],
        }

    def frozen_params(self) -> Dict[str, Any]:
        return {
            "contracts_per_trade": 2,
            "vol_lookback": 10,
            "max_base_bars": 3,
            "rbr_lookback": 15,
            "initial_capital": 50000.0,
            "point_value": 2.0,
            "fee_per_contract": 0.62,
        }

    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "fast_len":           {"options": [5, 7, 8, 9, 12],                  "default": [7, 9, 12],              "label": "Fast EMA"},
            "slow_len":           {"options": [15, 18, 21, 26, 30],              "default": [18, 21, 26],            "label": "Slow EMA"},
            "vol_multiplier":     {"options": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5],   "default": [1.0, 1.5, 2.0],        "label": "Vol Multiplier"},
            "rbr_body_ratio":     {"options": [0.45, 0.5, 0.55, 0.6, 0.65],     "default": [0.5, 0.55, 0.6],       "label": "Rally Body Ratio"},
            "base_doji_ratio":    {"options": [0.35, 0.4, 0.45, 0.5],           "default": [0.4, 0.45],             "label": "Base Body Ratio"},
            "fixed_risk_points":  {"options": [45.0, 60.0, 75.0, 90.0, 110.0],  "default": [60.0, 75.0, 90.0],     "label": "Fixed Risk Pts"},
            "first_rr_ratio":     {"options": [1.0, 1.2, 1.5, 2.0],             "default": [1.2, 1.5],              "label": "TP1 R:R"},
            "runner_rr_ratio":    {"options": [2.5, 3.0, 3.75, 4.0, 4.5],       "default": [3.0, 3.75],             "label": "Runner R:R"},
        }

    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_tokyo_rbr_1h import run_backtest as tokyo_rbr_backtest
        merged = {**self.frozen_params(), **params}
        return tokyo_rbr_backtest(df, merged)


# ══════════════════════════════════════════════════════════════════════════════
# Strategy: London RBR 1H (Rally Base Rally)
# ══════════════════════════════════════════════════════════════════════════════

class London_RBR_1H_Strategy(BaseStrategy):
    """MNQ London Session Rally-Base-Rally — 6 contracts: TP1(3) + Runner(3)."""

    @property
    def name(self) -> str:
        return "RBR London 1h MNQ"

    @property
    def description(self) -> str:
        return (
            "EMA 9/21 trend + Rally-Base-Rally pattern + volume spike + EMA touch. "
            "London session 08:00-13:00. 6 contracts: TP1 (1.4R x3), Runner (3.1R x3). Fixed 70pt risk."
        )

    def default_grid(self) -> Dict[str, List]:
        return {
            "fast_len":           [7, 9, 12],
            "slow_len":           [18, 21, 26],
            "vol_multiplier":     [1.0, 1.5, 2.0],
            "rbr_body_ratio":     [0.5, 0.55, 0.6],
            "base_doji_ratio":    [0.4, 0.45, 0.5],
            "fixed_risk_points":  [45.0, 55.0, 70.0, 90.0],
            "first_rr_ratio":     [1.0, 1.2, 1.4],
            "runner_rr_ratio":    [2.5, 3.1, 4.0],
        }

    def frozen_params(self) -> Dict[str, Any]:
        return {
            "contracts_per_trade": 6,
            "vol_lookback": 10,
            "max_base_bars": 3,
            "rbr_lookback": 15,
            "initial_capital": 50000.0,
            "point_value": 2.0,
            "fee_per_contract": 0.62,
        }

    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "fast_len":           {"options": [5, 7, 8, 9, 12],                  "default": [7, 9, 12],              "label": "Fast EMA"},
            "slow_len":           {"options": [15, 18, 21, 26, 30],              "default": [18, 21, 26],            "label": "Slow EMA"},
            "vol_multiplier":     {"options": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5],   "default": [1.0, 1.5, 2.0],        "label": "Vol Multiplier"},
            "rbr_body_ratio":     {"options": [0.45, 0.5, 0.55, 0.6, 0.65],     "default": [0.5, 0.55, 0.6],       "label": "Rally Body Ratio"},
            "base_doji_ratio":    {"options": [0.35, 0.4, 0.45, 0.5],           "default": [0.4, 0.45],             "label": "Base Body Ratio"},
            "fixed_risk_points":  {"options": [35.0, 45.0, 55.0, 70.0, 90.0],   "default": [45.0, 55.0, 70.0],     "label": "Fixed Risk Pts"},
            "first_rr_ratio":     {"options": [1.0, 1.2, 1.4, 1.6],             "default": [1.2, 1.4],              "label": "TP1 R:R"},
            "runner_rr_ratio":    {"options": [2.0, 2.5, 3.1, 3.5, 4.0],        "default": [2.5, 3.1],              "label": "Runner R:R"},
        }

    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_london_rbr_1h import run_backtest as london_rbr_backtest
        merged = {**self.frozen_params(), **params}
        return london_rbr_backtest(df, merged)


# ══════════════════════════════════════════════════════════════════════════════
# Strategy: Precision Sniper 1H
# ══════════════════════════════════════════════════════════════════════════════

class PrecisionSniper_1H_Strategy(BaseStrategy):
    """Precision Sniper v7.5 adapted for 1H MNQ — EMA crossover + pullback + confluence scoring."""

    @property
    def name(self) -> str:
        return "Precision Sniper NY 1h MNQ"

    @property
    def description(self) -> str:
        return (
            "EMA crossover + pullback entries with 10-point confluence scoring, "
            "Supertrend filter, prop-firm safety. Signal-based exits. 1H timeframe."
        )

    def default_grid(self) -> Dict[str, List]:
        return {
            "ema_fast_len":     [7, 9, 12],
            "ema_slow_len":     [18, 21, 26],
            "ema_trend_len":    [40, 50, 60],
            "min_score":        [4, 5, 6],
            "rsi_len":          [18, 21, 26],
            "fixed_risk_pts":   [40.0, 50.0, 60.0, 80.0],
            "tp1_rr":           [1.5, 2.0, 2.5],
            "tp2_rr":           [3.0, 3.5, 4.5],
            "st_factor":        [3.0, 3.5, 4.0, 4.5],
            "st_atr_len":       [7, 10, 14, 20],
            "max_trade_loss":   [200.0, 300.0, 400.0, 500.0],
            "daily_max_loss":   [800.0, 1000.0, 1200.0, 1500.0],
        }

    def frozen_params(self) -> Dict[str, Any]:
        return {
            "contracts_per_trade": 2,
            "use_pullback": True,
            "pullback_min_score": 4,
            "use_supertrend": True,
            "use_session": True,
            "use_force_close": True,
            "force_close_hour": 16,
            "initial_capital": 50000.0,
            "point_value": 2.0,
            "fee_per_contract": 0.62,
        }

    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "ema_fast_len":     {"options": [5, 7, 8, 9, 10, 12],               "default": [7, 9, 12],            "label": "Fast EMA"},
            "ema_slow_len":     {"options": [15, 18, 21, 26, 30],               "default": [18, 21, 26],          "label": "Slow EMA"},
            "ema_trend_len":    {"options": [35, 40, 50, 55, 60],               "default": [40, 50, 60],          "label": "Trend EMA"},
            "min_score":        {"options": [3, 4, 5, 6, 7],                    "default": [4, 5, 6],             "label": "Min Score"},
            "rsi_len":          {"options": [14, 18, 21, 26],                   "default": [18, 21],              "label": "RSI Length"},
            "fixed_risk_pts":   {"options": [30.0, 40.0, 50.0, 60.0, 80.0, 100.0], "default": [40.0, 60.0, 80.0], "label": "Fixed Risk Pts"},
            "tp1_rr":           {"options": [1.0, 1.5, 2.0, 2.5, 3.0],         "default": [1.5, 2.0],            "label": "TP1 R:R"},
            "tp2_rr":           {"options": [2.5, 3.0, 3.5, 4.0, 4.5, 5.0],    "default": [3.0, 3.5],            "label": "TP2 R:R"},
            "st_factor":        {"options": [2.5, 3.0, 3.5, 4.0, 4.5, 5.0],    "default": [3.5, 4.0],            "label": "Supertrend Factor"},
            "st_atr_len":       {"options": [5, 7, 10, 14, 20, 30],              "default": [7, 10, 14],           "label": "ST ATR Length"},
            "max_trade_loss":   {"options": [150.0, 200.0, 300.0, 400.0, 500.0, 600.0], "default": [300.0, 400.0],  "label": "Max Loss Per Trade ($)"},
            "daily_max_loss":   {"options": [600.0, 800.0, 1000.0, 1200.0, 1500.0, 2000.0], "default": [1000.0, 1200.0], "label": "Daily Max Loss ($)"},
        }

    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_precision_sniper_1h import run_backtest as ps_backtest
        merged = {**self.frozen_params(), **params}
        return ps_backtest(df, merged)


# ══════════════════════════════════════════════════════════════════════════════
# 1-Minute ORB Strategies
# ══════════════════════════════════════════════════════════════════════════════

class ORB_NY_1m_Strategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "ORB NY 1min MNQ"
    @property
    def description(self) -> str:
        return "NY session 15-min opening range breakout on 1-min bars. 3 contracts (1/1/1). Vol+ATR filters."
    def default_grid(self) -> Dict[str, List]:
        return {
            "first_tp_points": [20.0, 25.0, 30.0],
            "second_tp_points": [30.0, 35.0, 40.0],
            "trail_distance_points": [25.0, 30.0, 35.0],
            "runner_be_trigger_points": [30.0, 35.0, 40.0],
            "max_stop_points": [120.0, 145.0, 170.0],
            "min_range_width": [70.0, 90.0, 110.0],
            "max_range_width": [280.0, 340.0, 400.0],
            "volume_multiplier": [0.3, 0.5, 0.7],
            "min_atr_multiplier": [0.7, 0.9, 1.1],
            "max_atr_multiplier": [1.5, 2.0, 2.5],
        }
    def frozen_params(self) -> Dict[str, Any]:
        return {"contracts_per_trade": 3,
                "use_volume_filter": True, "volume_lookback": 20,
                "use_atr_filter": True, "atr_length": 12, "atr_avg_lookback": 40,
                "initial_capital": 50000.0, "point_value": 2.0, "fee_per_contract": 0.62}
    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "first_tp_points":          {"options": [15.0, 20.0, 25.0, 30.0, 35.0],        "default": [20.0, 25.0, 30.0],    "label": "TP1 Points"},
            "second_tp_points":         {"options": [25.0, 30.0, 35.0, 40.0, 45.0],        "default": [30.0, 35.0, 40.0],    "label": "TP2 Points"},
            "trail_distance_points":    {"options": [20.0, 25.0, 30.0, 35.0, 40.0],        "default": [25.0, 30.0, 35.0],    "label": "Runner Trail Distance"},
            "runner_be_trigger_points": {"options": [25.0, 30.0, 35.0, 40.0, 45.0],        "default": [30.0, 35.0, 40.0],    "label": "Runner BE Trigger"},
            "max_stop_points":          {"options": [100.0, 120.0, 145.0, 170.0, 200.0],   "default": [120.0, 145.0, 170.0], "label": "Max Stop Loss"},
            "min_range_width":          {"options": [50.0, 70.0, 90.0, 110.0, 130.0],      "default": [70.0, 90.0, 110.0],   "label": "Min Range Width"},
            "max_range_width":          {"options": [240.0, 280.0, 340.0, 400.0],           "default": [280.0, 340.0, 400.0], "label": "Max Range Width"},
            "volume_multiplier":        {"options": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],       "default": [0.3, 0.5, 0.7],       "label": "Volume Multiplier"},
            "min_atr_multiplier":       {"options": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1],       "default": [0.7, 0.9],            "label": "Min ATR Multiplier"},
            "max_atr_multiplier":       {"options": [1.5, 1.8, 2.0, 2.2, 2.5],            "default": [1.5, 2.0, 2.5],       "label": "Max ATR Multiplier"},
        }
    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_orb_ny_1m import run_backtest
        return run_backtest(df, {**self.frozen_params(), **params})


class ORB_Tokyo_1m_Strategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "ORB Tokyo 1min MNQ"
    @property
    def description(self) -> str:
        return "Tokyo session 15-min opening range breakout on 1-min bars. 6 contracts (2/2/2)."
    def default_grid(self) -> Dict[str, List]:
        return {
            "first_tp_points": [4.0, 6.0, 8.0],
            "second_tp_points": [5.0, 7.0, 9.0],
            "trail_distance_points": [4.0, 6.0, 8.0],
            "runner_be_trigger_points": [4.0, 6.0, 8.0],
            "max_stop_points": [40.0, 59.0, 80.0],
            "min_range_width": [20.0, 34.0, 50.0],
            "max_range_width": [120.0, 160.0, 200.0],
        }
    def frozen_params(self) -> Dict[str, Any]:
        return {"contracts_per_trade": 6,
                "initial_capital": 50000.0, "point_value": 2.0, "fee_per_contract": 0.62}
    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "first_tp_points":          {"options": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],       "default": [4.0, 6.0, 8.0],       "label": "TP1 Points"},
            "second_tp_points":         {"options": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],       "default": [5.0, 7.0, 9.0],       "label": "TP2 Points"},
            "trail_distance_points":    {"options": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],       "default": [4.0, 6.0, 8.0],       "label": "Runner Trail Distance"},
            "runner_be_trigger_points": {"options": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],       "default": [4.0, 6.0, 8.0],       "label": "Runner BE Trigger"},
            "max_stop_points":          {"options": [30.0, 40.0, 50.0, 59.0, 70.0, 80.0], "default": [40.0, 59.0, 80.0],    "label": "Max Stop Loss"},
            "min_range_width":          {"options": [15.0, 20.0, 25.0, 34.0, 40.0, 50.0], "default": [20.0, 34.0, 50.0],    "label": "Min Range Width"},
            "max_range_width":          {"options": [100.0, 120.0, 140.0, 160.0, 180.0, 200.0], "default": [120.0, 160.0, 200.0], "label": "Max Range Width"},
        }
    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_orb_tokyo_1m import run_backtest
        return run_backtest(df, {**self.frozen_params(), **params})


class ORB_London_1m_Strategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "ORB London 1min MNQ"
    @property
    def description(self) -> str:
        return "London session 15-min opening range breakout on 1-min bars. 3 contracts (1/1/1)."
    def default_grid(self) -> Dict[str, List]:
        return {
            "first_tp_points": [15.0, 19.0, 25.0],
            "second_tp_points": [16.0, 20.0, 26.0],
            "trail_distance_points": [1.0, 2.0, 4.0],
            "runner_be_trigger_points": [15.0, 19.0, 25.0],
            "max_stop_points": [100.0, 130.0, 160.0],
            "min_range_width": [35.0, 49.0, 65.0],
            "max_range_width": [200.0, 275.0, 350.0],
        }
    def frozen_params(self) -> Dict[str, Any]:
        return {"contracts_per_trade": 3,
                "initial_capital": 50000.0, "point_value": 2.0, "fee_per_contract": 0.62}
    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "first_tp_points":          {"options": [12.0, 15.0, 19.0, 22.0, 25.0],       "default": [15.0, 19.0, 25.0],    "label": "TP1 Points"},
            "second_tp_points":         {"options": [14.0, 16.0, 20.0, 23.0, 26.0],       "default": [16.0, 20.0, 26.0],    "label": "TP2 Points"},
            "trail_distance_points":    {"options": [1.0, 2.0, 3.0, 4.0, 5.0],            "default": [1.0, 2.0, 4.0],       "label": "Runner Trail Distance"},
            "runner_be_trigger_points": {"options": [12.0, 15.0, 19.0, 22.0, 25.0],       "default": [15.0, 19.0, 25.0],    "label": "Runner BE Trigger"},
            "max_stop_points":          {"options": [80.0, 100.0, 115.0, 130.0, 150.0, 160.0], "default": [100.0, 130.0, 160.0], "label": "Max Stop Loss"},
            "min_range_width":          {"options": [30.0, 35.0, 40.0, 49.0, 55.0, 65.0], "default": [35.0, 49.0, 65.0],    "label": "Min Range Width"},
            "max_range_width":          {"options": [175.0, 200.0, 240.0, 275.0, 320.0, 350.0], "default": [200.0, 275.0, 350.0], "label": "Max Range Width"},
        }
    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_orb_london_1m import run_backtest
        return run_backtest(df, {**self.frozen_params(), **params})


# ══════════════════════════════════════════════════════════════════════════════
# 1-Minute RBR Strategies
# ══════════════════════════════════════════════════════════════════════════════

class RBR_NY_1m_Strategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "RBR NY 1min MNQ"
    @property
    def description(self) -> str:
        return "EMA 8/20 + Rally-Base-Rally, NY 09:30-11:30 on 1-min bars. 2 contracts (1/1). Fixed 65pt risk."
    def default_grid(self) -> Dict[str, List]:
        return {
            "fixed_risk_points": [50.0, 65.0, 80.0],
            "first_rr_ratio": [1.2, 1.5, 2.0],
            "runner_rr_ratio": [3.0, 3.6, 4.5],
            "fast_len": [6, 8, 10],
            "slow_len": [18, 20, 24],
            "vol_multiplier": [2.0, 2.5, 3.0],
            "rbr_body_ratio": [0.5, 0.6, 0.7],
            "base_doji_ratio": [0.3, 0.4, 0.5],
        }
    def frozen_params(self) -> Dict[str, Any]:
        return {"contracts_per_trade": 2, "vol_lookback": 20,
                "max_base_bars": 3, "rbr_lookback": 40,
                "initial_capital": 50000.0, "point_value": 2.0, "fee_per_contract": 0.62}
    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "fixed_risk_points":  {"options": [40.0, 50.0, 55.0, 65.0, 75.0, 80.0, 90.0], "default": [50.0, 65.0, 80.0],    "label": "Fixed Risk (pts)"},
            "first_rr_ratio":     {"options": [1.0, 1.2, 1.5, 1.8, 2.0, 2.5],             "default": [1.2, 1.5, 2.0],       "label": "TP1 R:R"},
            "runner_rr_ratio":    {"options": [2.5, 3.0, 3.3, 3.6, 4.0, 4.5, 5.0],        "default": [3.0, 3.6, 4.5],       "label": "Runner R:R"},
            "fast_len":           {"options": [5, 6, 7, 8, 9, 10, 12],                     "default": [6, 8, 10],             "label": "Fast EMA"},
            "slow_len":           {"options": [15, 18, 20, 22, 24, 26],                     "default": [18, 20, 24],           "label": "Slow EMA"},
            "vol_multiplier":     {"options": [1.5, 2.0, 2.5, 3.0, 3.5],                   "default": [2.0, 2.5, 3.0],       "label": "Volume Spike Mult"},
            "rbr_body_ratio":     {"options": [0.45, 0.5, 0.55, 0.6, 0.65, 0.7],           "default": [0.5, 0.6, 0.7],       "label": "Rally Body Ratio"},
            "base_doji_ratio":    {"options": [0.25, 0.3, 0.35, 0.4, 0.45, 0.5],           "default": [0.3, 0.4, 0.5],       "label": "Base Body Ratio"},
        }
    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_rbr_ny_1m import run_backtest
        return run_backtest(df, {**self.frozen_params(), **params})


class RBR_Tokyo_1m_Strategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "RBR Tokyo 1min MNQ"
    @property
    def description(self) -> str:
        return "EMA 8/20 + Rally-Base-Rally, Tokyo 09:00-11:00 on 1-min bars. 2 contracts (1/1). Fixed 47pt risk."
    def default_grid(self) -> Dict[str, List]:
        return {
            "fixed_risk_points": [35.0, 47.0, 60.0],
            "first_rr_ratio": [1.2, 1.5, 2.0],
            "runner_rr_ratio": [3.0, 3.75, 4.5],
            "fast_len": [6, 8, 10],
            "slow_len": [18, 20, 24],
            "vol_multiplier": [2.0, 2.5, 3.0],
            "rbr_body_ratio": [0.5, 0.6, 0.7],
            "base_doji_ratio": [0.3, 0.4, 0.5],
        }
    def frozen_params(self) -> Dict[str, Any]:
        return {"contracts_per_trade": 2, "vol_lookback": 20,
                "max_base_bars": 3, "rbr_lookback": 40,
                "initial_capital": 50000.0, "point_value": 2.0, "fee_per_contract": 0.62}
    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "fixed_risk_points":  {"options": [25.0, 35.0, 40.0, 47.0, 55.0, 60.0, 70.0], "default": [35.0, 47.0, 60.0],    "label": "Fixed Risk (pts)"},
            "first_rr_ratio":     {"options": [1.0, 1.2, 1.5, 1.8, 2.0, 2.5],             "default": [1.2, 1.5, 2.0],       "label": "TP1 R:R"},
            "runner_rr_ratio":    {"options": [2.5, 3.0, 3.5, 3.75, 4.0, 4.5, 5.0],       "default": [3.0, 3.75, 4.5],      "label": "Runner R:R"},
            "fast_len":           {"options": [5, 6, 7, 8, 9, 10, 12],                     "default": [6, 8, 10],             "label": "Fast EMA"},
            "slow_len":           {"options": [15, 18, 20, 22, 24, 26],                     "default": [18, 20, 24],           "label": "Slow EMA"},
            "vol_multiplier":     {"options": [1.5, 2.0, 2.5, 3.0, 3.5],                   "default": [2.0, 2.5, 3.0],       "label": "Volume Spike Mult"},
            "rbr_body_ratio":     {"options": [0.45, 0.5, 0.55, 0.6, 0.65, 0.7],           "default": [0.5, 0.6, 0.7],       "label": "Rally Body Ratio"},
            "base_doji_ratio":    {"options": [0.25, 0.3, 0.35, 0.4, 0.45, 0.5],           "default": [0.3, 0.4, 0.5],       "label": "Base Body Ratio"},
        }
    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_rbr_tokyo_1m import run_backtest
        return run_backtest(df, {**self.frozen_params(), **params})


class RBR_London_1m_Strategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "RBR London 1min MNQ"
    @property
    def description(self) -> str:
        return "EMA 8/20 + Rally-Base-Rally, London 08:00-10:00 on 1-min bars. 6 contracts (3/3). Fixed 30pt risk."
    def default_grid(self) -> Dict[str, List]:
        return {
            "fixed_risk_points": [20.0, 30.0, 40.0],
            "first_rr_ratio": [1.0, 1.2, 1.4],
            "runner_rr_ratio": [2.5, 3.1, 4.0],
            "fast_len": [6, 8, 10],
            "slow_len": [18, 20, 24],
            "vol_multiplier": [2.0, 2.5, 3.0],
            "rbr_body_ratio": [0.5, 0.6, 0.7],
            "base_doji_ratio": [0.3, 0.4, 0.5],
        }
    def frozen_params(self) -> Dict[str, Any]:
        return {"contracts_per_trade": 6, "vol_lookback": 20,
                "max_base_bars": 3, "rbr_lookback": 40,
                "initial_capital": 50000.0, "point_value": 2.0, "fee_per_contract": 0.62}
    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "fixed_risk_points":  {"options": [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0], "default": [20.0, 30.0, 40.0],    "label": "Fixed Risk (pts)"},
            "first_rr_ratio":     {"options": [0.8, 1.0, 1.2, 1.4, 1.6, 1.8],             "default": [1.0, 1.2, 1.4],       "label": "TP1 R:R"},
            "runner_rr_ratio":    {"options": [2.0, 2.5, 3.0, 3.1, 3.5, 4.0, 4.5],        "default": [2.5, 3.1, 4.0],       "label": "Runner R:R"},
            "fast_len":           {"options": [5, 6, 7, 8, 9, 10, 12],                     "default": [6, 8, 10],             "label": "Fast EMA"},
            "slow_len":           {"options": [15, 18, 20, 22, 24, 26],                     "default": [18, 20, 24],           "label": "Slow EMA"},
            "vol_multiplier":     {"options": [1.5, 2.0, 2.5, 3.0, 3.5],                   "default": [2.0, 2.5, 3.0],       "label": "Volume Spike Mult"},
            "rbr_body_ratio":     {"options": [0.45, 0.5, 0.55, 0.6, 0.65, 0.7],           "default": [0.5, 0.6, 0.7],       "label": "Rally Body Ratio"},
            "base_doji_ratio":    {"options": [0.25, 0.3, 0.35, 0.4, 0.45, 0.5],           "default": [0.3, 0.4, 0.5],       "label": "Base Body Ratio"},
        }
    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_rbr_london_1m import run_backtest
        return run_backtest(df, {**self.frozen_params(), **params})


# ══════════════════════════════════════════════════════════════════════════════
# Combined All-Sessions Strategy
# ══════════════════════════════════════════════════════════════════════════════

class Combined_All_1m_Strategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "Combined All Sessions 1min MNQ"
    @property
    def description(self) -> str:
        return (
            "All 6 strategies (ORB + RBR for NY, Tokyo, London) on 1-min bars. "
            "Direction lock: once one strategy enters, others can only add in the same direction."
        )
    def default_grid(self) -> Dict[str, List]:
        return {
            "orb_ny_first_tp_points": [20.0, 25.0, 30.0],
            "orb_ny_second_tp_points": [30.0, 35.0, 40.0],
            "orb_tk_first_tp_points": [4.0, 6.0, 8.0],
            "orb_tk_second_tp_points": [5.0, 7.0, 9.0],
            "rbr_ny_fixed_risk_points": [50.0, 65.0, 80.0],
            "rbr_tk_fixed_risk_points": [35.0, 47.0, 60.0],
            "rbr_ld_fixed_risk_points": [20.0, 30.0, 40.0],
        }
    def frozen_params(self) -> Dict[str, Any]:
        return {
            # ORB NY
            "orb_ny_trail_distance_points": 30.0, "orb_ny_runner_be_trigger_points": 35.0,
            "orb_ny_contracts_per_trade": 3, "orb_ny_max_stop_points": 145.0,
            "orb_ny_min_range_width": 90.0, "orb_ny_max_range_width": 340.0,
            "orb_ny_use_volume_filter": True, "orb_ny_volume_lookback": 20, "orb_ny_volume_multiplier": 0.5,
            "orb_ny_use_atr_filter": True, "orb_ny_atr_length": 12, "orb_ny_atr_avg_lookback": 40,
            "orb_ny_min_atr_multiplier": 0.9, "orb_ny_max_atr_multiplier": 2.0,
            # ORB Tokyo
            "orb_tk_trail_distance_points": 6.0, "orb_tk_runner_be_trigger_points": 6.0,
            "orb_tk_contracts_per_trade": 6, "orb_tk_max_stop_points": 59.0,
            "orb_tk_min_range_width": 34.0, "orb_tk_max_range_width": 160.0,
            # ORB London
            "orb_ld_first_tp_points": 19.0, "orb_ld_second_tp_points": 20.0,
            "orb_ld_trail_distance_points": 2.0, "orb_ld_runner_be_trigger_points": 19.0,
            "orb_ld_contracts_per_trade": 3, "orb_ld_max_stop_points": 130.0,
            "orb_ld_min_range_width": 49.0, "orb_ld_max_range_width": 275.0,
            # RBR NY
            "rbr_ny_first_rr_ratio": 1.5, "rbr_ny_runner_rr_ratio": 3.6,
            "rbr_ny_contracts_per_trade": 2, "rbr_ny_fast_len": 8, "rbr_ny_slow_len": 20,
            "rbr_ny_vol_lookback": 20, "rbr_ny_vol_multiplier": 2.5,
            "rbr_ny_rbr_body_ratio": 0.6, "rbr_ny_base_doji_ratio": 0.4,
            "rbr_ny_max_base_bars": 3, "rbr_ny_rbr_lookback": 40,
            # RBR Tokyo
            "rbr_tk_first_rr_ratio": 1.5, "rbr_tk_runner_rr_ratio": 3.75,
            "rbr_tk_contracts_per_trade": 2, "rbr_tk_fast_len": 8, "rbr_tk_slow_len": 20,
            "rbr_tk_vol_lookback": 20, "rbr_tk_vol_multiplier": 2.5,
            "rbr_tk_rbr_body_ratio": 0.6, "rbr_tk_base_doji_ratio": 0.4,
            "rbr_tk_max_base_bars": 3, "rbr_tk_rbr_lookback": 40,
            # RBR London
            "rbr_ld_first_rr_ratio": 1.4, "rbr_ld_runner_rr_ratio": 3.1,
            "rbr_ld_contracts_per_trade": 6, "rbr_ld_fast_len": 8, "rbr_ld_slow_len": 20,
            "rbr_ld_vol_lookback": 20, "rbr_ld_vol_multiplier": 2.5,
            "rbr_ld_rbr_body_ratio": 0.6, "rbr_ld_base_doji_ratio": 0.4,
            "rbr_ld_max_base_bars": 3, "rbr_ld_rbr_lookback": 40,
            # General
            "initial_capital": 50000.0, "point_value": 2.0, "fee_per_contract": 0.62,
        }
    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "orb_ny_first_tp_points":      {"options": [15.0, 20.0, 25.0, 30.0, 35.0],   "default": [20.0, 25.0, 30.0],  "label": "ORB NY TP1"},
            "orb_ny_second_tp_points":     {"options": [25.0, 30.0, 35.0, 40.0, 45.0],   "default": [30.0, 35.0, 40.0],  "label": "ORB NY TP2"},
            "orb_tk_first_tp_points":      {"options": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],  "default": [4.0, 6.0, 8.0],     "label": "ORB Tokyo TP1"},
            "orb_tk_second_tp_points":     {"options": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],  "default": [5.0, 7.0, 9.0],     "label": "ORB Tokyo TP2"},
            "rbr_ny_fixed_risk_points":    {"options": [40.0, 50.0, 65.0, 80.0, 100.0],  "default": [50.0, 65.0, 80.0],  "label": "RBR NY Risk"},
            "rbr_tk_fixed_risk_points":    {"options": [25.0, 35.0, 47.0, 60.0, 75.0],   "default": [35.0, 47.0, 60.0],  "label": "RBR Tokyo Risk"},
            "rbr_ld_fixed_risk_points":    {"options": [15.0, 20.0, 30.0, 40.0, 50.0],   "default": [20.0, 30.0, 40.0],  "label": "RBR London Risk"},
        }
    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_combined_all_1m import run_backtest
        return run_backtest(df, {**self.frozen_params(), **params})


# ══════════════════════════════════════════════════════════════════════════════
# MYM 1-Minute ORB Strategies
# ══════════════════════════════════════════════════════════════════════════════

class MYM_ORB_NY_1m_Strategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "ORB NY 1min MYM"
    @property
    def description(self) -> str:
        return "NY session 15-min opening range breakout on 1-min bars. 6 contracts (2/2/2). Vol+ATR filters. MYM."
    def default_grid(self) -> Dict[str, List]:
        return {
            "first_tp_points": [33.0, 43.0, 53.0],
            "second_tp_points": [35.0, 45.0, 55.0],
            "trail_distance_points": [45.0, 55.0, 65.0],
            "runner_be_trigger_points": [35.0, 45.0, 55.0],
            "max_stop_points": [140.0, 180.0, 220.0],
            "min_range_width": [85.0, 115.0, 145.0],
            "max_range_width": [165.0, 205.0, 245.0],
            "volume_multiplier": [0.3, 0.5, 0.7],
            "min_atr_multiplier": [0.7, 0.9, 1.1],
            "max_atr_multiplier": [1.5, 2.0, 2.5],
        }
    def frozen_params(self) -> Dict[str, Any]:
        return {"contracts_per_trade": 6,
                "use_volume_filter": True, "volume_lookback": 20,
                "use_atr_filter": True, "atr_length": 12, "atr_avg_lookback": 40,
                "initial_capital": 50000.0, "point_value": 0.50, "fee_per_contract": 0.62}
    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "first_tp_points":          {"options": [28.0, 33.0, 38.0, 43.0, 48.0, 53.0],     "default": [33.0, 43.0, 53.0],    "label": "TP1 Points"},
            "second_tp_points":         {"options": [30.0, 35.0, 40.0, 45.0, 50.0, 55.0],     "default": [35.0, 45.0, 55.0],    "label": "TP2 Points"},
            "trail_distance_points":    {"options": [40.0, 45.0, 50.0, 55.0, 60.0, 65.0],     "default": [45.0, 55.0, 65.0],    "label": "Runner Trail Distance"},
            "runner_be_trigger_points": {"options": [30.0, 35.0, 40.0, 45.0, 50.0, 55.0],     "default": [35.0, 45.0, 55.0],    "label": "Runner BE Trigger"},
            "max_stop_points":          {"options": [120.0, 140.0, 160.0, 180.0, 200.0, 220.0], "default": [140.0, 180.0, 220.0], "label": "Max Stop Loss"},
            "min_range_width":          {"options": [75.0, 85.0, 100.0, 115.0, 130.0, 145.0], "default": [85.0, 115.0, 145.0],  "label": "Min Range Width"},
            "max_range_width":          {"options": [150.0, 165.0, 185.0, 205.0, 225.0, 245.0], "default": [165.0, 205.0, 245.0], "label": "Max Range Width"},
            "volume_multiplier":        {"options": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],          "default": [0.3, 0.5, 0.7],       "label": "Volume Multiplier"},
            "min_atr_multiplier":       {"options": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1],          "default": [0.7, 0.9, 1.1],       "label": "Min ATR Multiplier"},
            "max_atr_multiplier":       {"options": [1.5, 1.8, 2.0, 2.2, 2.5],               "default": [1.5, 2.0, 2.5],       "label": "Max ATR Multiplier"},
        }
    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_mym_orb_ny_1m import run_backtest
        return run_backtest(df, {**self.frozen_params(), **params})


class MYM_ORB_Tokyo_1m_Strategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "ORB Tokyo 1min MYM"
    @property
    def description(self) -> str:
        return "Tokyo session 15-min opening range breakout on 1-min bars. 6 contracts (2/2/2). MYM."
    def default_grid(self) -> Dict[str, List]:
        return {
            "first_tp_points": [35.0, 45.0, 55.0],
            "second_tp_points": [40.0, 50.0, 60.0],
            "trail_distance_points": [35.0, 45.0, 55.0],
            "runner_be_trigger_points": [40.0, 50.0, 60.0],
            "max_stop_points": [50.0, 70.0, 90.0],
            "min_range_width": [31.0, 41.0, 55.0],
            "max_range_width": [90.0, 120.0, 150.0],
        }
    def frozen_params(self) -> Dict[str, Any]:
        return {"contracts_per_trade": 6,
                "initial_capital": 50000.0, "point_value": 0.50, "fee_per_contract": 0.62}
    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "first_tp_points":          {"options": [30.0, 35.0, 40.0, 45.0, 50.0, 55.0],     "default": [35.0, 45.0, 55.0],    "label": "TP1 Points"},
            "second_tp_points":         {"options": [35.0, 40.0, 45.0, 50.0, 55.0, 60.0],     "default": [40.0, 50.0, 60.0],    "label": "TP2 Points"},
            "trail_distance_points":    {"options": [30.0, 35.0, 40.0, 45.0, 50.0, 55.0],     "default": [35.0, 45.0, 55.0],    "label": "Runner Trail Distance"},
            "runner_be_trigger_points": {"options": [35.0, 40.0, 45.0, 50.0, 55.0, 60.0],     "default": [40.0, 50.0, 60.0],    "label": "Runner BE Trigger"},
            "max_stop_points":          {"options": [40.0, 50.0, 60.0, 70.0, 80.0, 90.0],     "default": [50.0, 70.0, 90.0],    "label": "Max Stop Loss"},
            "min_range_width":          {"options": [25.0, 31.0, 36.0, 41.0, 48.0, 55.0],     "default": [31.0, 41.0, 55.0],    "label": "Min Range Width"},
            "max_range_width":          {"options": [80.0, 90.0, 105.0, 120.0, 135.0, 150.0], "default": [90.0, 120.0, 150.0],  "label": "Max Range Width"},
        }
    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_mym_orb_tokyo_1m import run_backtest
        return run_backtest(df, {**self.frozen_params(), **params})


class MYM_ORB_London_1m_Strategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "ORB London 1min MYM"
    @property
    def description(self) -> str:
        return "London session 15-min opening range breakout on 1-min bars. 6 contracts (2/2/2). MYM."
    def default_grid(self) -> Dict[str, List]:
        return {
            "first_tp_points": [15.0, 25.0, 35.0],
            "second_tp_points": [20.0, 30.0, 40.0],
            "trail_distance_points": [12.0, 20.0, 28.0],
            "runner_be_trigger_points": [20.0, 30.0, 40.0],
            "max_stop_points": [57.0, 77.0, 97.0],
            "min_range_width": [40.0, 60.0, 80.0],
            "max_range_width": [85.0, 115.0, 145.0],
        }
    def frozen_params(self) -> Dict[str, Any]:
        return {"contracts_per_trade": 6,
                "initial_capital": 50000.0, "point_value": 0.50, "fee_per_contract": 0.62}
    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "first_tp_points":          {"options": [15.0, 20.0, 25.0, 30.0, 35.0],           "default": [15.0, 25.0, 35.0],    "label": "TP1 Points"},
            "second_tp_points":         {"options": [18.0, 22.0, 26.0, 30.0, 35.0, 40.0],     "default": [20.0, 30.0, 40.0],    "label": "TP2 Points"},
            "trail_distance_points":    {"options": [10.0, 14.0, 18.0, 20.0, 24.0, 28.0],     "default": [12.0, 20.0, 28.0],    "label": "Runner Trail Distance"},
            "runner_be_trigger_points": {"options": [15.0, 20.0, 25.0, 30.0, 35.0, 40.0],     "default": [20.0, 30.0, 40.0],    "label": "Runner BE Trigger"},
            "max_stop_points":          {"options": [47.0, 57.0, 67.0, 77.0, 87.0, 97.0],     "default": [57.0, 77.0, 97.0],    "label": "Max Stop Loss"},
            "min_range_width":          {"options": [35.0, 40.0, 50.0, 60.0, 70.0, 80.0],     "default": [40.0, 60.0, 80.0],    "label": "Min Range Width"},
            "max_range_width":          {"options": [75.0, 85.0, 100.0, 115.0, 130.0, 145.0], "default": [85.0, 115.0, 145.0],  "label": "Max Range Width"},
        }
    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_mym_orb_london_1m import run_backtest
        return run_backtest(df, {**self.frozen_params(), **params})


# ══════════════════════════════════════════════════════════════════════════════
# MYM 1-Minute RBR Strategies
# ══════════════════════════════════════════════════════════════════════════════

class MYM_RBR_NY_1m_Strategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "RBR NY 1min MYM"
    @property
    def description(self) -> str:
        return "EMA 8/20 + Rally-Base-Rally, NY 09:30-11:30 on 1-min bars. 8 contracts (4/4). Fixed 35pt risk. MYM."
    def default_grid(self) -> Dict[str, List]:
        return {
            "fixed_risk_points": [25.0, 35.0, 45.0],
            "first_rr_ratio": [1.0, 1.4, 1.8],
            "runner_rr_ratio": [2.8, 3.4, 4.0],
            "fast_len": [6, 8, 10],
            "slow_len": [18, 20, 24],
            "vol_multiplier": [1.3, 1.7, 2.1],
            "rbr_body_ratio": [0.5, 0.6, 0.7],
            "base_doji_ratio": [0.3, 0.4, 0.5],
        }
    def frozen_params(self) -> Dict[str, Any]:
        return {"contracts_per_trade": 8, "tp1_contracts": 4, "vol_lookback": 45,
                "max_base_bars": 3, "rbr_lookback": 40,
                "rbr_body_ratio": 0.6, "base_doji_ratio": 0.4,
                "initial_capital": 50000.0, "point_value": 0.50, "fee_per_contract": 0.62}
    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "fixed_risk_points":  {"options": [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0],    "default": [25.0, 35.0, 45.0],    "label": "Fixed Risk (pts)"},
            "first_rr_ratio":     {"options": [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],           "default": [1.0, 1.4, 1.8],       "label": "TP1 R:R"},
            "runner_rr_ratio":    {"options": [2.4, 2.8, 3.0, 3.4, 3.8, 4.0, 4.5],           "default": [2.8, 3.4, 4.0],       "label": "Runner R:R"},
            "fast_len":           {"options": [5, 6, 7, 8, 9, 10, 12],                        "default": [6, 8, 10],             "label": "Fast EMA"},
            "slow_len":           {"options": [15, 18, 20, 22, 24, 26],                        "default": [18, 20, 24],           "label": "Slow EMA"},
            "vol_multiplier":     {"options": [1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.5],           "default": [1.3, 1.7, 2.1],       "label": "Volume Spike Mult"},
            "rbr_body_ratio":     {"options": [0.45, 0.5, 0.55, 0.6, 0.65, 0.7],              "default": [0.5, 0.6, 0.7],       "label": "Rally Body Ratio"},
            "base_doji_ratio":    {"options": [0.25, 0.3, 0.35, 0.4, 0.45, 0.5],              "default": [0.3, 0.4, 0.5],       "label": "Base Body Ratio"},
        }
    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_mym_rbr_ny_1m import run_backtest
        return run_backtest(df, {**self.frozen_params(), **params})


class MYM_RBR_Tokyo_1m_Strategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "RBR Tokyo 1min MYM"
    @property
    def description(self) -> str:
        return "EMA 5/20 + Rally-Base-Rally, Tokyo 09:00-11:00 on 1-min bars. 6 contracts (2/2/2). Fixed 14pt risk. MYM."
    def default_grid(self) -> Dict[str, List]:
        return {
            "fixed_risk_points": [10.0, 14.0, 18.0],
            "first_rr_ratio": [0.8, 1.2, 1.6],
            "runner_rr_ratio": [2.5, 3.1, 3.7],
            "fast_len": [4, 5, 7],
            "slow_len": [18, 20, 24],
            "vol_multiplier": [1.8, 2.3, 2.8],
            "rbr_body_ratio": [0.5, 0.6, 0.7],
            "base_doji_ratio": [0.3, 0.4, 0.5],
        }
    def frozen_params(self) -> Dict[str, Any]:
        return {"contracts_per_trade": 6, "tp1_contracts": 2, "tp2_contracts": 2,
                "vol_lookback": 20, "max_base_bars": 3, "rbr_lookback": 40,
                "rbr_body_ratio": 0.6, "base_doji_ratio": 0.4,
                "initial_capital": 50000.0, "point_value": 0.50, "fee_per_contract": 0.62}
    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "fixed_risk_points":  {"options": [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 22.0],     "default": [10.0, 14.0, 18.0],    "label": "Fixed Risk (pts)"},
            "first_rr_ratio":     {"options": [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],           "default": [0.8, 1.2, 1.6],       "label": "TP1 R:R"},
            "runner_rr_ratio":    {"options": [2.0, 2.5, 2.8, 3.1, 3.4, 3.7, 4.0],           "default": [2.5, 3.1, 3.7],       "label": "Runner R:R"},
            "fast_len":           {"options": [3, 4, 5, 6, 7, 8],                             "default": [4, 5, 7],              "label": "Fast EMA"},
            "slow_len":           {"options": [15, 18, 20, 22, 24, 26],                        "default": [18, 20, 24],           "label": "Slow EMA"},
            "vol_multiplier":     {"options": [1.5, 1.8, 2.0, 2.3, 2.5, 2.8, 3.0],           "default": [1.8, 2.3, 2.8],       "label": "Volume Spike Mult"},
            "rbr_body_ratio":     {"options": [0.45, 0.5, 0.55, 0.6, 0.65, 0.7],              "default": [0.5, 0.6, 0.7],       "label": "Rally Body Ratio"},
            "base_doji_ratio":    {"options": [0.25, 0.3, 0.35, 0.4, 0.45, 0.5],              "default": [0.3, 0.4, 0.5],       "label": "Base Body Ratio"},
        }
    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_mym_rbr_tokyo_1m import run_backtest
        return run_backtest(df, {**self.frozen_params(), **params})


class MYM_RBR_London_1m_Strategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "RBR London 1min MYM"
    @property
    def description(self) -> str:
        return "EMA 7/21 + Rally-Base-Rally, London 08:00-10:00 on 1-min bars. 6 contracts (2/2/2). Fixed 10pt risk. MYM."
    def default_grid(self) -> Dict[str, List]:
        return {
            "fixed_risk_points": [6.0, 10.0, 14.0],
            "first_rr_ratio": [1.1, 1.5, 1.9],
            "runner_rr_ratio": [2.8, 3.5, 4.2],
            "fast_len": [5, 7, 9],
            "slow_len": [18, 21, 24],
            "vol_multiplier": [1.6, 2.1, 2.6],
            "rbr_body_ratio": [0.5, 0.6, 0.7],
            "base_doji_ratio": [0.3, 0.4, 0.5],
        }
    def frozen_params(self) -> Dict[str, Any]:
        return {"contracts_per_trade": 6, "tp1_contracts": 2, "tp2_contracts": 2,
                "vol_lookback": 5, "max_base_bars": 3, "rbr_lookback": 40,
                "rbr_body_ratio": 0.6, "base_doji_ratio": 0.4,
                "initial_capital": 50000.0, "point_value": 0.50, "fee_per_contract": 0.62}
    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "fixed_risk_points":  {"options": [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 18.0],       "default": [6.0, 10.0, 14.0],     "label": "Fixed Risk (pts)"},
            "first_rr_ratio":     {"options": [0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1],           "default": [1.1, 1.5, 1.9],       "label": "TP1 R:R"},
            "runner_rr_ratio":    {"options": [2.2, 2.8, 3.0, 3.5, 3.8, 4.2, 4.5],           "default": [2.8, 3.5, 4.2],       "label": "Runner R:R"},
            "fast_len":           {"options": [4, 5, 6, 7, 8, 9, 10],                         "default": [5, 7, 9],              "label": "Fast EMA"},
            "slow_len":           {"options": [15, 18, 20, 21, 23, 24, 26],                    "default": [18, 21, 24],           "label": "Slow EMA"},
            "vol_multiplier":     {"options": [1.3, 1.6, 1.8, 2.1, 2.4, 2.6, 3.0],           "default": [1.6, 2.1, 2.6],       "label": "Volume Spike Mult"},
            "rbr_body_ratio":     {"options": [0.45, 0.5, 0.55, 0.6, 0.65, 0.7],              "default": [0.5, 0.6, 0.7],       "label": "Rally Body Ratio"},
            "base_doji_ratio":    {"options": [0.25, 0.3, 0.35, 0.4, 0.45, 0.5],              "default": [0.3, 0.4, 0.5],       "label": "Base Body Ratio"},
        }
    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_mym_rbr_london_1m import run_backtest
        return run_backtest(df, {**self.frozen_params(), **params})


# ══════════════════════════════════════════════════════════════════════════════
# MYM Combined All-Sessions Strategy
# ══════════════════════════════════════════════════════════════════════════════

class MYM_Combined_All_1m_Strategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "Combined All 6 MYM 1min"
    @property
    def description(self) -> str:
        return (
            "All 6 MYM sessions combined (ORB NY/Tokyo/London + RBR NY/Tokyo/London) "
            "with direction lock. 1-min bars."
        )
    def default_grid(self) -> Dict[str, List]:
        return {
            "orb_ny_first_tp_points": [33.0, 43.0, 53.0],
            "orb_ny_second_tp_points": [35.0, 45.0, 55.0],
            "orb_tk_first_tp_points": [35.0, 45.0, 55.0],
            "orb_tk_second_tp_points": [40.0, 50.0, 60.0],
            "rbr_ny_fixed_risk_points": [25.0, 35.0, 45.0],
            "rbr_tk_fixed_risk_points": [10.0, 14.0, 18.0],
            "rbr_ld_fixed_risk_points": [6.0, 10.0, 14.0],
        }
    def frozen_params(self) -> Dict[str, Any]:
        return {
            # ORB NY
            "orb_ny_trail_distance_points": 55.0, "orb_ny_runner_be_trigger_points": 45.0,
            "orb_ny_contracts_per_trade": 6, "orb_ny_max_stop_points": 180.0,
            "orb_ny_min_range_width": 115.0, "orb_ny_max_range_width": 205.0,
            "orb_ny_use_volume_filter": True, "orb_ny_volume_lookback": 20, "orb_ny_volume_multiplier": 0.5,
            "orb_ny_use_atr_filter": True, "orb_ny_atr_length": 12, "orb_ny_atr_avg_lookback": 40,
            "orb_ny_min_atr_multiplier": 0.9, "orb_ny_max_atr_multiplier": 2.0,
            # ORB Tokyo
            "orb_tk_trail_distance_points": 45.0, "orb_tk_runner_be_trigger_points": 50.0,
            "orb_tk_contracts_per_trade": 6, "orb_tk_max_stop_points": 70.0,
            "orb_tk_min_range_width": 41.0, "orb_tk_max_range_width": 120.0,
            # ORB London
            "orb_ld_first_tp_points": 25.0, "orb_ld_second_tp_points": 30.0,
            "orb_ld_trail_distance_points": 20.0, "orb_ld_runner_be_trigger_points": 30.0,
            "orb_ld_contracts_per_trade": 6, "orb_ld_max_stop_points": 77.0,
            "orb_ld_min_range_width": 60.0, "orb_ld_max_range_width": 115.0,
            # RBR NY
            "rbr_ny_first_rr_ratio": 1.4, "rbr_ny_runner_rr_ratio": 3.4,
            "rbr_ny_contracts_per_trade": 8, "rbr_ny_tp1_contracts": 4,
            "rbr_ny_fast_len": 8, "rbr_ny_slow_len": 20,
            "rbr_ny_vol_lookback": 45, "rbr_ny_vol_multiplier": 1.7,
            "rbr_ny_rbr_body_ratio": 0.6, "rbr_ny_base_doji_ratio": 0.4,
            "rbr_ny_max_base_bars": 3, "rbr_ny_rbr_lookback": 40,
            # RBR Tokyo
            "rbr_tk_first_rr_ratio": 1.2, "rbr_tk_runner_rr_ratio": 3.1,
            "rbr_tk_contracts_per_trade": 6, "rbr_tk_tp1_contracts": 2, "rbr_tk_tp2_contracts": 2,
            "rbr_tk_fast_len": 5, "rbr_tk_slow_len": 20,
            "rbr_tk_vol_lookback": 20, "rbr_tk_vol_multiplier": 2.3,
            "rbr_tk_rbr_body_ratio": 0.6, "rbr_tk_base_doji_ratio": 0.4,
            "rbr_tk_max_base_bars": 3, "rbr_tk_rbr_lookback": 40,
            # RBR London
            "rbr_ld_first_rr_ratio": 1.5, "rbr_ld_runner_rr_ratio": 3.5,
            "rbr_ld_contracts_per_trade": 6, "rbr_ld_tp1_contracts": 2, "rbr_ld_tp2_contracts": 2,
            "rbr_ld_fast_len": 7, "rbr_ld_slow_len": 21,
            "rbr_ld_vol_lookback": 5, "rbr_ld_vol_multiplier": 2.1,
            "rbr_ld_rbr_body_ratio": 0.6, "rbr_ld_base_doji_ratio": 0.4,
            "rbr_ld_max_base_bars": 3, "rbr_ld_rbr_lookback": 40,
            # General
            "initial_capital": 50000.0, "point_value": 0.50, "fee_per_contract": 0.62,
        }
    def sidebar_grid_options(self) -> Dict[str, Dict]:
        return {
            "orb_ny_first_tp_points":      {"options": [28.0, 33.0, 38.0, 43.0, 48.0, 53.0], "default": [33.0, 43.0, 53.0],  "label": "ORB NY TP1"},
            "orb_ny_second_tp_points":     {"options": [30.0, 35.0, 40.0, 45.0, 50.0, 55.0], "default": [35.0, 45.0, 55.0],  "label": "ORB NY TP2"},
            "orb_tk_first_tp_points":      {"options": [30.0, 35.0, 40.0, 45.0, 50.0, 55.0], "default": [35.0, 45.0, 55.0],  "label": "ORB Tokyo TP1"},
            "orb_tk_second_tp_points":     {"options": [35.0, 40.0, 45.0, 50.0, 55.0, 60.0], "default": [40.0, 50.0, 60.0],  "label": "ORB Tokyo TP2"},
            "rbr_ny_fixed_risk_points":    {"options": [20.0, 25.0, 30.0, 35.0, 40.0, 45.0], "default": [25.0, 35.0, 45.0],  "label": "RBR NY Risk"},
            "rbr_tk_fixed_risk_points":    {"options": [8.0, 10.0, 12.0, 14.0, 16.0, 18.0],  "default": [10.0, 14.0, 18.0],  "label": "RBR Tokyo Risk"},
            "rbr_ld_fixed_risk_points":    {"options": [4.0, 6.0, 8.0, 10.0, 12.0, 14.0],    "default": [6.0, 10.0, 14.0],   "label": "RBR London Risk"},
        }
    def run(self, df: pd.DataFrame, params: dict) -> dict:
        from strat_mym_combined_all_1m import run_backtest
        return run_backtest(df, {**self.frozen_params(), **params})


# ══════════════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════════════

STRATEGY_REGISTRY = {
    # Combined
    "combined_all_1m":     Combined_All_1m_Strategy(),
    # 1-Minute strategies
    "orb_ny_1m":           ORB_NY_1m_Strategy(),
    "orb_tokyo_1m":        ORB_Tokyo_1m_Strategy(),
    "orb_london_1m":       ORB_London_1m_Strategy(),
    "rbr_ny_1m":           RBR_NY_1m_Strategy(),
    "rbr_tokyo_1m":        RBR_Tokyo_1m_Strategy(),
    "rbr_london_1m":       RBR_London_1m_Strategy(),
    # MYM Combined
    "mym_combined_all_1m":     MYM_Combined_All_1m_Strategy(),
    # MYM 1-Minute strategies
    "mym_orb_ny_1m":           MYM_ORB_NY_1m_Strategy(),
    "mym_orb_tokyo_1m":        MYM_ORB_Tokyo_1m_Strategy(),
    "mym_orb_london_1m":       MYM_ORB_London_1m_Strategy(),
    "mym_rbr_ny_1m":           MYM_RBR_NY_1m_Strategy(),
    "mym_rbr_tokyo_1m":        MYM_RBR_Tokyo_1m_Strategy(),
    "mym_rbr_london_1m":       MYM_RBR_London_1m_Strategy(),
    # 1-Hour strategies
    "precision_sniper_1h": PrecisionSniper_1H_Strategy(),
    "orb_1h":              ORB_1H_Strategy(),
    "tokyo_orb_1h":        TokyoORB_1H_Strategy(),
    "london_orb_1h":       LondonORB_1H_Strategy(),
    "ny_rbr_1h":           NY_RBR_1H_Strategy(),
    "tokyo_rbr_1h":        Tokyo_RBR_1H_Strategy(),
    "london_rbr_1h":       London_RBR_1H_Strategy(),
    # Other strategies
    "bw_atr_optimized":    ButterworthATR_Optimized_Strategy(),
    "butterworth_atr":     ButterworthATRStrategy(),
    "butterworth_atr_1h":  ButterworthATR_1H_Strategy(),
    "precision_sniper":    PrecisionSniperStrategy(),
    "ema_crossover":       EMACrossoverStrategy(),
    "rsi_reversion":       RSIMeanReversionStrategy(),
    "macd_supertrend":     MACDSupertrendStrategy(),
}

def get_strategy(key: str) -> BaseStrategy:
    return STRATEGY_REGISTRY[key]

def list_strategies() -> list:
    return [(k, s.name, s.description) for k, s in STRATEGY_REGISTRY.items()]
