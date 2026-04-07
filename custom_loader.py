"""
Custom Strategy Loader — Dynamically loads user-provided Python strategy code.
The user writes a strategy following a simple template, and this module
compiles it into a BaseStrategy-compatible object for the walk-forward engine.
"""

from __future__ import annotations
import traceback
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from strategy import compute_stats, ema, rsi, macd, atr, supertrend, dmi, vwap
from strategy_registry import BaseStrategy


# ──────────────────────────────────────────────────────────────────────────────
# Template shown to the user
# ──────────────────────────────────────────────────────────────────────────────

STRATEGY_TEMPLATE = '''# ═══════════════════════════════════════════════════════════════
# Custom Strategy Template
# ═══════════════════════════════════════════════════════════════
# Fill in the 3 sections below. The walk-forward engine will
# call run_backtest() with your parameter combos automatically.
#
# Available indicators (already imported):
#   ema(series, length)          — Exponential Moving Average
#   rsi(series, length)          — Relative Strength Index
#   macd(series, fast, slow, sig) — returns (macd_line, signal, histogram)
#   atr(high, low, close, length) — Average True Range
#   supertrend(high, low, close, factor, atr_len) — returns (line, direction)
#   dmi(high, low, close, length) — returns (di_plus, di_minus, adx)
#   vwap(high, low, close, volume) — Volume Weighted Avg Price
#
# Available libraries: numpy (np), pandas (pd)
# ═══════════════════════════════════════════════════════════════

# ── SECTION 1: Strategy info ─────────────────────────────────
STRATEGY_NAME = "My Custom Strategy"
STRATEGY_DESCRIPTION = "Describe what your strategy does here."

# ── SECTION 2: Parameter grid ────────────────────────────────
# These are the values the optimizer will search through.
# Each key becomes a parameter passed to run_backtest().
PARAM_GRID = {
    "ema_fast": [8, 10, 13],
    "ema_slow": [21, 26, 50],
    "atr_len":  [14, 20],
    "sl_mult":  [1.5, 2.0],
    "tp_mult":  [2.0, 3.0],
}

# These stay fixed (not optimized).
FROZEN_PARAMS = {
    "initial_capital": 50000.0,
    "contracts": 2,
    "point_value": 2.0,
    "fee_pct": 0.0015,   # 0.10% exchange + 0.05% slippage
}

# ── SECTION 3: Backtest logic ────────────────────────────────
def run_backtest(df, params):
    """
    Args:
        df:     DataFrame with columns: open, high, low, close, volume
                Index is DatetimeIndex in US/Eastern.
        params: dict with all PARAM_GRID keys + FROZEN_PARAMS merged.

    Must return:
        dict with keys: "trades" (list of dicts), "equity" (pd.Series)

    Each trade dict must have at minimum:
        entry_time, exit_time, entry_price, exit_price,
        direction (1=long, -1=short), contracts, pnl,
        exit_reason, entry_type, tp1_hit, tp2_hit
    """
    # Unpack params
    fast_len  = int(params["ema_fast"])
    slow_len  = int(params["ema_slow"])
    atr_len   = int(params["atr_len"])
    sl_mult   = float(params["sl_mult"])
    tp_mult   = float(params["tp_mult"])
    capital   = float(params["initial_capital"])
    contracts = int(params["contracts"])
    pt_val    = float(params["point_value"])
    fee       = float(params["fee_pct"])

    # Compute indicators
    ef = ema(df["close"], fast_len)
    es = ema(df["close"], slow_len)
    atr_val = atr(df["high"], df["low"], df["close"], atr_len)

    warmup = max(slow_len, atr_len, 50)
    equity = capital
    trades = []
    equity_curve = []
    dates = []

    pos = 0            # +n long, -n short, 0 flat
    entry_px = 0.0
    sl_px = 0.0
    tp_px = 0.0
    direction = 0
    entry_time = None

    for i in range(len(df)):
        bar = df.index[i]
        c = df["close"].iloc[i]
        h = df["high"].iloc[i]
        l = df["low"].iloc[i]

        # Session filter: 9:30 AM - 4:00 PM ET
        bar_mins = bar.hour * 60 + bar.minute
        in_session = 570 <= bar_mins < 960

        # Force close at session end
        if bar_mins >= 960 and pos != 0:
            cost = abs(c * abs(pos) * pt_val) * fee
            pnl = (c - entry_px) * pos * pt_val - cost
            equity += pnl
            trades.append({
                "entry_time": entry_time, "exit_time": bar,
                "entry_price": entry_px, "exit_price": c,
                "direction": direction, "contracts": abs(pos),
                "pnl": pnl, "exit_reason": "Session Close",
                "entry_type": "Custom", "tp1_hit": False, "tp2_hit": False,
            })
            pos = 0; direction = 0

        if i < warmup:
            equity_curve.append(equity)
            dates.append(bar)
            continue

        cur_atr = float(atr_val.iloc[i]) if not np.isnan(atr_val.iloc[i]) else 20.0

        # ── EXIT LOGIC ──
        if pos > 0:
            if h >= tp_px:
                cost = tp_px * pos * pt_val * fee
                pnl = (tp_px - entry_px) * pos * pt_val - cost
                equity += pnl
                trades.append({"entry_time": entry_time, "exit_time": bar,
                    "entry_price": entry_px, "exit_price": tp_px,
                    "direction": 1, "contracts": pos, "pnl": pnl,
                    "exit_reason": "TP", "entry_type": "Custom",
                    "tp1_hit": True, "tp2_hit": False})
                pos = 0; direction = 0
            elif l <= sl_px:
                cost = sl_px * pos * pt_val * fee
                pnl = (sl_px - entry_px) * pos * pt_val - cost
                equity += pnl
                trades.append({"entry_time": entry_time, "exit_time": bar,
                    "entry_price": entry_px, "exit_price": sl_px,
                    "direction": 1, "contracts": pos, "pnl": pnl,
                    "exit_reason": "SL", "entry_type": "Custom",
                    "tp1_hit": False, "tp2_hit": False})
                pos = 0; direction = 0
        elif pos < 0:
            if l <= tp_px:
                cost = tp_px * abs(pos) * pt_val * fee
                pnl = (entry_px - tp_px) * abs(pos) * pt_val - cost
                equity += pnl
                trades.append({"entry_time": entry_time, "exit_time": bar,
                    "entry_price": entry_px, "exit_price": tp_px,
                    "direction": -1, "contracts": abs(pos), "pnl": pnl,
                    "exit_reason": "TP", "entry_type": "Custom",
                    "tp1_hit": True, "tp2_hit": False})
                pos = 0; direction = 0
            elif h >= sl_px:
                cost = sl_px * abs(pos) * pt_val * fee
                pnl = (entry_px - sl_px) * abs(pos) * pt_val - cost
                equity += pnl
                trades.append({"entry_time": entry_time, "exit_time": bar,
                    "entry_price": entry_px, "exit_price": sl_px,
                    "direction": -1, "contracts": abs(pos), "pnl": pnl,
                    "exit_reason": "SL", "entry_type": "Custom",
                    "tp1_hit": False, "tp2_hit": False})
                pos = 0; direction = 0

        # ── ENTRY LOGIC (edit this!) ──
        if pos == 0 and in_session and i >= 1:
            ef_now  = float(ef.iloc[i])
            es_now  = float(es.iloc[i])
            ef_prev = float(ef.iloc[i-1])
            es_prev = float(es.iloc[i-1])

            # Long: fast EMA crosses above slow EMA
            if ef_prev <= es_prev and ef_now > es_now:
                entry_px = c
                sl_px = c - cur_atr * sl_mult
                tp_px = c + cur_atr * tp_mult
                pos = contracts; direction = 1; entry_time = bar
                equity -= c * contracts * pt_val * fee

            # Short: fast EMA crosses below slow EMA
            elif ef_prev >= es_prev and ef_now < es_now:
                entry_px = c
                sl_px = c + cur_atr * sl_mult
                tp_px = c - cur_atr * tp_mult
                pos = -contracts; direction = -1; entry_time = bar
                equity -= c * contracts * pt_val * fee

        equity_curve.append(equity)
        dates.append(bar)

    # Close any open position at end
    if pos != 0:
        c = df["close"].iloc[-1]
        cost = c * abs(pos) * pt_val * fee
        pnl = (c - entry_px) * pos * pt_val - cost
        equity += pnl
        trades.append({"entry_time": entry_time, "exit_time": df.index[-1],
            "entry_price": entry_px, "exit_price": c,
            "direction": direction, "contracts": abs(pos), "pnl": pnl,
            "exit_reason": "End of Data", "entry_type": "Custom",
            "tp1_hit": False, "tp2_hit": False})
        equity_curve[-1] = equity

    return {"trades": trades, "equity": pd.Series(equity_curve, index=dates)}
'''


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic strategy wrapper
# ──────────────────────────────────────────────────────────────────────────────

class CustomStrategy(BaseStrategy):
    """Wraps dynamically-loaded user code into the BaseStrategy interface."""

    def __init__(self, user_namespace: dict):
        self._ns = user_namespace

    @property
    def name(self) -> str:
        return self._ns.get("STRATEGY_NAME", "Custom Strategy")

    @property
    def description(self) -> str:
        return self._ns.get("STRATEGY_DESCRIPTION", "User-uploaded strategy.")

    def default_grid(self) -> Dict[str, List]:
        return self._ns.get("PARAM_GRID", {})

    def frozen_params(self) -> Dict[str, Any]:
        return self._ns.get("FROZEN_PARAMS", {})

    def sidebar_grid_options(self) -> Dict[str, Dict]:
        grid = self.default_grid()
        return {
            k: {"options": v, "default": v, "label": k.replace("_", " ").title()}
            for k, v in grid.items()
        }

    def run(self, df: pd.DataFrame, params: dict) -> dict:
        fn = self._ns.get("run_backtest")
        if fn is None:
            raise ValueError("Custom code must define a run_backtest(df, params) function.")
        result = fn(df, params)
        # Compute stats if not provided
        if "stats" not in result:
            result["stats"] = compute_stats(
                result["trades"], result["equity"],
                params.get("initial_capital", 50000.0)
            )
        if "params" not in result:
            result["params"] = params
        return result


def compile_custom_strategy(code: str) -> tuple:
    """
    Compile user code and return (CustomStrategy, error_string).
    On success: (strategy, None)
    On failure: (None, error_message)
    """
    # Build a namespace with all the indicators pre-imported
    namespace = {
        "np": np,
        "pd": pd,
        "ema": ema,
        "rsi": rsi,
        "macd": macd,
        "atr": atr,
        "supertrend": supertrend,
        "dmi": dmi,
        "vwap": vwap,
        "compute_stats": compute_stats,
    }

    try:
        exec(compile(code, "<custom_strategy>", "exec"), namespace)
    except Exception:
        return None, traceback.format_exc()

    # Validate required pieces
    if "run_backtest" not in namespace:
        return None, "Error: Your code must define a `run_backtest(df, params)` function."
    if "PARAM_GRID" not in namespace:
        return None, "Error: Your code must define a `PARAM_GRID` dict."

    strat = CustomStrategy(namespace)
    return strat, None


def save_strategy_to_disk(code: str, filename: str) -> str:
    """Save custom strategy code to the strategies folder."""
    import os
    save_dir = os.path.join(os.path.dirname(__file__), "saved_strategies")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    with open(path, "w") as f:
        f.write(code)
    return path


def list_saved_strategies() -> list:
    """List all .py files in the saved_strategies folder."""
    import os
    save_dir = os.path.join(os.path.dirname(__file__), "saved_strategies")
    if not os.path.exists(save_dir):
        return []
    files = [f for f in os.listdir(save_dir) if f.endswith(".py")]
    files.sort()
    return files


def load_saved_strategy(filename: str) -> str:
    """Load code from saved_strategies folder."""
    import os
    path = os.path.join(os.path.dirname(__file__), "saved_strategies", filename)
    with open(path, "r") as f:
        return f.read()
