"""
Walk-Forward Analysis Engine
Train: 12-month rolling window  |  Test: 3-month blind window
Only out-of-sample equity is stitched together.

v2: Parallelised with multiprocessing for 8-10x speedup.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from typing import Optional, Callable
from functools import partial
import numpy as np
import pandas as pd

from strategy import StrategyParams, run_backtest


# ──────────────────────────────────────────────────────────────────────────────
# Parameter grid (optimizable)
# ──────────────────────────────────────────────────────────────────────────────

PARAM_GRID = {
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

# Frozen params (not optimised, kept as strategy defaults)
FROZEN_PARAMS = {
    "tp3_qty":          0,
    "use_supertrend":   True,
    "use_trail":        True,
    "trail_after_tp":   "TP2",
    "tp1_qty":          1,
    "tp2_qty":          1,
    "use_force_close":  True,
    "use_max_trade_loss": True,
    "use_daily_max_loss": True,
    "max_trade_loss":   200.0,
    "daily_max_loss":   650.0,
    "exchange_fee_pct": 0.0010,
    "slippage_pct":     0.0005,
    "point_value":      2.0,
    "initial_capital":  50000.0,
    "allow_longs":      True,
    "allow_shorts":     True,
    "use_session":      True,
    "use_pullback":     True,
    "pullback_score":   4,
    "use_atr_risk":     False,
}


def _valid_tp_order(c: dict) -> bool:
    """Reject combos where runner/TP2 < TP1 across all naming patterns."""
    import re
    # Pattern 1: tp1_rr / tp2_rr (with optional prefix)
    tp_vals = {}
    for k, v in c.items():
        m = re.match(r"^(.*)tp(\d)_rr$", k)
        if m:
            tp_vals.setdefault(m.group(1), {})[int(m.group(2))] = v
    for tps in tp_vals.values():
        if 1 in tps and 2 in tps and tps[2] < tps[1]:
            return False
        if 2 in tps and 3 in tps and tps[3] < tps[2]:
            return False

    # Pattern 2: first_tp_points / second_tp_points (with optional prefix)
    tp_pts = {}
    for k, v in c.items():
        m = re.match(r"^(.*)first_tp_points$", k)
        if m:
            tp_pts.setdefault(m.group(1), {})["first"] = v
        m = re.match(r"^(.*)second_tp_points$", k)
        if m:
            tp_pts.setdefault(m.group(1), {})["second"] = v
    for tps in tp_pts.values():
        if "first" in tps and "second" in tps and tps["second"] < tps["first"]:
            return False

    # Pattern 3: first_rr_ratio / runner_rr_ratio (with optional prefix)
    rr_pairs = {}
    for k, v in c.items():
        m = re.match(r"^(.*)first_rr_ratio$", k)
        if m:
            rr_pairs.setdefault(m.group(1), {})["first"] = v
        m = re.match(r"^(.*)runner_rr_ratio$", k)
        if m:
            rr_pairs.setdefault(m.group(1), {})["runner"] = v
    for tps in rr_pairs.values():
        if "first" in tps and "runner" in tps and tps["runner"] < tps["first"]:
            return False

    return True


def build_param_combinations(grid: dict, frozen: Optional[dict] = None) -> list:
    if frozen is None:
        frozen = FROZEN_PARAMS
    keys   = list(grid.keys())
    values = list(grid.values())
    combos = []
    for combo in itertools.product(*values):
        d = dict(zip(keys, combo))
        d.update(frozen)
        if _valid_tp_order(d):
            combos.append(d)
    return combos


# ──────────────────────────────────────────────────────────────────────────────
# Optimization objective
# ──────────────────────────────────────────────────────────────────────────────

def objective_score(stats: dict) -> float:
    """
    Composite score = Sharpe * profit_factor * (1 - |max_drawdown|)
    Rewards risk-adjusted returns; penalises drawdown heavily.
    """
    sharpe = max(stats.get("sharpe", 0.0), 0.0)
    pf     = min(stats.get("profit_factor", 0.0), 10.0)
    mdd    = abs(stats.get("max_drawdown", 0.0))
    n      = stats.get("total_trades", 0)
    if n < 5:
        return -999.0
    return sharpe * pf * max(0.0, 1.0 - mdd)


# ──────────────────────────────────────────────────────────────────────────────
# Worker function for multiprocessing (must be top-level for pickling)
# ──────────────────────────────────────────────────────────────────────────────

# Module-level variable set before pool.map — tells workers which strategy to use
_worker_strategy_key = None

def _init_worker(strategy_key):
    """Called once per worker process to set the strategy key."""
    global _worker_strategy_key
    _worker_strategy_key = strategy_key

def _eval_combo(combo: dict, train_df: pd.DataFrame) -> tuple:
    """Evaluate a single parameter combo. Returns (score, combo, stats, equity)."""
    try:
        if _worker_strategy_key is not None:
            from strategy_registry import get_strategy
            strat = get_strategy(_worker_strategy_key)
            result = strat.run(train_df, combo)
        else:
            # Fallback: original Precision Sniper
            params = StrategyParams(**combo)
            result = run_backtest(train_df, params)
        score = objective_score(result["stats"])
        return (score, combo, result["stats"], result["equity"])
    except Exception:
        return (-999.0, combo, {}, pd.Series(dtype=float))


# ──────────────────────────────────────────────────────────────────────────────
# Walk-forward engine (parallelised)
# ──────────────────────────────────────────────────────────────────────────────

def run_walk_forward(
    df: pd.DataFrame,
    train_months: int = 12,
    test_months:  int = 3,
    param_grid:   Optional[dict] = None,
    progress_cb:  Optional[Callable] = None,
    n_workers:    Optional[int] = None,
    strategy_key: Optional[str] = None,
) -> dict:
    """
    Returns:
        folds        : list of fold dicts
        oos_equity   : pd.Series — stitched out-of-sample equity
        oos_trades   : list of all OOS trades
        is_equity    : pd.Series — best in-sample equity per fold (for comparison)
        windows      : list of (train_start, train_end, test_start, test_end)
    """
    if param_grid is None:
        param_grid = PARAM_GRID

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)  # leave 1 core for UI

    # Get frozen params from strategy if specified, otherwise use defaults
    if strategy_key is not None:
        from strategy_registry import get_strategy
        frozen = get_strategy(strategy_key).frozen_params()
    else:
        frozen = FROZEN_PARAMS
    combos = build_param_combinations(param_grid, frozen)

    # Build fold windows
    df_start = df.index[0]
    df_end   = df.index[-1]

    windows = []
    train_start = df_start
    while True:
        train_end  = train_start + relativedelta(months=train_months)
        test_start = train_end
        test_end   = test_start + relativedelta(months=test_months)
        if test_end > df_end:
            test_end = df_end
        if test_start >= df_end:
            break
        windows.append((train_start, train_end, test_start, test_end))
        train_start = train_start + relativedelta(months=test_months)  # rolling step

    total_folds = len(windows)
    folds       = []
    oos_equity_pieces = []
    is_equity_pieces  = []
    oos_trades        = []

    prev_oos_equity = 50000.0   # base capital for stitching

    for fold_i, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        # Slice data
        train_df = df[(df.index >= tr_s) & (df.index < tr_e)].copy()
        test_df  = df[(df.index >= te_s) & (df.index < te_e)].copy()

        if len(train_df) < 200 or len(test_df) < 10:
            continue

        if progress_cb:
            pct = fold_i / total_folds
            progress_cb(fold_i, total_folds, f"training ({len(combos)} combos, {n_workers} cores)", pct)

        # ── Parallelised optimisation on training window ────────────────────
        worker_fn = partial(_eval_combo, train_df=train_df)

        best_score  = -np.inf
        best_params = None
        best_is_stats = {}
        best_is_equity = pd.Series(dtype=float)

        # Use chunks + imap_unordered so we can update progress
        chunk_size = max(1, len(combos) // (n_workers * 4))

        # Tell worker processes which strategy to use
        global _worker_strategy_key
        _worker_strategy_key = strategy_key

        with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(strategy_key,)) as pool:
            completed = 0
            for score, combo, stats, eq in pool.imap_unordered(worker_fn, combos, chunksize=chunk_size):
                completed += 1
                if score > best_score:
                    best_score     = score
                    best_params    = combo
                    best_is_stats  = stats
                    best_is_equity = eq

                # Update progress every ~5% of combos
                if progress_cb and completed % max(1, len(combos) // 20) == 0:
                    pct = (fold_i + completed / len(combos)) / total_folds
                    progress_cb(fold_i, total_folds, f"fold {fold_i+1} — {completed}/{len(combos)}", min(pct, 0.99))

        if best_params is None:
            continue

        # ── Test on blind window ─────────────────────────────────────────────
        if progress_cb:
            pct = (fold_i + 0.95) / total_folds
            progress_cb(fold_i, total_folds, f"fold {fold_i+1} — blind test", min(pct, 0.99))

        try:
            if strategy_key is not None:
                from strategy_registry import get_strategy
                strat = get_strategy(strategy_key)
                oos_result = strat.run(test_df, best_params)
            else:
                test_params = StrategyParams(**best_params)
                oos_result  = run_backtest(test_df, test_params)
        except Exception:
            continue

        # Stitch OOS equity: normalise to continue from previous OOS equity
        oos_eq = oos_result["equity"].copy()
        if len(oos_eq) == 0:
            continue

        shift = prev_oos_equity - oos_eq.iloc[0]
        oos_eq = oos_eq + shift
        prev_oos_equity = float(oos_eq.iloc[-1])

        # IS equity (just for visual comparison; not stitched)
        is_eq = best_is_equity.copy() if len(best_is_equity) > 0 else pd.Series(dtype=float)

        oos_equity_pieces.append(oos_eq)
        is_equity_pieces.append(is_eq)

        for t in oos_result["trades"]:
            oos_trades.append(t)

        folds.append({
            "fold":          fold_i + 1,
            "train_start":   tr_s,
            "train_end":     tr_e,
            "test_start":    te_s,
            "test_end":      te_e,
            "best_params":   best_params,
            "is_score":      best_score,
            "is_stats":      best_is_stats,
            "oos_stats":     oos_result["stats"],
            "is_equity":     is_eq,
            "oos_equity":    oos_eq,
        })

        if progress_cb:
            progress_cb(fold_i + 1, total_folds, "done", (fold_i + 1) / total_folds)

    # ── Stitch equity curves ─────────────────────────────────────────────────
    if oos_equity_pieces:
        oos_equity = pd.concat(oos_equity_pieces).sort_index()
        oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")]
    else:
        oos_equity = pd.Series(dtype=float)

    if is_equity_pieces:
        is_equity = pd.concat(is_equity_pieces).sort_index()
        is_equity = is_equity[~is_equity.index.duplicated(keep="last")]
    else:
        is_equity = pd.Series(dtype=float)

    from strategy import compute_stats
    oos_stats = compute_stats(oos_trades, oos_equity, 50000.0)

    return {
        "folds":       folds,
        "oos_equity":  oos_equity,
        "oos_trades":  oos_trades,
        "oos_stats":   oos_stats,
        "is_equity":   is_equity,
        "windows":     windows,
        "total_folds": total_folds,
    }
