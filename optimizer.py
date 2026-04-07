"""
Strategy Optimizer — Uses backtrader's broker for accurate results.
Exhaustive grid search with multiprocessing.
"""

from __future__ import annotations
import itertools
import multiprocessing as mp
from functools import partial
from typing import Optional, Callable
import numpy as np
import pandas as pd


# Module-level for multiprocessing pickling
_opt_strategy_key = None
_opt_use_native = False

def _init_opt_worker(strategy_key, use_native=False):
    global _opt_strategy_key, _opt_use_native
    _opt_strategy_key = strategy_key
    _opt_use_native = use_native


def _eval_single_bt(combo: dict, df: pd.DataFrame, extra_params: Optional[dict] = None) -> dict:
    """Evaluate one parameter combo. Uses backtrader or native engine based on _opt_use_native."""
    try:
        if _opt_use_native:
            from bt_bw_atr_strategy import run_bt_generic_fast
            merged_combo = {**(extra_params or {}), **combo}
            result = run_bt_generic_fast(df, _opt_strategy_key, merged_combo)
        else:
            from bt_bw_atr_strategy import run_bt_generic
            merged_combo = {**(extra_params or {}), **combo}
            result = run_bt_generic(df, _opt_strategy_key, merged_combo)
        stats = result.get("stats", {})

        sharpe = max(stats.get("sharpe", 0.0), 0.0)
        pf     = min(stats.get("profit_factor", 0.0), 10.0)
        mdd    = abs(stats.get("max_drawdown", 0.0))
        n      = stats.get("total_trades", 0)

        if n < 3:
            score = -999.0
        else:
            score = sharpe * pf * max(0.0, 1.0 - mdd)

        return {
            "params":         combo,
            "score":          score,
            "total_pnl":      stats.get("total_pnl", 0.0),
            "win_rate":       stats.get("win_rate", 0.0),
            "profit_factor":  stats.get("profit_factor", 0.0),
            "sharpe":         stats.get("sharpe", 0.0),
            "max_drawdown":   stats.get("max_drawdown", 0.0),
            "total_trades":   stats.get("total_trades", 0),
            "expectancy":     stats.get("expectancy", 0.0) if "expectancy" in stats else 0.0,
            "net_return_pct": stats.get("net_return_pct", 0.0),
            "final_value":    stats.get("final_value", 0.0),
        }
    except Exception as e:
        return {
            "params": combo, "score": -999.0,
            "total_pnl": 0, "win_rate": 0, "profit_factor": 0,
            "sharpe": 0, "max_drawdown": 0, "total_trades": 0,
            "expectancy": 0, "net_return_pct": 0, "final_value": 0,
            "error": str(e),
        }


def run_optimization(
    df: pd.DataFrame,
    strategy_key: str,
    param_grid: dict,
    n_workers: Optional[int] = None,
    progress_cb: Optional[Callable] = None,
    top_n: int = 20,
    extra_params: Optional[dict] = None,
    rank_by: str = "score",
    use_native: bool = False,
) -> dict:
    """
    Exhaustive grid search.

    use_native: If False (default), uses backtrader broker for accurate results.
                If True, uses the native engine (faster, same as walk-forward).
    extra_params: fixed params (capital, risk limits, etc.) merged into every combo.
    rank_by:      metric key to sort results by.
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    # Build combos
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    # Filter out invalid TP ordering — runner/TP2 must always be >= TP1.
    # Covers all naming patterns used across strategies:
    #   tp1_rr / tp2_rr                         (Precision Sniper, BW-ATR)
    #   first_tp_points / second_tp_points       (ORB strategies)
    #   first_rr_ratio / runner_rr_ratio         (RBR strategies)
    #   prefixed: orb_ny_first_tp_points, rbr_ny_first_rr_ratio, etc. (Combined)
    def _valid_tp_order(c):
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

    pre_filter = len(combos)
    combos = [c for c in combos if _valid_tp_order(c)]
    if len(combos) < pre_filter and progress_cb:
        progress_cb(0, len(combos))  # signal updated total

    total = len(combos)

    if total == 0:
        return {"results": [], "best": None, "top_n": [], "total": 0}

    chunk_size = max(1, total // (n_workers * 2))
    worker_fn = partial(_eval_single_bt, df=df, extra_params=extra_params)

    all_results = []

    with mp.Pool(processes=n_workers, initializer=_init_opt_worker, initargs=(strategy_key, use_native)) as pool:
        completed = 0
        for result in pool.imap_unordered(worker_fn, combos, chunksize=chunk_size):
            all_results.append(result)
            completed += 1
            if progress_cb and completed % max(1, total // 50) == 0:
                progress_cb(completed, total)

    # Sort by chosen metric
    if rank_by == "max_drawdown":
        # For drawdown, closer to 0 is better (values are negative, so sort ascending by abs)
        all_results.sort(key=lambda x: abs(x.get("max_drawdown", -999)))
    else:
        all_results.sort(key=lambda x: x.get(rank_by, -999), reverse=True)

    return {
        "results":  all_results,
        "best":     all_results[0] if all_results else None,
        "top_n":    all_results[:top_n],
        "total":    total,
    }
