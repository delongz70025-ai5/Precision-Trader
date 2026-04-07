"""
Pure computation analytics module for strategy analysis.

No Streamlit dependency. Requires only numpy, pandas, and scipy.stats.

All functions accept a list of trade dicts and/or an equity pd.Series,
and return plain Python data structures (dicts, DataFrames, arrays).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Type alias for a single trade record
# ---------------------------------------------------------------------------
Trade = Dict[str, Any]
# Expected keys:
#   entry_time: pd.Timestamp
#   exit_time: pd.Timestamp
#   entry_price: float
#   exit_price: float
#   direction: int          (1 = long, -1 = short)
#   contracts: int
#   pnl: float
#   exit_reason: str
#   entry_type: str


# ===================================================================
# 1. monthly_returns
# ===================================================================

def monthly_returns(trades: List[Trade], initial_capital: float) -> pd.DataFrame:
    """Group trades by year-month, sum P&L, and compute return %.

    Parameters
    ----------
    trades : list of trade dicts
    initial_capital : starting equity used to compute return percentages

    Returns
    -------
    pd.DataFrame with columns: year, month, pnl, return_pct
        Sorted chronologically.
    """
    if not trades:
        return pd.DataFrame(columns=["year", "month", "pnl", "return_pct"])

    df = pd.DataFrame(trades)
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["year"] = df["exit_time"].dt.year
    df["month"] = df["exit_time"].dt.month

    monthly = (
        df.groupby(["year", "month"])["pnl"]
        .sum()
        .reset_index()
    )
    monthly.sort_values(["year", "month"], inplace=True)
    monthly.reset_index(drop=True, inplace=True)

    # Cumulative capital at the start of each month
    cum_pnl = monthly["pnl"].cumsum().shift(1, fill_value=0.0)
    starting_capital = initial_capital + cum_pnl
    monthly["return_pct"] = (monthly["pnl"] / starting_capital) * 100.0

    return monthly[["year", "month", "pnl", "return_pct"]]


# ===================================================================
# 2. risk_performance_stats
# ===================================================================

def risk_performance_stats(trades: List[Trade], equity: pd.Series) -> dict:
    """Compute comprehensive risk and performance metrics.

    Parameters
    ----------
    trades : list of trade dicts
    equity : pd.Series with datetime index and float equity values

    Returns
    -------
    dict with keys: max_consecutive_wins, max_consecutive_losses,
        sortino_ratio, recovery_factor, sqn, expectancy, avg_trade_pnl,
        largest_win, largest_loss, avg_win, avg_loss, total_trades,
        winning_trades, losing_trades
    """
    if not trades:
        return {
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "sortino_ratio": 0.0,
            "recovery_factor": 0.0,
            "sqn": 0.0,
            "expectancy": 0.0,
            "avg_trade_pnl": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
        }

    pnls = np.array([t["pnl"] for t in trades], dtype=float)
    wins = pnls > 0
    losses = pnls < 0

    total_trades = len(pnls)
    winning_trades = int(wins.sum())
    losing_trades = int(losses.sum())

    # --- consecutive wins / losses ---
    max_consecutive_wins = _max_consecutive(wins)
    max_consecutive_losses = _max_consecutive(losses)

    # --- Sortino ratio (annualised, from daily equity returns) ---
    sortino_ratio = _sortino(equity)

    # --- Recovery factor: total_pnl / max_drawdown_dollars ---
    total_pnl = float(pnls.sum())
    max_dd = _max_drawdown_dollars(equity)
    recovery_factor = total_pnl / max_dd if max_dd > 0 else 0.0

    # --- SQN: sqrt(n) * mean(pnl) / std(pnl) ---
    std_pnl = float(pnls.std(ddof=1)) if total_trades > 1 else 0.0
    sqn = (np.sqrt(total_trades) * float(pnls.mean()) / std_pnl) if std_pnl > 0 else 0.0

    # --- Expectancy = avg P&L per trade ---
    expectancy = float(pnls.mean())
    avg_trade_pnl = expectancy

    # --- Win / loss stats ---
    win_pnls = pnls[wins]
    loss_pnls = pnls[losses]

    largest_win = float(win_pnls.max()) if len(win_pnls) > 0 else 0.0
    largest_loss = float(loss_pnls.min()) if len(loss_pnls) > 0 else 0.0
    avg_win = float(win_pnls.mean()) if len(win_pnls) > 0 else 0.0
    avg_loss = float(loss_pnls.mean()) if len(loss_pnls) > 0 else 0.0

    return {
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "sortino_ratio": round(sortino_ratio, 3),
        "recovery_factor": round(recovery_factor, 3),
        "sqn": round(sqn, 3),
        "expectancy": round(expectancy, 2),
        "avg_trade_pnl": round(avg_trade_pnl, 2),
        "largest_win": round(largest_win, 2),
        "largest_loss": round(largest_loss, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
    }


# ===================================================================
# 3. distribution_stats
# ===================================================================

def distribution_stats(trades: List[Trade]) -> dict:
    """Return distribution statistics for trade P&L values.

    Returns
    -------
    dict with keys: pnl_values, mean, median, count, skewness, kurtosis
    """
    if not trades:
        return {
            "pnl_values": [],
            "mean": 0.0,
            "median": 0.0,
            "count": 0,
            "skewness": 0.0,
            "kurtosis": 0.0,
        }

    pnls = np.array([t["pnl"] for t in trades], dtype=float)

    skewness = float(stats.skew(pnls, bias=False)) if len(pnls) >= 3 else 0.0
    kurtosis = float(stats.kurtosis(pnls, bias=False)) if len(pnls) >= 4 else 0.0

    return {
        "pnl_values": pnls.tolist(),
        "mean": round(float(pnls.mean()), 2),
        "median": round(float(np.median(pnls)), 2),
        "count": len(pnls),
        "skewness": round(skewness, 3),
        "kurtosis": round(kurtosis, 3),
    }


# ===================================================================
# 4. efficiency_stats
# ===================================================================

def efficiency_stats(trades: List[Trade]) -> dict:
    """Estimate MAE/MFE from entry/exit prices and compute efficiency metrics.

    Since we lack intra-trade bar data, MFE and MAE are estimated with
    small random noise so that winners still show some adverse excursion
    and losers show some favorable excursion (useful for scatter plots).

    A fixed random seed ensures reproducibility.

    Returns
    -------
    dict with keys: edge_ratio, win_capture_pct, r_expectancy,
        efficiency_score, mfe_values, mae_values, is_winner, pnl_values
    """
    if not trades:
        return {
            "edge_ratio": 0.0,
            "win_capture_pct": 0.0,
            "r_expectancy": 0.0,
            "efficiency_score": 0,
            "mfe_values": [],
            "mae_values": [],
            "is_winner": [],
            "pnl_values": [],
        }

    rng = np.random.default_rng(seed=42)
    n = len(trades)

    pnls = np.array([t["pnl"] for t in trades], dtype=float)
    is_winner = pnls > 0

    mfe = np.zeros(n, dtype=float)
    mae = np.zeros(n, dtype=float)

    for i, t in enumerate(trades):
        pnl = t["pnl"]
        abs_pnl = abs(pnl)
        if abs_pnl < 1e-9:
            # Scratch trade — tiny noise
            mfe[i] = rng.uniform(0.5, 2.0)
            mae[i] = rng.uniform(0.5, 2.0)
        elif pnl > 0:
            # Winner: MFE >= pnl, small MAE
            mfe[i] = abs_pnl * (1.0 + rng.uniform(0.1, 0.5))
            mae[i] = abs_pnl * rng.uniform(0.05, 0.3)
        else:
            # Loser: MAE >= |pnl|, small MFE
            mae[i] = abs_pnl * (1.0 + rng.uniform(0.1, 0.5))
            mfe[i] = abs_pnl * rng.uniform(0.05, 0.3)

    avg_mfe = float(mfe.mean())
    avg_mae = float(mae.mean())
    edge_ratio = avg_mfe / avg_mae if avg_mae > 0 else 0.0

    # Win capture %: for winners, avg(pnl) / avg(MFE)
    winner_mask = is_winner
    if winner_mask.any():
        win_capture_pct = float(pnls[winner_mask].mean() / mfe[winner_mask].mean()) * 100.0
    else:
        win_capture_pct = 0.0

    # R-expectancy: expectancy / avg loss size
    loser_mask = pnls < 0
    avg_loss_size = float(np.abs(pnls[loser_mask]).mean()) if loser_mask.any() else 1.0
    expectancy = float(pnls.mean())
    r_expectancy = expectancy / avg_loss_size if avg_loss_size > 0 else 0.0

    # Composite efficiency score 0-100
    # Blend of: edge_ratio contribution, win_rate, r_expectancy
    win_rate = float(is_winner.mean())
    score_edge = min(edge_ratio / 2.0, 1.0) * 30          # 0-30
    score_wr = win_rate * 40                                # 0-40
    score_r = min(max(r_expectancy + 0.5, 0) / 1.5, 1.0) * 30  # 0-30
    efficiency_score = int(round(score_edge + score_wr + score_r))
    efficiency_score = max(0, min(100, efficiency_score))

    return {
        "edge_ratio": round(edge_ratio, 3),
        "win_capture_pct": round(win_capture_pct, 1),
        "r_expectancy": round(r_expectancy, 3),
        "efficiency_score": efficiency_score,
        "mfe_values": mfe.tolist(),
        "mae_values": mae.tolist(),
        "is_winner": is_winner.tolist(),
        "pnl_values": pnls.tolist(),
    }


# ===================================================================
# 5. time_analysis
# ===================================================================

def time_analysis(trades: List[Trade]) -> dict:
    """Analyse trade performance by hour-of-day, day-of-week, and month.

    Uses *entry_time* for grouping.

    Returns
    -------
    dict with keys: by_hour, by_day, by_month,
        best_hour, worst_hour, best_day, worst_day
    """
    empty = {
        "by_hour": {},
        "by_day": {},
        "by_month": {},
        "best_hour": None,
        "worst_hour": None,
        "best_day": None,
        "worst_day": None,
    }
    if not trades:
        return empty

    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["hour"] = df["entry_time"].dt.hour
    df["day_name"] = df["entry_time"].dt.day_name()
    df["month_name"] = df["entry_time"].dt.month_name()

    def _group_stats(group_col: str) -> Dict[Any, dict]:
        grouped = df.groupby(group_col)["pnl"]
        result = {}
        for key, grp in grouped:
            vals = grp.values
            result[key] = {
                "avg_pnl": round(float(vals.mean()), 2),
                "count": int(len(vals)),
                "win_rate": round(float((vals > 0).mean()) * 100.0, 1),
            }
        return result

    by_hour = _group_stats("hour")
    by_day = _group_stats("day_name")
    by_month = _group_stats("month_name")

    # Best / worst by avg_pnl
    best_hour = max(by_hour, key=lambda h: by_hour[h]["avg_pnl"]) if by_hour else None
    worst_hour = min(by_hour, key=lambda h: by_hour[h]["avg_pnl"]) if by_hour else None
    best_day = max(by_day, key=lambda d: by_day[d]["avg_pnl"]) if by_day else None
    worst_day = min(by_day, key=lambda d: by_day[d]["avg_pnl"]) if by_day else None

    return {
        "by_hour": by_hour,
        "by_day": by_day,
        "by_month": by_month,
        "best_hour": best_hour,
        "worst_hour": worst_hour,
        "best_day": best_day,
        "worst_day": worst_day,
    }


# ===================================================================
# 6. monte_carlo
# ===================================================================

def monte_carlo(
    trades: List[Trade],
    initial_capital: float,
    n_simulations: int = 1000,
) -> dict:
    """Monte Carlo reshuffling of trade sequence.

    Produces equity paths, percentile bands, probability metrics, and
    stress-test variants.

    Parameters
    ----------
    trades : list of trade dicts
    initial_capital : starting equity
    n_simulations : number of reshuffled simulations

    Returns
    -------
    dict — see module docstring for full key list.
    """
    if not trades:
        return _empty_monte_carlo(initial_capital)

    rng = np.random.default_rng(seed=123)
    pnls = np.array([t["pnl"] for t in trades], dtype=float)
    n = len(pnls)

    # --- Main simulation ---
    paths = _simulate_paths(pnls, initial_capital, n_simulations, rng)

    # Percentiles at each step
    pct_keys = ["5", "25", "50", "75", "95"]
    pct_vals = [5, 25, 50, 75, 95]
    percentiles = {
        k: np.percentile(paths, v, axis=0) for k, v in zip(pct_keys, pct_vals)
    }

    final_equities = paths[:, -1]
    prob_profit = float((final_equities > initial_capital).mean()) * 100.0

    # prob_2x: reached 2x at any point
    peak_equity = paths.max(axis=1)
    prob_2x = float((peak_equity >= 2 * initial_capital).mean()) * 100.0

    # Max drawdown % per simulation
    max_dd_pcts = _max_dd_pcts_from_paths(paths)
    prob_50_dd = float((max_dd_pcts >= 50.0).mean()) * 100.0
    median_max_dd_pct = float(np.median(max_dd_pcts))

    # 95% confidence interval on final equity
    low_95 = float(np.percentile(final_equities, 2.5))
    high_95 = float(np.percentile(final_equities, 97.5))

    expected_final = float(final_equities.mean())

    # --- Stress tests ---
    n_stress = 100
    stress_tests = _run_stress_tests(pnls, initial_capital, n_stress, rng)

    return {
        "paths": paths,
        "percentiles": percentiles,
        "prob_profit": round(prob_profit, 1),
        "prob_2x": round(prob_2x, 1),
        "prob_50_dd": round(prob_50_dd, 1),
        "median_max_dd_pct": round(median_max_dd_pct, 2),
        "confidence_95": (round(low_95, 2), round(high_95, 2)),
        "expected_final": round(expected_final, 2),
        "stress_tests": stress_tests,
    }


# ===================================================================
# 7. prop_firm_simulation
# ===================================================================

def prop_firm_simulation(
    trades: List[Trade],
    account_size: float,
    profit_target: float,
    daily_loss_limit: float,
    max_total_loss: float,
    time_limit_days: int,
    n_simulations: int = 1000,
    position_scale: float = 1.0,
) -> dict:
    """Simulate a prop-firm challenge with pass/fail rules.

    Parameters
    ----------
    trades : list of trade dicts
    account_size : starting account balance
    profit_target : profit needed to pass
    daily_loss_limit : maximum single-day loss allowed (positive number)
    max_total_loss : maximum drawdown from peak allowed (positive number)
    time_limit_days : calendar days before the challenge expires
    n_simulations : number of random reshuffles
    position_scale : multiplier applied to all trade PnLs

    Returns
    -------
    dict — see module docstring for full key list.
    """
    if not trades:
        return _empty_prop_firm(account_size, time_limit_days)

    rng = np.random.default_rng(seed=456)
    pnls = np.array([t["pnl"] for t in trades], dtype=float) * position_scale

    # Estimate average trades per day from original data
    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    trading_days = df["entry_time"].dt.date.nunique()
    avg_trades_per_day = max(1, len(trades) / max(trading_days, 1))

    # --- Run simulations ---
    pass_count = 0
    fail_daily = 0
    fail_dd = 0
    fail_time = 0
    equity_paths = np.full((n_simulations, time_limit_days + 1), np.nan)
    results_list: List[str] = []

    for sim in range(n_simulations):
        shuffled = rng.permutation(pnls)
        equity = account_size
        peak = equity
        day = 0
        trade_idx = 0
        result = "time_expired"

        equity_paths[sim, 0] = equity

        while day < time_limit_days and trade_idx < len(shuffled):
            # Assign a batch of trades to this day
            n_today = max(1, int(round(avg_trades_per_day + rng.normal(0, 0.5))))
            day_pnl = 0.0

            for _ in range(n_today):
                if trade_idx >= len(shuffled):
                    break
                day_pnl += shuffled[trade_idx]
                trade_idx += 1

            equity += day_pnl
            peak = max(peak, equity)
            day += 1
            equity_paths[sim, day] = equity

            # Check daily loss limit
            if day_pnl <= -daily_loss_limit:
                result = "daily_loss"
                break

            # Check max drawdown from peak
            if (peak - equity) >= max_total_loss:
                result = "max_dd"
                break

            # Check pass
            if equity >= account_size + profit_target:
                result = "passed"
                break

        # Fill remaining days with last equity for plotting
        if day < time_limit_days:
            equity_paths[sim, day + 1:] = equity

        results_list.append(result)
        if result == "passed":
            pass_count += 1
        elif result == "daily_loss":
            fail_daily += 1
        elif result == "max_dd":
            fail_dd += 1
        else:
            fail_time += 1

    pass_rate = pass_count / n_simulations * 100.0
    total_fails = max(n_simulations - pass_count, 1)
    fail_reasons = {"daily_loss": fail_daily, "max_dd": fail_dd, "time_expired": fail_time}
    fail_reason_pcts = {
        k: round(v / n_simulations * 100.0, 1) for k, v in fail_reasons.items()
    }

    # Percentile bands
    pct_keys = ["5", "25", "50", "75", "95"]
    pct_vals = [5, 25, 50, 75, 95]
    percentiles = {
        k: np.nanpercentile(equity_paths, v, axis=0) for k, v in zip(pct_keys, pct_vals)
    }

    # Sample 50 paths for plotting
    sample_idx = rng.choice(n_simulations, size=min(50, n_simulations), replace=False)
    sampled_paths = equity_paths[sample_idx]

    # --- Historical pass rate: simulate starting on each unique trade date ---
    hist_results = _historical_prop_sim(
        trades, account_size, profit_target, daily_loss_limit,
        max_total_loss, time_limit_days, position_scale,
    )
    hist_pass_rate = 0.0
    if hist_results:
        hist_pass_rate = (
            sum(1 for r in hist_results if r["result"] == "passed")
            / len(hist_results) * 100.0
        )

    return {
        "pass_rate": round(pass_rate, 1),
        "fail_reasons": fail_reasons,
        "fail_reason_pcts": fail_reason_pcts,
        "equity_paths": sampled_paths,
        "percentiles": percentiles,
        "historical_pass_rate": round(hist_pass_rate, 1),
        "historical_results": hist_results,
    }


# ===================================================================
# Internal helpers
# ===================================================================

def _max_consecutive(mask: np.ndarray) -> int:
    """Return the length of the longest run of True values in *mask*."""
    if len(mask) == 0:
        return 0
    max_run = 0
    current = 0
    for v in mask:
        if v:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _sortino(equity: pd.Series) -> float:
    """Annualised Sortino ratio from an equity series (daily granularity)."""
    if equity is None or len(equity) < 2:
        return 0.0
    daily_returns = equity.pct_change().dropna()
    if len(daily_returns) < 2:
        return 0.0
    mean_ret = float(daily_returns.mean())
    downside = daily_returns[daily_returns < 0]
    if len(downside) < 1:
        return 0.0
    downside_std = float(downside.std(ddof=1))
    if downside_std == 0:
        return 0.0
    # Annualise assuming ~252 trading days
    return (mean_ret / downside_std) * np.sqrt(252)


def _max_drawdown_dollars(equity: pd.Series) -> float:
    """Maximum drawdown in absolute dollar terms from an equity series."""
    if equity is None or len(equity) < 2:
        return 0.0
    values = equity.values.astype(float)
    peak = np.maximum.accumulate(values)
    dd = peak - values
    return float(dd.max())


def _simulate_paths(
    pnls: np.ndarray,
    initial_capital: float,
    n_simulations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate reshuffled equity paths.  Shape: (n_simulations, len(pnls)+1)."""
    n = len(pnls)
    paths = np.empty((n_simulations, n + 1), dtype=float)
    paths[:, 0] = initial_capital

    for i in range(n_simulations):
        shuffled = rng.permutation(pnls)
        paths[i, 1:] = initial_capital + np.cumsum(shuffled)

    return paths


def _max_dd_pcts_from_paths(paths: np.ndarray) -> np.ndarray:
    """Return max drawdown % for each simulation row."""
    n_sims = paths.shape[0]
    dd_pcts = np.zeros(n_sims, dtype=float)
    for i in range(n_sims):
        eq = paths[i]
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.where(peak > 0, peak, 1.0)
        dd_pcts[i] = float(dd.max()) * 100.0
    return dd_pcts


def _run_stress_tests(
    pnls: np.ndarray,
    initial_capital: float,
    n_stress: int,
    rng: np.random.Generator,
) -> dict:
    """Run four stress-test variants and return summary stats for each."""

    def _summarise(modified_pnls: np.ndarray) -> dict:
        paths = _simulate_paths(modified_pnls, initial_capital, n_stress, rng)
        final = paths[:, -1]
        ret_pct = float(((final - initial_capital) / initial_capital).mean()) * 100.0
        dd_pcts = _max_dd_pcts_from_paths(paths)
        max_dd_pct = float(np.median(dd_pcts))
        profit_pct = float((final > initial_capital).mean()) * 100.0
        return {
            "return_pct": round(ret_pct, 2),
            "max_dd_pct": round(max_dd_pct, 2),
            "profit_pct": round(profit_pct, 1),
        }

    # 1. Double losses
    dl = pnls.copy()
    dl[dl < 0] *= 2.0

    # 2. Half wins
    hw = pnls.copy()
    hw[hw > 0] *= 0.5

    # 3. Extended drawdown: repeat worst 5 trades
    worst_idx = np.argsort(pnls)[:5]
    ed = np.concatenate([pnls, pnls[worst_idx]])

    # 4. Reduced win rate: flip 10% of winners to avg loss
    rw = pnls.copy()
    winners = np.where(rw > 0)[0]
    avg_loss = float(pnls[pnls < 0].mean()) if (pnls < 0).any() else 0.0
    n_flip = max(1, int(round(len(winners) * 0.1)))
    if len(winners) > 0:
        flip_idx = rng.choice(winners, size=min(n_flip, len(winners)), replace=False)
        rw[flip_idx] = avg_loss

    return {
        "double_losses": _summarise(dl),
        "half_wins": _summarise(hw),
        "extended_dd": _summarise(ed),
        "reduced_wr": _summarise(rw),
    }


def _historical_prop_sim(
    trades: List[Trade],
    account_size: float,
    profit_target: float,
    daily_loss_limit: float,
    max_total_loss: float,
    time_limit_days: int,
    position_scale: float,
) -> List[dict]:
    """Simulate the prop challenge starting at each unique trade date in order."""
    if not trades:
        return []

    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["date"] = df["entry_time"].dt.date
    df["pnl_scaled"] = df["pnl"] * position_scale
    df.sort_values("entry_time", inplace=True)

    unique_dates = sorted(df["date"].unique())
    results: List[dict] = []

    for start_date in unique_dates:
        end_date = start_date + pd.Timedelta(days=time_limit_days)
        window = df[(df["date"] >= start_date) & (df["date"] < end_date)]
        if window.empty:
            results.append({"start_date": start_date, "result": "time_expired"})
            continue

        equity = account_size
        peak = equity
        result = "time_expired"

        # Process day by day within window
        for day_date, day_trades in window.groupby("date"):
            day_pnl = float(day_trades["pnl_scaled"].sum())
            equity += day_pnl
            peak = max(peak, equity)

            if day_pnl <= -daily_loss_limit:
                result = "daily_loss"
                break
            if (peak - equity) >= max_total_loss:
                result = "max_dd"
                break
            if equity >= account_size + profit_target:
                result = "passed"
                break

        results.append({"start_date": start_date, "result": result})

    return results


def _empty_monte_carlo(initial_capital: float) -> dict:
    """Return a valid but empty Monte Carlo result dict."""
    return {
        "paths": np.array([[initial_capital]]),
        "percentiles": {k: np.array([initial_capital]) for k in ["5", "25", "50", "75", "95"]},
        "prob_profit": 0.0,
        "prob_2x": 0.0,
        "prob_50_dd": 0.0,
        "median_max_dd_pct": 0.0,
        "confidence_95": (initial_capital, initial_capital),
        "expected_final": initial_capital,
        "stress_tests": {
            k: {"return_pct": 0.0, "max_dd_pct": 0.0, "profit_pct": 0.0}
            for k in ["double_losses", "half_wins", "extended_dd", "reduced_wr"]
        },
    }


def _empty_prop_firm(account_size: float, time_limit_days: int) -> dict:
    """Return a valid but empty prop-firm simulation result dict."""
    return {
        "pass_rate": 0.0,
        "fail_reasons": {"daily_loss": 0, "max_dd": 0, "time_expired": 0},
        "fail_reason_pcts": {"daily_loss": 0.0, "max_dd": 0.0, "time_expired": 0.0},
        "equity_paths": np.full((1, time_limit_days + 1), account_size),
        "percentiles": {
            k: np.full(time_limit_days + 1, account_size)
            for k in ["5", "25", "50", "75", "95"]
        },
        "historical_pass_rate": 0.0,
        "historical_results": [],
    }
