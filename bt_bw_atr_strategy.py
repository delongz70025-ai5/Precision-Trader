"""
Backtrader implementation of Butterworth ATR Vol-Momentum Strategy.
Precomputes signals to guarantee numerical parity with the custom engine,
then uses backtrader's broker simulation for proper commission/slippage/margin.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import backtrader as bt
from scipy.signal import butter, lfilter
from strategy import atr as calc_atr, ema


# ──────────────────────────────────────────────────────────────────────────────
# Butterworth filter (same as strat_bw_atr_optimized.py)
# ──────────────────────────────────────────────────────────────────────────────

def butterworth_lowpass(series: np.ndarray, cutoff_period: float, order: int = 2) -> np.ndarray:
    wn = 2.0 / max(cutoff_period, 3)
    wn = np.clip(wn, 1e-6, 0.9999)
    b, a = butter(order, wn, btype="low", analog=False)
    return lfilter(b, a, series)


# ──────────────────────────────────────────────────────────────────────────────
# Precompute signals — add indicator columns to the DataFrame
# ──────────────────────────────────────────────────────────────────────────────

def precompute_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    close_arr = df["close"].values.astype(float)

    bw_fast = int(params.get("bw_fast_period", 3))
    bw_slow = int(params.get("bw_slow_period", 5))
    atr_len = int(params.get("atr_len", 10))
    vol_ema_len = int(params.get("vol_ema_len", 75))
    roc_len = int(params.get("atr_roc_len", 4))

    df["bw_fast"] = butterworth_lowpass(close_arr, bw_fast, order=2)
    df["bw_slow"] = butterworth_lowpass(close_arr, bw_slow, order=2)

    atr_series = calc_atr(df["high"], df["low"], df["close"], atr_len)
    df["atr_val"] = atr_series
    df["atr_ema"] = ema(atr_series, vol_ema_len)

    atr_arr = atr_series.values.astype(float)
    atr_roc = np.zeros(len(atr_arr))
    for i in range(roc_len, len(atr_arr)):
        prev = atr_arr[i - roc_len]
        if prev > 0:
            atr_roc[i] = (atr_arr[i] - prev) / prev
    df["atr_roc"] = atr_roc

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Custom PandasData feed with signal columns
# ──────────────────────────────────────────────────────────────────────────────

class BWATRPandasData(bt.feeds.PandasData):
    lines = ("bw_fast", "bw_slow", "atr_val", "atr_ema", "atr_roc")
    params = (
        ("bw_fast", "bw_fast"),
        ("bw_slow", "bw_slow"),
        ("atr_val", "atr_val"),
        ("atr_ema", "atr_ema"),
        ("atr_roc", "atr_roc"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Equity curve analyzer
# ──────────────────────────────────────────────────────────────────────────────

class EquityCurveAnalyzer(bt.Analyzer):
    def start(self):
        self.dates = []
        self.values = []

    def next(self):
        self.dates.append(self.data.datetime.datetime(0))
        self.values.append(self.strategy.broker.getvalue())

    def get_analysis(self):
        return {"dates": self.dates, "values": self.values}


# ──────────────────────────────────────────────────────────────────────────────
# Futures commission scheme (MNQ / NQ)
# ──────────────────────────────────────────────────────────────────────────────

class FuturesCommInfo(bt.CommInfoBase):
    params = (
        ("commission", 0.62),      # $0.62 per contract per side (typical MNQ)
        ("mult", 2.0),             # point value ($2 per point for MNQ)
        ("margin", 1000.0),        # ~$1000 margin per MNQ contract (not full notional)
        ("stocklike", False),
        ("commtype", bt.CommInfoBase.COMM_FIXED),
    )

    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * self.p.commission  # flat $ per contract


# ──────────────────────────────────────────────────────────────────────────────
# Backtrader Strategy
# ──────────────────────────────────────────────────────────────────────────────

class ButterworthATR_BTStrategy(bt.Strategy):
    params = (
        ("vol_expansion", 0.6),
        ("sl_atr_mult", 5.0),
        ("tp1_rr", 2.0),
        ("tp2_rr", 5.0),
        ("tp1_qty", 1),
        ("tp2_qty", 1),
        ("use_trail", True),
        ("session_open_et", 19),
        ("session_close_et", 16),
        ("use_force_close", True),
    )

    def __init__(self):
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.tp1_price = 0.0
        self.tp2_price = 0.0
        self.trail_stop = 0.0
        self.trade_dir = 0
        self.tp1_hit = False
        self.contracts = self.p.tp1_qty + self.p.tp2_qty
        self.trade_log = []

    def _bar_et_mins(self):
        dt = self.data.datetime.datetime(0)
        return dt.hour * 60 + dt.minute

    def _in_session(self):
        bar_mins = self._bar_et_mins()
        so = self.p.session_open_et * 60
        sc = self.p.session_close_et * 60
        if so > sc:
            return bar_mins >= so or bar_mins < sc
        return so <= bar_mins < sc

    def _past_ny_close(self):
        bar_mins = self._bar_et_mins()
        so = self.p.session_open_et * 60
        sc = self.p.session_close_et * 60
        return sc <= bar_mins < so

    def _log_trade(self, exit_reason):
        pnl = 0.0
        size = abs(self.position.size)
        if self.trade_dir == 1:
            pnl = (self.data.close[0] - self.entry_price) * size * 2.0  # point_value=2
        elif self.trade_dir == -1:
            pnl = (self.entry_price - self.data.close[0]) * size * 2.0
        self.trade_log.append({
            "exit_time": self.data.datetime.datetime(0),
            "entry_price": self.entry_price,
            "exit_price": self.data.close[0],
            "direction": self.trade_dir,
            "contracts": size,
            "pnl": pnl,
            "exit_reason": exit_reason,
        })

    def next(self):
        dt = self.data.datetime.datetime(0)
        c = self.data.close[0]
        h = self.data.high[0]
        l = self.data.low[0]

        # ── Force close at NY close ──────────────────────────────────────────
        if self.p.use_force_close and self._past_ny_close() and self.position.size != 0:
            self._log_trade("NY Close")
            self.close()
            self.trade_dir = 0
            self.tp1_hit = False
            return

        # ── Read precomputed signals ─────────────────────────────────────────
        if len(self.data) < 2:
            return

        fast_now = self.data.bw_fast[0]
        slow_now = self.data.bw_slow[0]
        fast_prev = self.data.bw_fast[-1]
        slow_prev = self.data.bw_slow[-1]

        atr_val = self.data.atr_val[0]
        atr_ema_val = self.data.atr_ema[0]
        atr_roc_val = self.data.atr_roc[0]

        if np.isnan(atr_val) or np.isnan(atr_ema_val):
            return

        vol_expanding = atr_val > self.p.vol_expansion * atr_ema_val
        vol_momentum = atr_roc_val > 0

        bull_cross = (fast_prev <= slow_prev) and (fast_now > slow_now)
        bear_cross = (fast_prev >= slow_prev) and (fast_now < slow_now)

        risk_pts = atr_val * self.p.sl_atr_mult if atr_val * self.p.sl_atr_mult > 0 else 25.0

        # ── Exit logic (bar-by-bar, matching custom engine) ──────────────────
        if self.position.size > 0 and self.trade_dir == 1:
            if not self.tp1_hit and h >= self.tp1_price:
                self.tp1_hit = True
                self.sell(size=self.p.tp1_qty)
                if self.p.use_trail:
                    self.trail_stop = self.entry_price

            if self.position.size > 0 and h >= self.tp2_price:
                self.sell(size=min(self.p.tp2_qty, self.position.size))

            if self.position.size > 0:
                stop = self.trail_stop if self.tp1_hit else self.sl_price
                if l <= stop:
                    self._log_trade("SL")
                    self.close()
                    self.trade_dir = 0
                    self.tp1_hit = False

        elif self.position.size < 0 and self.trade_dir == -1:
            if not self.tp1_hit and l <= self.tp1_price:
                self.tp1_hit = True
                self.buy(size=self.p.tp1_qty)
                if self.p.use_trail:
                    self.trail_stop = self.entry_price

            if self.position.size < 0 and l <= self.tp2_price:
                self.buy(size=min(self.p.tp2_qty, abs(self.position.size)))

            if self.position.size < 0:
                stop = self.trail_stop if self.tp1_hit else self.sl_price
                if h >= stop:
                    self._log_trade("SL")
                    self.close()
                    self.trade_dir = 0
                    self.tp1_hit = False

        # ── Entry logic ──────────────────────────────────────────────────────
        if self.position.size == 0 and self._in_session() and not self._past_ny_close():
            if bull_cross and vol_expanding and vol_momentum:
                self.entry_price = c
                self.sl_price = c - risk_pts
                self.tp1_price = c + risk_pts * self.p.tp1_rr
                self.tp2_price = c + risk_pts * self.p.tp2_rr
                self.trail_stop = self.sl_price
                self.trade_dir = 1
                self.tp1_hit = False
                self.buy(size=self.contracts)

            elif bear_cross and vol_expanding and vol_momentum:
                self.entry_price = c
                self.sl_price = c + risk_pts
                self.tp1_price = c - risk_pts * self.p.tp1_rr
                self.tp2_price = c - risk_pts * self.p.tp2_rr
                self.trail_stop = self.sl_price
                self.trade_dir = -1
                self.tp1_hit = False
                self.sell(size=self.contracts)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point — run a single backtrader backtest
# ──────────────────────────────────────────────────────────────────────────────

def run_bt_backtest(df: pd.DataFrame, params: dict) -> dict:
    """
    Run the BW-ATR strategy through backtrader's engine.
    Returns dict with: equity (pd.Series), stats (dict), trades (list), analyzers (dict)
    """
    capital = float(params.get("initial_capital", 50000.0))
    point_value = float(params.get("point_value", 2.0))

    # Precompute signals
    df_sig = precompute_signals(df, params)
    df_sig = df_sig.dropna(subset=["bw_fast", "bw_slow", "atr_val", "atr_ema", "atr_roc"])

    # Strip timezone for backtrader compatibility
    if df_sig.index.tz is not None:
        df_sig.index = df_sig.index.tz_localize(None)

    # Setup cerebro (optimized for large datasets)
    cerebro = bt.Cerebro(
        preload=True,
        runonce=True,
        optdatas=True,
        optreturn=True,
        exactbars=False,
    )
    cerebro.broker.setcash(capital)

    # Futures commission
    comminfo = FuturesCommInfo(mult=point_value)
    cerebro.broker.addcommissioninfo(comminfo)

    # Data feed
    data = BWATRPandasData(dataname=df_sig, datetime=None)
    cerebro.adddata(data)

    # Strategy
    _tp1 = float(params.get("tp1_rr", 2.0))
    _tp2 = float(params.get("tp2_rr", 5.0))
    if _tp2 < _tp1:
        _tp2 = _tp1
    strat_params = {
        "vol_expansion": float(params.get("vol_expansion", 0.6)),
        "sl_atr_mult": float(params.get("sl_atr_mult", 5.0)),
        "tp1_rr": _tp1,
        "tp2_rr": _tp2,
        "tp1_qty": int(params.get("tp1_qty", 1)),
        "tp2_qty": int(params.get("tp2_qty", 1)),
        "use_trail": bool(params.get("use_trail", True)),
        "session_open_et": int(params.get("session_open_et", 19)),
        "session_close_et": int(params.get("session_close_et", 16)),
        "use_force_close": bool(params.get("use_force_close", True)),
    }
    cerebro.addstrategy(ButterworthATR_BTStrategy, **strat_params)

    # Analyzers
    cerebro.addanalyzer(EquityCurveAnalyzer, _name="equity_curve")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    # Run
    results = cerebro.run()
    strat = results[0]

    # Extract equity curve
    eq_data = strat.analyzers.equity_curve.get_analysis()
    eq_series = pd.Series(eq_data["values"], index=eq_data["dates"])

    # Extract analyzer results
    sharpe_res = strat.analyzers.sharpe.get_analysis()
    dd_res = strat.analyzers.drawdown.get_analysis()
    trade_res = strat.analyzers.trades.get_analysis()
    returns_res = strat.analyzers.returns.get_analysis()

    # Build stats dict (compatible with app's metric cards)
    total_closed = trade_res.get("total", {}).get("closed", 0)
    won = trade_res.get("won", {}).get("total", 0)
    lost = trade_res.get("lost", {}).get("total", 0)
    win_rate = won / total_closed if total_closed > 0 else 0.0

    gross_profit = trade_res.get("won", {}).get("pnl", {}).get("total", 0.0)
    gross_loss = abs(trade_res.get("lost", {}).get("pnl", {}).get("total", 0.0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    final_value = cerebro.broker.getvalue()
    total_pnl = final_value - capital

    stats = {
        "total_trades": total_closed,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "profit_factor": pf,
        "max_drawdown": -dd_res.get("max", {}).get("drawdown", 0.0) / 100.0,
        "sharpe": sharpe_res.get("sharperatio", 0.0) or 0.0,
        "avg_win": trade_res.get("won", {}).get("pnl", {}).get("average", 0.0),
        "avg_loss": trade_res.get("lost", {}).get("pnl", {}).get("average", 0.0),
        "net_return_pct": (final_value - capital) / capital * 100,
        "final_value": final_value,
    }

    # Analyzer details for display
    analyzers = {
        "sharpe": sharpe_res,
        "drawdown": dd_res,
        "trades": trade_res,
        "returns": returns_res,
    }

    return {
        "equity": eq_series,
        "stats": stats,
        "trades": strat.trade_log,
        "analyzers": analyzers,
        "cerebro": cerebro,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Generic backtrader runner — works with ANY strategy from the registry
# ──────────────────────────────────────────────────────────────────────────────
# Approach: run the strategy's own engine to get trade signals, then replay
# those trades through backtrader's broker for proper commission/slippage/margin.

class SignalReplayStrategy(bt.Strategy):
    """Replays pre-computed trades through backtrader's broker."""
    params = (("trade_signals", None),)

    def __init__(self):
        # Build a lookup: bar_index -> list of actions
        self._actions = {}
        if self.p.trade_signals:
            for sig in self.p.trade_signals:
                idx = sig["bar_index"]
                if idx not in self._actions:
                    self._actions[idx] = []
                self._actions[idx].append(sig)
        self._bar = 0

    def next(self):
        actions = self._actions.get(self._bar, [])
        for act in actions:
            if act["action"] == "buy":
                self.buy(size=act["size"])
            elif act["action"] == "sell":
                self.sell(size=act["size"])
            elif act["action"] == "close":
                self.close()
            elif act["action"] == "sell_partial":
                # Partial close of a long position
                self.sell(size=act["size"])
            elif act["action"] == "buy_partial":
                # Partial close of a short position
                self.buy(size=act["size"])
        self._bar += 1


def run_bt_generic_fast(df: pd.DataFrame, strategy_key: str, params: dict) -> dict:
    """
    Fast path: Run strategy's native engine ONLY (no backtrader replay).
    Use this for optimizer, walk-forward, and analysis where speed matters.
    The native engine already computes accurate P&L with commissions.
    """
    from strategy_registry import get_strategy

    capital = float(params.get("initial_capital", 50000.0))
    strat = get_strategy(strategy_key)
    merged = {**strat.frozen_params(), **params}
    result = strat.run(df, merged)

    native_trades = result.get("trades", [])
    stats = result.get("stats", {})
    equity = result.get("equity", pd.Series(dtype=float))

    # Ensure all expected stat keys are present
    if "final_value" not in stats:
        stats["final_value"] = capital + stats.get("total_pnl", 0)
    if "net_return_pct" not in stats:
        stats["net_return_pct"] = (stats["final_value"] - capital) / capital * 100 if capital else 0
    for k in ("total_trades", "win_rate", "total_pnl", "profit_factor",
              "max_drawdown", "sharpe", "avg_win", "avg_loss"):
        if k not in stats:
            stats[k] = 0

    # Compute analyzer-compatible stats from native trades
    won = sum(1 for t in native_trades if t.get("pnl", 0) > 0)
    lost = sum(1 for t in native_trades if t.get("pnl", 0) < 0)
    total = len(native_trades)

    return {
        "equity": equity,
        "stats": stats,
        "trades": native_trades,
        "analyzers": {
            "trades": {
                "total": {"closed": stats.get("total_trades", total)},
                "won": {"total": won},
                "lost": {"total": lost},
            },
            "drawdown": {"max": {"drawdown": abs(stats.get("max_drawdown", 0)) * 100}},
            "sharpe": {"sharperatio": stats.get("sharpe", 0)},
        },
    }


def resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a higher timeframe.
    Call this BEFORE passing data to run_bt_generic or any strategy.

    target_tf: pandas resample rule — "1h", "15min", "5min", "1D", etc.
    """
    if df.empty:
        return df
    resampled = df.resample(target_tf).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open", "high", "low", "close"])
    return resampled


def run_bt_generic(df: pd.DataFrame, strategy_key: str, params: dict) -> dict:
    """
    Run ANY strategy from the registry through backtrader's broker.

    1. Runs the strategy's own backtest engine to get trades
    2. Converts trades into bar-indexed signals
    3. Replays signals through backtrader for broker-accurate P&L
    """
    from strategy_registry import get_strategy

    capital = float(params.get("initial_capital", 50000.0))
    point_value = float(params.get("point_value", 2.0))

    # Step 1: Run the strategy's native engine
    strat = get_strategy(strategy_key)
    merged = {**strat.frozen_params(), **params}
    result = strat.run(df, merged)
    native_trades = result.get("trades", [])

    if not native_trades:
        return {
            "equity": pd.Series(dtype=float),
            "stats": {"total_trades": 0, "win_rate": 0, "total_pnl": 0,
                      "profit_factor": 0, "max_drawdown": 0, "sharpe": 0,
                      "avg_win": 0, "avg_loss": 0, "net_return_pct": 0, "final_value": capital},
            "trades": [],
            "analyzers": {},
            "native_result": result,
        }

    # Step 2: Convert trades to bar-indexed signals
    dt_to_idx = {dt: i for i, dt in enumerate(df.index)}

    from collections import defaultdict
    trade_groups = defaultdict(list)
    for t in native_trades:
        key = (t.get("entry_time"), t.get("direction", 0))
        trade_groups[key].append(t)

    signals = []
    for (entry_dt, direction), group in trade_groups.items():
        if entry_dt not in dt_to_idx:
            continue
        entry_idx = dt_to_idx[entry_dt]
        total_contracts = sum(t.get("contracts", 1) for t in group)

        if direction == 1:
            signals.append({"bar_index": entry_idx, "action": "buy", "size": total_contracts})
        elif direction == -1:
            signals.append({"bar_index": entry_idx, "action": "sell", "size": total_contracts})

        for t in group:
            exit_dt = t.get("exit_time")
            contracts = t.get("contracts", 1)
            if exit_dt in dt_to_idx:
                exit_idx = dt_to_idx[exit_dt]
                if direction == 1:
                    signals.append({"bar_index": exit_idx, "action": "sell_partial", "size": contracts})
                elif direction == -1:
                    signals.append({"bar_index": exit_idx, "action": "buy_partial", "size": contracts})

    # Step 3: Run through backtrader (optimized for large datasets)
    # Avoid full copy of 800K+ row DataFrame — only strip timezone from index
    if df.index.tz is not None:
        df_bt = df.set_index(df.index.tz_localize(None))
    else:
        df_bt = df

    cerebro = bt.Cerebro(
        preload=True,         # Load entire dataset into memory upfront (faster iteration)
        runonce=True,         # Vectorized indicator mode (batch, not bar-by-bar)
        optdatas=True,        # Optimize data access patterns
        optreturn=True,       # Lightweight return objects (less memory overhead)
        exactbars=False,      # Keep full data in memory (faster than streaming)
    )
    cerebro.broker.setcash(capital)
    comminfo = FuturesCommInfo(mult=point_value)
    cerebro.broker.addcommissioninfo(comminfo)

    data = bt.feeds.PandasData(dataname=df_bt, datetime=None)
    cerebro.adddata(data)
    cerebro.addstrategy(SignalReplayStrategy, trade_signals=signals)

    cerebro.addanalyzer(EquityCurveAnalyzer, _name="equity_curve")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    results = cerebro.run()
    bt_strat = results[0]

    # Extract results
    eq_data = bt_strat.analyzers.equity_curve.get_analysis()
    eq_series = pd.Series(eq_data["values"], index=eq_data["dates"])

    sharpe_res = bt_strat.analyzers.sharpe.get_analysis()
    dd_res = bt_strat.analyzers.drawdown.get_analysis()
    trade_res = bt_strat.analyzers.trades.get_analysis()
    returns_res = bt_strat.analyzers.returns.get_analysis()

    total_closed = trade_res.get("total", {}).get("closed", 0)
    won = trade_res.get("won", {}).get("total", 0)
    lost = trade_res.get("lost", {}).get("total", 0)
    win_rate = won / total_closed if total_closed > 0 else 0.0

    gross_profit = trade_res.get("won", {}).get("pnl", {}).get("total", 0.0)
    gross_loss = abs(trade_res.get("lost", {}).get("pnl", {}).get("total", 0.0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    final_value = cerebro.broker.getvalue()
    total_pnl = final_value - capital

    stats = {
        "total_trades": total_closed,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "profit_factor": pf,
        "max_drawdown": -dd_res.get("max", {}).get("drawdown", 0.0) / 100.0,
        "sharpe": sharpe_res.get("sharperatio", 0.0) or 0.0,
        "avg_win": trade_res.get("won", {}).get("pnl", {}).get("average", 0.0),
        "avg_loss": trade_res.get("lost", {}).get("pnl", {}).get("average", 0.0),
        "net_return_pct": (final_value - capital) / capital * 100,
        "final_value": final_value,
    }

    return {
        "equity": eq_series,
        "stats": stats,
        "trades": native_trades,
        "analyzers": {
            "sharpe": sharpe_res, "drawdown": dd_res,
            "trades": trade_res, "returns": returns_res,
        },
        "native_result": result,
    }
