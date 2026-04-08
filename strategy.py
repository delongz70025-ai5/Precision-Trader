"""
Precision Sniper v7.5 — Python translation of the PineScript strategy.
All logic mirrors the original as closely as possible.
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Helper indicators
# ──────────────────────────────────────────────────────────────────────────────

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=length - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=length - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=length - 1, adjust=False).mean()


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               factor: float = 4.0, atr_len: int = 10):
    atr_val = atr(high, low, close, atr_len)
    hl2 = (high + low) / 2
    ub = (hl2 + factor * atr_val).values.copy()
    lb = (hl2 - factor * atr_val).values.copy()
    close_arr = close.values
    n = len(close_arr)
    dir_arr = np.ones(n)
    st_arr = np.full(n, np.nan)

    for i in range(1, n):
        # Clamp bands
        if ub[i] >= ub[i - 1] and close_arr[i - 1] <= ub[i - 1]:
            ub[i] = ub[i - 1]
        if lb[i] <= lb[i - 1] and close_arr[i - 1] >= lb[i - 1]:
            lb[i] = lb[i - 1]

        if dir_arr[i - 1] == 1 and close_arr[i] < lb[i]:
            dir_arr[i] = -1
        elif dir_arr[i - 1] == -1 and close_arr[i] > ub[i]:
            dir_arr[i] = 1
        else:
            dir_arr[i] = dir_arr[i - 1]

        st_arr[i] = lb[i] if dir_arr[i] == 1 else ub[i]

    return (pd.Series(st_arr, index=close.index),
            pd.Series(dir_arr, index=close.index))


def dmi(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_val = atr(high, low, close, length)
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(com=length - 1, adjust=False).mean() / atr_val
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(com=length - 1, adjust=False).mean() / atr_val
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(com=length - 1, adjust=False).mean()
    return plus_di, minus_di, adx


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    hlc3 = (high + low + close) / 3
    return (hlc3 * volume).cumsum() / volume.cumsum()


# ──────────────────────────────────────────────────────────────────────────────
# Strategy parameters dataclass
# ──────────────────────────────────────────────────────────────────────────────

class StrategyParams:
    def __init__(self, **kwargs):
        # Entry Engine
        self.ema_fast_len      = kwargs.get("ema_fast_len", 10)
        self.ema_slow_len      = kwargs.get("ema_slow_len", 21)
        self.ema_trend_len     = kwargs.get("ema_trend_len", 55)
        self.min_score         = kwargs.get("min_score", 5)
        self.rsi_len           = kwargs.get("rsi_len", 26)
        self.use_pullback      = kwargs.get("use_pullback", True)
        self.pullback_score    = kwargs.get("pullback_score", 4)

        # Risk Management
        self.use_atr_risk      = kwargs.get("use_atr_risk", False)
        self.atr_risk_len      = kwargs.get("atr_risk_len", 15)
        self.atr_risk_mult     = kwargs.get("atr_risk_mult", 1.5)
        self.fixed_risk_pts    = kwargs.get("fixed_risk_pts", 25.0)
        self.tp1_rr            = kwargs.get("tp1_rr", 2.0)
        self.tp2_rr            = kwargs.get("tp2_rr", 3.5)
        self.tp3_rr            = kwargs.get("tp3_rr", 6.0)
        # Enforce TP ordering: tp1 <= tp2 <= tp3
        if self.tp2_rr < self.tp1_rr:
            self.tp2_rr = self.tp1_rr
        if self.tp3_rr < self.tp2_rr:
            self.tp3_rr = self.tp2_rr
        self.use_trail         = kwargs.get("use_trail", True)
        self.trail_after_tp    = kwargs.get("trail_after_tp", "TP2")
        self.tp1_qty           = kwargs.get("tp1_qty", 1)
        self.tp2_qty           = kwargs.get("tp2_qty", 1)
        self.tp3_qty           = kwargs.get("tp3_qty", 0)

        # Filters
        self.use_supertrend    = kwargs.get("use_supertrend", True)
        self.st_atr_len        = kwargs.get("st_atr_len", 10)
        self.st_factor         = kwargs.get("st_factor", 4.0)

        # Prop Firm Safety
        self.use_force_close      = kwargs.get("use_force_close", True)
        self.force_close_hour     = kwargs.get("force_close_hour", 16)
        self.force_close_minute   = kwargs.get("force_close_minute", 0)
        self.no_hold_before_hour  = kwargs.get("no_hold_before_hour", 8)
        self.no_hold_before_min   = kwargs.get("no_hold_before_min", 0)
        self.use_max_trade_loss   = kwargs.get("use_max_trade_loss", True)
        self.max_trade_loss       = kwargs.get("max_trade_loss", 200.0)
        self.use_daily_max_loss   = kwargs.get("use_daily_max_loss", True)
        self.daily_max_loss       = kwargs.get("daily_max_loss", 650.0)

        # General
        self.allow_longs          = kwargs.get("allow_longs", True)
        self.allow_shorts         = kwargs.get("allow_shorts", True)
        self.use_session          = kwargs.get("use_session", True)
        self.initial_capital      = kwargs.get("initial_capital", 50000.0)

        # MNQ contract spec
        self.point_value          = kwargs.get("point_value", 2.0)   # $2 per point for MNQ
        self.tick_size            = kwargs.get("tick_size", 0.25)

        # Costs
        self.exchange_fee_pct     = kwargs.get("exchange_fee_pct", 0.0010)   # 0.10%
        self.slippage_pct         = kwargs.get("slippage_pct", 0.0005)       # 0.05%

    @property
    def total_contracts(self):
        return self.tp1_qty + self.tp2_qty + self.tp3_qty

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


# ──────────────────────────────────────────────────────────────────────────────
# Core backtest engine
# ──────────────────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, params: StrategyParams) -> dict:
    """
    Run a single backtest on df (must have: open, high, low, close, volume columns,
    DatetimeIndex in US/Eastern or UTC).

    Returns a dict with:
        trades   : list of trade dicts
        equity   : pd.Series (daily equity)
        stats    : summary metrics dict
    """
    p = params
    total_contracts = p.total_contracts
    if total_contracts == 0:
        total_contracts = 2

    # ── Pre-compute indicators ──────────────────────────────────────────────
    src = df["close"]
    hi  = df["high"]
    lo  = df["low"]
    vol = df["volume"]

    ema_fast  = ema(src, p.ema_fast_len)
    ema_slow  = ema(src, p.ema_slow_len)
    ema_trend = ema(src, p.ema_trend_len)

    rsi_val  = rsi(src, p.rsi_len)
    atr_risk = atr(hi, lo, src, p.atr_risk_len)
    atr_st   = atr(hi, lo, src, p.st_atr_len)

    macd_line, macd_sig, macd_hist = macd(src, 12, 26, 9)
    vol_sma   = vol.rolling(20).mean()
    vol_above = vol > vol_sma * 1.2

    di_plus, di_minus, adx_val = dmi(hi, lo, src, 14)
    strong_trend = adx_val > 20

    vwap_val = vwap(hi, lo, src, vol)

    atr_sma_global = atr_risk.rolling(42).mean()
    vol_ratio = atr_risk / atr_sma_global.replace(0, np.nan)

    st_line, st_dir = supertrend(hi, lo, src, p.st_factor, p.st_atr_len)
    st_bull = st_dir == 1
    st_bear = st_dir == -1

    warmup = max(p.ema_trend_len, 50)

    # ── Pre-extract numpy arrays for speed (avoids .iloc[] overhead) ────────
    _open_arr   = df["open"].values
    _high_arr   = df["high"].values
    _low_arr    = df["low"].values
    _close_arr  = df["close"].values
    _vol_arr    = vol.values
    _ef_arr     = ema_fast.values
    _es_arr     = ema_slow.values
    _et_arr     = ema_trend.values
    _rsi_arr    = rsi_val.values
    _mh_arr     = macd_hist.values
    _ml_arr     = macd_line.values
    _ms_arr     = macd_sig.values
    _vw_arr     = vwap_val.values
    _va_arr     = vol_above.values
    _st_strong  = strong_trend.values
    _dp_arr     = di_plus.values
    _dm_arr     = di_minus.values
    _atr_risk_arr = atr_risk.values
    _stbull_arr = st_bull.values
    _stbear_arr = st_bear.values
    _stdir_arr  = st_dir.values
    _vr_arr     = vol_ratio.values
    _idx_arr    = df.index

    # ── Trade state ─────────────────────────────────────────────────────────
    equity        = p.initial_capital
    trades        = []
    equity_curve  = []
    dates         = []

    pos_size      = 0          # +n long, -n short (contracts)
    entry_price   = np.nan
    sl_price      = np.nan
    tp1_price     = np.nan
    tp2_price     = np.nan
    tp3_price     = np.nan
    trail_price   = np.nan
    trade_dir     = 0
    trade_risk    = np.nan
    tp1_hit       = False
    tp2_hit       = False
    entry_type    = ""
    last_entry_bar = -1

    day_start_equity = equity
    daily_loss_breached = False
    prev_date     = None

    # Cost per contract per side (entry + exit)
    cost_pct = p.exchange_fee_pct + p.slippage_pct   # 0.15% per side

    n = len(df)

    for i in range(n):
        bar_time  = _idx_arr[i]
        bar_date  = bar_time.date()
        o = _open_arr[i]
        h = _high_arr[i]
        l = _low_arr[i]
        c = _close_arr[i]
        v = _vol_arr[i]

        # ── New day reset ───────────────────────────────────────────────────
        if prev_date is None or bar_date != prev_date:
            day_start_equity    = equity
            daily_loss_breached = False
            prev_date           = bar_date

        daily_pnl = equity - day_start_equity
        if p.use_daily_max_loss and daily_pnl <= -p.daily_max_loss and pos_size != 0:
            daily_loss_breached = True

        # ── Prop firm session check ─────────────────────────────────────────
        bar_hour   = bar_time.hour
        bar_minute = bar_time.minute
        bar_t_mins = bar_hour * 60 + bar_minute
        close_t    = p.force_close_hour * 60 + p.force_close_minute
        open_t     = p.no_hold_before_hour * 60 + p.no_hold_before_min
        outside_window = p.use_force_close and (bar_t_mins >= close_t or bar_t_mins < open_t)

        # ── US regular session 08:30–15:00 CT (09:30–16:00 ET) ─────────────
        if p.use_session:
            in_session = (bar_t_mins >= 9 * 60 + 30) and (bar_t_mins < 16 * 60)
        else:
            in_session = True

        # ── Force close ─────────────────────────────────────────────────────
        def close_position(comment=""):
            nonlocal pos_size, entry_price, sl_price, tp1_price, tp2_price
            nonlocal tp3_price, trail_price, trade_dir, trade_risk
            nonlocal tp1_hit, tp2_hit, entry_type, equity
            if pos_size == 0:
                return
            exit_p = c
            cost   = abs(exit_p * abs(pos_size) * p.point_value) * cost_pct
            pnl    = (exit_p - entry_price) * pos_size * p.point_value - cost
            equity += pnl
            trades.append({
                "entry_time": entry_time,
                "exit_time":  bar_time,
                "entry_price": entry_price,
                "exit_price":  exit_p,
                "direction":   trade_dir,
                "contracts":   abs(pos_size),
                "pnl":         pnl,
                "exit_reason": comment,
                "entry_type":  entry_type,
                "tp1_hit":     tp1_hit,
                "tp2_hit":     tp2_hit,
            })
            pos_size   = 0
            trade_dir  = 0
            entry_price = np.nan

        if outside_window and pos_size != 0:
            close_position("Session Close")
        if daily_loss_breached and pos_size != 0:
            close_position("Daily Max Loss")

        # ── Unrealized P&L and emergency exit ───────────────────────────────
        if pos_size != 0 and p.use_max_trade_loss:
            unreal = (c - entry_price) * pos_size * p.point_value
            if unreal <= -p.max_trade_loss:
                close_position("Max Trade Loss")

        # ── Risk points ─────────────────────────────────────────────────────
        risk_pts = max(float(_atr_risk_arr[i]) * p.atr_risk_mult, 4.0) if p.use_atr_risk else p.fixed_risk_pts

        # ── Confluence scores ───────────────────────────────────────────────
        ef  = _ef_arr[i]
        es  = _es_arr[i]
        et  = _et_arr[i]
        rsi_ = _rsi_arr[i] if not np.isnan(_rsi_arr[i]) else 50.0
        mh  = _mh_arr[i] if not np.isnan(_mh_arr[i]) else 0.0
        ml  = _ml_arr[i] if not np.isnan(_ml_arr[i]) else 0.0
        ms  = _ms_arr[i] if not np.isnan(_ms_arr[i]) else 0.0
        vw  = _vw_arr[i] if not np.isnan(_vw_arr[i]) else c
        va  = bool(_va_arr[i]) if not np.isnan(_va_arr[i]) else True
        st_ = bool(_st_strong[i]) if not np.isnan(_st_strong[i]) else False
        dp  = _dp_arr[i] if not np.isnan(_dp_arr[i]) else 0.0
        dm  = _dm_arr[i] if not np.isnan(_dm_arr[i]) else 0.0

        bull_score = 0.0
        bull_score += 1.0 if ef > es else 0.0
        bull_score += 1.0 if c > et else 0.0
        bull_score += 1.0 if 50 < rsi_ < 70 else 0.0
        bull_score += 1.0 if mh > 0 else 0.0
        bull_score += 1.0 if ml > ms else 0.0
        bull_score += 1.0 if c > vw else 0.0
        bull_score += 1.0 if va else 0.0
        bull_score += 1.0 if st_ and dp > dm else 0.0
        # HTF bias: without actual HTF data, score 0 (neutral — matches PineScript when htfBias == 0)
        bull_score += 0.5 if c > ef else 0.0

        bear_score = 0.0
        bear_score += 1.0 if ef < es else 0.0
        bear_score += 1.0 if c < et else 0.0
        bear_score += 1.0 if 30 < rsi_ < 50 else 0.0
        bear_score += 1.0 if mh < 0 else 0.0
        bear_score += 1.0 if ml < ms else 0.0
        bear_score += 1.0 if c < vw else 0.0
        bear_score += 1.0 if va else 0.0
        bear_score += 1.0 if st_ and dm > dp else 0.0
        bear_score += 0.5 if c < ef else 0.0

        # ── Supertrend filter ────────────────────────────────────────────────
        st_b = bool(_stbull_arr[i]) if not np.isnan(_stdir_arr[i]) else True
        st_br = bool(_stbear_arr[i]) if not np.isnan(_stdir_arr[i]) else False
        st_pass_long  = (not p.use_supertrend) or st_b
        st_pass_short = (not p.use_supertrend) or st_br

        # ── Signals ──────────────────────────────────────────────────────────
        if i < warmup or i < 1:
            equity_curve.append(equity)
            dates.append(bar_time)
            continue

        ef_prev = _ef_arr[i - 1]
        es_prev = _es_arr[i - 1]

        ema_bull_cross = (ef_prev <= es_prev) and (ef > es)
        ema_bear_cross = (ef_prev >= es_prev) and (ef < es)

        rsi_not_ob = rsi_ < 75
        rsi_not_os = rsi_ > 25

        cross_buy  = p.allow_longs  and ema_bull_cross and rsi_not_ob and bull_score >= p.min_score
        cross_sell = p.allow_shorts and ema_bear_cross and rsi_not_os and bear_score >= p.min_score

        trend_bullish = (ef > es) and (c > et)
        trend_bearish = (ef < es) and (c < et)

        hi_prev1 = _high_arr[i - 1]
        hi_prev2 = _high_arr[i - 2] if i >= 2 else hi_prev1
        lo_prev1 = _low_arr[i - 1]
        lo_prev2 = _low_arr[i - 2] if i >= 2 else lo_prev1
        ef_prev1 = _ef_arr[i - 1]
        ef_prev2 = _ef_arr[i - 2] if i >= 2 else ef_prev1

        bull_pb = trend_bullish and (c > ef) and (l <= ef or lo_prev1 <= ef_prev1 or lo_prev2 <= ef_prev2)
        bear_pb = trend_bearish and (c < ef) and (h >= ef or hi_prev1 >= ef_prev1 or hi_prev2 >= ef_prev2)

        pb_buy  = p.use_pullback and p.allow_longs  and bull_pb and rsi_not_ob and bull_score >= p.pullback_score and not cross_buy
        pb_sell = p.use_pullback and p.allow_shorts and bear_pb and rsi_not_os and bear_score >= p.pullback_score and not cross_sell

        raw_buy  = cross_buy  or pb_buy
        raw_sell = cross_sell or pb_sell

        filt_buy  = raw_buy  and st_pass_long  and in_session and not outside_window
        filt_sell = raw_sell and st_pass_short and in_session and not outside_window

        no_pos  = pos_size == 0
        buy_cond  = filt_buy  and (no_pos or pos_size < 0) and not daily_loss_breached
        sell_cond = filt_sell and (no_pos or pos_size > 0) and not daily_loss_breached

        can_enter = (i != last_entry_bar)

        if buy_cond and sell_cond:
            sell_cond = False

        # ── Entry ────────────────────────────────────────────────────────────
        def enter_long():
            nonlocal pos_size, entry_price, sl_price, tp1_price, tp2_price
            nonlocal tp3_price, trail_price, trade_dir, trade_risk
            nonlocal tp1_hit, tp2_hit, entry_type, last_entry_bar, entry_time, equity
            if pos_size < 0:
                close_position("Reverse to Long")
            entry_price = c
            trade_dir   = 1
            trade_risk  = risk_pts
            sl_price    = entry_price - trade_risk
            tp1_price   = entry_price + trade_risk * p.tp1_rr
            tp2_price   = entry_price + trade_risk * p.tp2_rr
            tp3_price   = entry_price + trade_risk * p.tp3_rr
            trail_price = sl_price
            tp1_hit     = False
            tp2_hit     = False
            entry_type  = "Cross" if cross_buy else "Pullback"
            last_entry_bar = i
            entry_time  = bar_time
            pos_size    = total_contracts
            # Entry cost
            cost = entry_price * pos_size * p.point_value * cost_pct
            equity -= cost

        def enter_short():
            nonlocal pos_size, entry_price, sl_price, tp1_price, tp2_price
            nonlocal tp3_price, trail_price, trade_dir, trade_risk
            nonlocal tp1_hit, tp2_hit, entry_type, last_entry_bar, entry_time, equity
            if pos_size > 0:
                close_position("Reverse to Short")
            entry_price = c
            trade_dir   = -1
            trade_risk  = risk_pts
            sl_price    = entry_price + trade_risk
            tp1_price   = entry_price - trade_risk * p.tp1_rr
            tp2_price   = entry_price - trade_risk * p.tp2_rr
            tp3_price   = entry_price - trade_risk * p.tp3_rr
            trail_price = sl_price
            tp1_hit     = False
            tp2_hit     = False
            entry_type  = "Cross" if cross_sell else "Pullback"
            last_entry_bar = i
            entry_time  = bar_time
            pos_size    = -total_contracts
            cost = entry_price * abs(pos_size) * p.point_value * cost_pct
            equity -= cost

        entry_time = bar_time  # will be set properly inside enter_*

        if buy_cond and can_enter:
            enter_long()
        elif sell_cond and can_enter:
            enter_short()

        # ── Trailing stop step-up (tracking only, no exits) ──────────────────
        # Matches PineScript v7.5 "NO-TP EXIT TEST": trades exit ONLY on
        # opposite signal, session close, max trade loss, or daily max loss.
        if pos_size > 0 and trade_dir == 1:
            if not tp1_hit and h >= tp1_price:
                tp1_hit = True
                if p.use_trail and p.trail_after_tp == "TP1":
                    trail_price = entry_price
            if not tp2_hit and h >= tp2_price:
                tp2_hit = True
                if p.use_trail:
                    trail_price = entry_price if p.trail_after_tp == "TP2" else tp1_price

        elif pos_size < 0 and trade_dir == -1:
            if not tp1_hit and l <= tp1_price:
                tp1_hit = True
                if p.use_trail and p.trail_after_tp == "TP1":
                    trail_price = entry_price
            if not tp2_hit and l <= tp2_price:
                tp2_hit = True
                if p.use_trail:
                    trail_price = entry_price if p.trail_after_tp == "TP2" else tp1_price

        equity_curve.append(equity)
        dates.append(bar_time)

    # ── Close any open position at end ───────────────────────────────────────
    if pos_size != 0:
        exit_p = df["close"].iloc[-1]
        bar_time = df.index[-1]
        cost = exit_p * abs(pos_size) * p.point_value * cost_pct
        pnl  = (exit_p - entry_price) * pos_size * p.point_value - cost
        equity += pnl
        trades.append({
            "entry_time": entry_time,
            "exit_time":  bar_time,
            "entry_price": entry_price,
            "exit_price":  exit_p,
            "direction":   trade_dir,
            "contracts":   abs(pos_size),
            "pnl":         pnl,
            "exit_reason": "End of Data",
            "entry_type":  entry_type,
            "tp1_hit":     tp1_hit,
            "tp2_hit":     tp2_hit,
        })
        equity_curve[-1] = equity

    eq_series = pd.Series(equity_curve, index=dates)
    stats = compute_stats(trades, eq_series, p.initial_capital)

    return {
        "trades":      trades,
        "equity":      eq_series,
        "stats":       stats,
        "params":      params.to_dict(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_stats(trades: list, equity: pd.Series, initial_capital: float) -> dict:
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "net_return_pct": 0.0,
        }

    pnls   = [t["pnl"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl   = sum(pnls)
    win_rate    = len(wins) / len(pnls) if pnls else 0.0
    avg_win     = np.mean(wins) if wins else 0.0
    avg_loss    = np.mean(losses) if losses else 0.0
    expectancy  = win_rate * avg_win + (1 - win_rate) * avg_loss
    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    pf           = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Drawdown
    running_max = equity.cummax()
    dd          = (equity - running_max) / running_max
    max_dd      = float(dd.min())

    # Daily returns for Sharpe
    daily_eq    = equity.resample("1D").last().dropna()
    daily_ret   = daily_eq.pct_change().dropna()
    sharpe      = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0

    net_ret_pct = (equity.iloc[-1] - initial_capital) / initial_capital * 100 if initial_capital > 0 else 0.0

    return {
        "total_trades":   len(pnls),
        "win_rate":       win_rate,
        "total_pnl":      total_pnl,
        "profit_factor":  pf,
        "max_drawdown":   max_dd,
        "sharpe":         float(sharpe),
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
        "expectancy":     expectancy,
        "net_return_pct": float(net_ret_pct),
    }
