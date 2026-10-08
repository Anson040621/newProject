"""Entry/exit indicators, ported line by line from the TradingView scripts.

Squeeze + momentum (from the "SEPA" strategy script):
    Bollinger Bands  SMA(close, 20) +/- 1.5 x stdev(close, 20)
                     (the script uses the KC multiplier for the bands, so we do too)
    Keltner Channel  SMA(close, 20) +/- 1.5 x SMA(true range, 20)
    squeeze on       BB inside KC  -> the blue crosses / "blue triangle accumulating"
    momentum         linreg(close - avg(avg(highest high 20, lowest low 20), SMA close 20), 20)
                     red bars = momentum below zero (bright: falling, dim: rising)

2x ATR Trailing Exit indicator:
    ATR(14) (Wilder), trails at hl2 -/+ 1.2 x ATR, trend flips when the close
    crosses the opposite trail; BUY = flip to up-trend, exit = flip to down-trend.

All functions work on one ticker: pandas Series of high, low, close (no gaps).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

SQUEEZE_LENGTH = 20
SQUEEZE_MULT = 1.5
ATR_LENGTH = 14
ATR_MULT = 1.2
EMA_LENGTH = 20


def true_range(high: pd.Series, low: pd.Series, close: pd.Series, first_bar_range: bool = False) -> pd.Series:
    """TradingView ta.tr: NaN on the first bar, or high - low there with first_bar_range (ta.tr(true))."""
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1, skipna=False)
    tr.iloc[0] = (high - low).iloc[0] if first_bar_range else np.nan
    return tr


def rma(values: pd.Series, length: int) -> pd.Series:
    """TradingView ta.rma (Wilder's average): starts with an SMA, then alpha = 1/length."""
    x = values.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    valid = np.flatnonzero(~np.isnan(x))
    if len(valid) < length:
        return pd.Series(out, index=values.index)
    start = valid[0] + length - 1
    out[start] = x[valid[0] : start + 1].mean()
    for i in range(start + 1, len(x)):
        out[i] = out[i - 1] + (x[i] - out[i - 1]) / length
    return pd.Series(out, index=values.index)


def linreg(values: pd.Series, length: int) -> pd.Series:
    """TradingView ta.linreg(src, length, 0): the regression line's value on the latest bar."""
    x = np.arange(length, dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) >= length:
        windows = sliding_window_view(values.to_numpy(dtype=float), length)
        slope = ((windows - windows.mean(axis=1, keepdims=True)) * (x - x.mean())).sum(axis=1) / ((x - x.mean()) ** 2).sum()
        out[length - 1 :] = windows.mean(axis=1) + slope * (length - 1 - x.mean())
    return pd.Series(out, index=values.index)


def squeeze(high: pd.Series, low: pd.Series, close: pd.Series,
            length: int = SQUEEZE_LENGTH, mult: float = SQUEEZE_MULT) -> pd.DataFrame:
    """Squeeze state, consecutive squeeze bars (blue crosses) and the momentum value."""
    basis = close.rolling(length).mean()
    dev = mult * close.rolling(length).std(ddof=0)  # ta.stdev is the population stdev
    range_ma = true_range(high, low, close).rolling(length).mean()
    squeeze_on = (basis - dev > basis - range_ma * mult) & (basis + dev < basis + range_ma * mult)

    blue_count = squeeze_on.astype(int).groupby((~squeeze_on).cumsum()).cumsum()

    mid = ((high.rolling(length).max() + low.rolling(length).min()) / 2 + basis) / 2
    momentum = linreg(close - mid, length)
    return pd.DataFrame({"squeeze_on": squeeze_on, "blue_count": blue_count, "momentum": momentum})


def atr_trail(high: pd.Series, low: pd.Series, close: pd.Series,
              length: int = ATR_LENGTH, mult: float = ATR_MULT) -> pd.DataFrame:
    """The "2x ATR Trailing Exit" indicator: trend (+1/-1/0), BUY and exit signals, active trail."""
    atr = rma(true_range(high, low, close, first_bar_range=True), length).to_numpy()
    hl2 = ((high + low) / 2).to_numpy()
    c = close.to_numpy(dtype=float)
    long_stop, short_stop = hl2 - mult * atr, hl2 + mult * atr

    n = len(c)
    trend = np.zeros(n, dtype=int)
    trail = np.full(n, np.nan)
    long_line = np.full(n, np.nan)  # the green line this bar has, or would have if the trend turned up here
    long_trail = short_trail = np.nan  # values at the end of the previous bar
    prev_trend = 0
    for i in range(n):
        lt = long_stop[i] if np.isnan(long_trail) else (long_stop[i] if long_stop[i] > long_trail else long_trail)
        st = short_stop[i] if np.isnan(short_trail) else (short_stop[i] if short_stop[i] < short_trail else short_trail)
        long_line[i] = lt
        if c[i] > short_trail:
            t = 1
        elif c[i] < long_trail:
            t = -1
        else:
            t = prev_trend
        long_trail = lt if t == 1 else long_stop[i]
        short_trail = st if t == -1 else short_stop[i]
        trend[i] = t
        trail[i] = long_trail if t == 1 else short_trail
        prev_trend = t

    previous = np.r_[0, trend[:-1]]
    return pd.DataFrame(
        {
            "trend": trend,
            "atr_buy": (trend == 1) & (previous != 1),
            "atr_exit": (trend == -1) & (previous == 1),
            "trail": trail,
            "long_line": long_line,
        },
        index=close.index,
    )


def at_touch(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, level: pd.Series,
             require_squeeze: bool = True, length: int = SQUEEZE_LENGTH, steps: int = 40) -> pd.DataFrame:
    """The live squeeze and momentum while the price rises through the ATR flip line during the day.

    For each day whose high reaches `level`, the price is walked up from the line
    (or the open, if it gaps above) to the day's high. At each price the day's bar
    so far is: open, high = last price = that price, low = the open; the previous
    19 bars are final - this is what the indicator panel shows live.

    touch_squeeze, touch_momentum: the readings the moment the price hits the line
    touch_price: the first price on the way up at which the squeeze is on (unless
                 require_squeeze=False) and the momentum bar is red; NaN if never
    """
    o, lv = open_.to_numpy(dtype=float), level.to_numpy(dtype=float)
    c, h, l = close.to_numpy(dtype=float), high.to_numpy(dtype=float), low.to_numpy(dtype=float)
    tr = true_range(high, low, close).to_numpy()
    mid = ((high.rolling(length).max() + low.rolling(length).min()) / 2 + close.rolling(length).mean()) / 2
    x = (close - mid).to_numpy()
    t = np.arange(length, dtype=float) - (length - 1) / 2

    n = len(c)
    squeeze_at_line, momentum_at_line, price = np.zeros(n, dtype=bool), np.full(n, np.nan), np.full(n, np.nan)
    for i in np.flatnonzero(h >= lv):  # NaN levels compare False
        if i < 2 * length or np.isnan(o[i]) or np.isnan(x[i - length + 1 : i]).any():
            continue
        first = max(lv[i], o[i])
        prices = np.linspace(first, max(first, h[i]), steps + 1)  # the walk up, [0] = at the line
        closes = np.hstack([np.tile(c[i - length + 1 : i], (len(prices), 1)), prices[:, None]])
        bar_tr = np.maximum.reduce([prices - o[i], np.abs(prices - c[i - 1]), np.full(len(prices), abs(o[i] - c[i - 1]))])
        ranges = (tr[i - length + 1 : i].sum() + bar_tr) / length
        # Bollinger inside Keltner (same multiplier on both): stdev < average true range.
        squeeze_on = closes.std(axis=1) < ranges
        top = np.maximum(h[i - length + 1 : i].max(), prices)
        bottom = min(l[i - length + 1 : i].min(), o[i])
        xs = np.hstack([np.tile(x[i - length + 1 : i], (len(prices), 1)),
                        (prices - ((top + bottom) / 2 + closes.mean(axis=1)) / 2)[:, None]])
        momentum = xs.mean(axis=1) + (xs - xs.mean(axis=1, keepdims=True)) @ t / (t @ t) * t[-1]
        ok = momentum < 0
        if require_squeeze:
            ok &= squeeze_on
        squeeze_at_line[i], momentum_at_line[i] = squeeze_on[0], momentum[0]
        if ok.any():
            price[i] = prices[np.argmax(ok)]
    return pd.DataFrame({"touch_squeeze": squeeze_at_line, "touch_momentum": momentum_at_line, "touch_price": price},
                        index=close.index)


def ema(close: pd.Series, length: int = EMA_LENGTH) -> pd.Series:
    """TradingView ta.ema (seeded with the first value)."""
    return close.ewm(span=length, adjust=False).mean()


def ticker_signals(high: pd.Series, low: pd.Series, close: pd.Series, require_squeeze: bool = True,
                   open_: pd.Series | None = None) -> pd.DataFrame:
    """Everything the strategy needs for one ticker, one row per trading day.

    entry_setup: squeeze on (blue crosses accumulating) AND momentum bar red
                 AND the ATR trail flips to BUY, all on the same bar. With
                 require_squeeze=False the squeeze condition is dropped.
    setup_dim_green: the same, but with a dim green momentum bar instead of red.
    touch_price: (with open_) the first price during the day, on the way up through
                 the flip level, at which the squeeze is on and the bar is red (at_touch).
    """
    sq = squeeze(high, low, close)
    trail = atr_trail(high, low, close)
    out = sq.join(trail)
    out["ema20"] = ema(close)
    out["sma50"] = close.rolling(50).mean()
    # The up-trend trail line as of the previous close: a price touching it during
    # the day is where the trend would flip (used for an intraday stop).
    out["atr_stop"] = out["trail"].where(out["trend"] == 1).shift(1)
    # The down-trend trail line as of the previous close: a price touching it
    # during the day is where the trend would flip to BUY (intraday entry).
    out["flip_level"] = out["trail"].where(out["trend"] != 1).shift(1)
    out["setup"] = out["momentum"] < 0  # red momentum bar (and the squeeze, below)
    if require_squeeze:
        out["setup"] &= out["squeeze_on"]
    out["entry_setup"] = out["setup"] & out["atr_buy"]
    # Dim green momentum bar: above zero but not rising (the script's dimGreen color).
    # Used for the optional follow-up entry (Rules.dim_green_days).
    out["setup_dim_green"] = (out["momentum"] > 0) & (out["momentum"] <= out["momentum"].shift(1))
    if require_squeeze:
        out["setup_dim_green"] &= out["squeeze_on"]
    if open_ is not None:
        out = out.join(at_touch(open_, high, low, close, out["flip_level"], require_squeeze))
    return out
