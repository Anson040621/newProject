"""StockCharts Technical Rank (SCTR) calculation.

Implements the published StockCharts methodology:
https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/stockcharts-technical-rank-sctr

Indicator score (per stock, per day):
    Long-term   30%  percent above/below 200-day EMA
                30%  125-day rate of change
    Medium-term 15%  percent above/below 50-day EMA
                15%  20-day rate of change
    Short-term   5%  3-day slope of PPO(12,26,9) histogram (scored 0..5 points)
                 5%  14-day RSI

The scores are then ranked within the universe on each day and mapped to 0.0-99.9.

All functions take a "wide" DataFrame of closing prices: one column per ticker,
one row per trading day (DatetimeIndex, ascending).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# StockCharts needs roughly a year of history before a stock gets an SCTR.
MIN_HISTORY_DAYS = 250

WEIGHTS = {
    "pct_ema200": 0.30,
    "roc125": 0.30,
    "pct_ema50": 0.15,
    "roc20": 0.15,
    "rsi14": 0.05,
}


def ema(close: pd.DataFrame, span: int) -> pd.DataFrame:
    """Exponential moving average, NaN until `span` bars are available."""
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def roc(close: pd.DataFrame, period: int) -> pd.DataFrame:
    """Rate of change in percent."""
    return (close / close.shift(period) - 1.0) * 100.0


def pct_from_ema(close: pd.DataFrame, span: int) -> pd.DataFrame:
    """Percent above (+) or below (-) the EMA."""
    return (close / ema(close, span) - 1.0) * 100.0


def rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    # No losses in the window -> RSI is 100 (rs is inf, which the formula already
    # handles); no movement at all -> treat as neutral 50.
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return out


def ppo_histogram(close: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """PPO histogram: PPO(fast, slow) minus its `signal`-period EMA."""
    ema_slow = ema(close, slow)
    ppo = (ema(close, fast) - ema_slow) / ema_slow * 100.0
    ppo_signal = ppo.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return ppo - ppo_signal


def ppo_slope_points(close: pd.DataFrame) -> pd.DataFrame:
    """Score (0..5 points) for the 3-day slope of the PPO histogram.

    slope = (hist_today - hist_3_days_ago) / 3
    slope > +1  -> 5 points
    slope < -1  -> 0 points
    otherwise   -> 5% of (slope + 1) * 50
    """
    hist = ppo_histogram(close)
    slope = (hist - hist.shift(3)) / 3.0
    return 0.05 * ((slope.clip(-1.0, 1.0) + 1.0) * 50.0)


def components(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Raw (unweighted) values of the six SCTR inputs, keyed by name.

    `ppo_points` is already in points (0..5); the others are raw values.
    """
    return {
        "pct_ema200": pct_from_ema(close, 200),
        "roc125": roc(close, 125),
        "pct_ema50": pct_from_ema(close, 50),
        "roc20": roc(close, 20),
        "ppo_points": ppo_slope_points(close),
        "rsi14": rsi(close, 14),
    }


def indicator_score(close: pd.DataFrame, min_history: int = MIN_HISTORY_DAYS) -> pd.DataFrame:
    """Weighted SCTR indicator score (before ranking).

    A ticker gets a score only once it has `min_history` valid closes.
    """
    comps = components(close)
    score = comps["ppo_points"].copy()
    for name, weight in WEIGHTS.items():
        score = score + weight * comps[name]
    history = close.notna().cumsum()
    return score.where(history >= min_history)


def rank_scores(score: pd.DataFrame, universe_mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rank indicator scores across tickers on each day and map to 0.0..99.9.

    The weakest stock in the universe that day gets 0.0, the strongest 99.9, and
    the rest are spread evenly in between (percentile buckets, like StockCharts).

    `universe_mask` (same shape as `score`, booleans) limits which tickers are
    eligible on each day, e.g. point-in-time index membership.
    """
    if universe_mask is not None:
        mask = universe_mask.reindex(index=score.index, columns=score.columns).fillna(False).astype(bool)
        score = score.where(mask)
    ranks = score.rank(axis=1, method="average")
    count = score.notna().sum(axis=1)
    sctr = (ranks - 1.0).div((count - 1).clip(lower=1), axis=0) * 99.9
    # StockCharts truncates to one decimal (99.79 shows as 99.7).
    return np.floor(sctr * 10 + 1e-9) / 10


def sctr_table(close: pd.DataFrame, date, universe_mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full SCTR breakdown for every eligible ticker on one date, best first."""
    score = indicator_score(close)
    sctr = rank_scores(score, universe_mask)
    comps = components(close)
    table = pd.DataFrame({"sctr": sctr.loc[date], "score": score.loc[date]})
    for name, frame in comps.items():
        table[name] = frame.loc[date]
    table = table.dropna(subset=["sctr"]).sort_values("score", ascending=False)
    table.index.name = "ticker"
    return table


def daily_top_n(close: pd.DataFrame, universe_mask: pd.DataFrame | None = None, n: int = 10) -> pd.DataFrame:
    """Long-format table of the top `n` tickers for every day.

    Columns: date, rank (1 = best), ticker, sctr, score.
    """
    score = indicator_score(close)
    sctr = rank_scores(score, universe_mask)
    # SCTR is the within-day rank of the score, so the raw score gives the same
    # order without the rounding ties.
    ranked = score.where(sctr.notna())
    rows = []
    for date, row in ranked.iterrows():
        best = row.dropna().nlargest(n)
        for rank, ticker in enumerate(best.index, start=1):
            rows.append((date, rank, ticker, sctr.at[date, ticker], best[ticker]))
    return pd.DataFrame(rows, columns=["date", "rank", "ticker", "sctr", "score"])
