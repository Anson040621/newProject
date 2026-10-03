"""Estimated daily market caps and the StockCharts-style large-cap universe.

StockCharts' Large-Cap SCTR universe is "US stocks with a market cap over $10
billion", rebalanced monthly, where a stock must qualify for three consecutive
months before it moves in or out.

Market cap on day t = split-adjusted close(t) x shares outstanding known on day t.

- Shares come from the most recent SEC filing *filed* on or before t, so the
  backtest never uses information from the future.
- Share counts are put on today's split basis (a 2023 NVIDIA filing is
  multiplied by 10 for the 2024 10-for-1 split) to match the split-adjusted
  prices from Yahoo.
- The whole history is then scaled so the latest value matches Nasdaq's market
  cap today. This fixes foreign ADRs (SEC reports ordinary shares, e.g. 5 per
  UMC ADR) and multi-class companies, while keeping the changes over time
  (buybacks, share issuance).
- With no SEC history, today's share count is used for every date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .universe import membership_mask

LARGE_CAP = 10e9
CONFIRM_MONTHS = 3
# A filing whose split-adjusted share count is 10x away from the company's
# median is treated as a data error (e.g. reported in thousands).
OUTLIER_FACTOR = 10.0


def split_factors_after(as_of: pd.Series, splits: pd.Series) -> np.ndarray:
    """For each as-of date, the product of all split ratios with an ex-date after it."""
    events = splits[(splits > 0) & (splits != 1)].sort_index()
    if events.empty:
        return np.ones(len(as_of))
    # suffix[i] = product of ratios from event i to the last event
    suffix = np.append(np.cumprod(events.to_numpy()[::-1])[::-1], 1.0)
    positions = np.searchsorted(events.index.to_numpy(), pd.DatetimeIndex(as_of).to_numpy(), side="right")
    return suffix[positions]


def shares_known_on(filings: pd.DataFrame, splits: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Split-adjusted shares outstanding as known on each date (point in time by filing date).

    Dates before the first filing use the first filing's value.
    """
    filings = filings.sort_values(["filed", "end"])
    adjusted = filings["shares"].to_numpy() * split_factors_after(filings["end"], splits)
    adjusted = pd.Series(adjusted, index=pd.DatetimeIndex(filings["filed"]))
    median = adjusted.median()
    adjusted = adjusted[(adjusted < median * OUTLIER_FACTOR) & (adjusted > median / OUTLIER_FACTOR)]
    known = adjusted.groupby(level=0).last()
    return known.reindex(index.union(known.index)).ffill().reindex(index).bfill()


def estimate_market_caps(
    close: pd.DataFrame, splits: pd.DataFrame, shares: pd.DataFrame, listings: pd.DataFrame
) -> pd.DataFrame:
    """Daily market cap estimates (dates x tickers); NaN where it can't be estimated.

    close:    split-adjusted closes (dates x tickers)
    splits:   split ratios on ex-dates (same shape as close)
    shares:   SEC share history, long format (ticker, end, filed, shares)
    listings: today's listings with market_cap and last_price, indexed by ticker
    """
    current_shares = (listings["market_cap"] / listings["last_price"]).dropna()
    filings_by_ticker = dict(tuple(shares.groupby("ticker"))) if len(shares) else {}
    caps = {}
    for ticker in close.columns:
        filings = filings_by_ticker.get(ticker)
        today = current_shares.get(ticker)
        if filings is not None and not filings.empty:
            history = shares_known_on(filings, splits[ticker], close.index)
            if today is not None and history.iloc[-1] > 0:
                history = history * (today / history.iloc[-1])
        elif today is not None:
            history = pd.Series(today, index=close.index)
        else:
            continue
        caps[ticker] = close[ticker] * history
    return pd.DataFrame(caps, index=close.index).reindex(columns=close.columns)


def large_cap_mask(mcap: pd.DataFrame, threshold: float = LARGE_CAP, confirm_months: int = CONFIRM_MONTHS) -> pd.DataFrame:
    """Large-cap universe membership (dates x tickers), rebalanced at each month end.

    A stock seen for the first time joins (or not) straight away based on its
    market cap. After that it switches only when it has been on the other side
    of `threshold` for `confirm_months` consecutive month ends. A stock with no
    price at a month end drops out and is re-classified when it comes back.
    The decision made at a month end applies from the next trading day.
    """
    month_ends = mcap.index.to_series().groupby(mcap.index.to_period("M")).max()
    state = np.full(mcap.shape[1], np.nan)  # 1 = large, 0 = not, NaN = unclassified
    streak = np.zeros(mcap.shape[1], dtype=int)
    decisions = []
    for day in month_ends:
        cap = mcap.loc[day].to_numpy(dtype=float)
        has_cap = ~np.isnan(cap)
        qualifies = (cap > threshold).astype(float)

        new = has_cap & np.isnan(state)
        state[new] = qualifies[new]
        streak[new] = 0

        differs = has_cap & ~new & (qualifies != state)
        streak = np.where(differs, streak + 1, 0)
        flip = streak >= confirm_months
        state[flip] = qualifies[flip]
        streak[flip] = 0

        state[~has_cap] = np.nan
        decisions.append(state == 1)

    decided = pd.DataFrame(np.array(decisions, dtype=float), index=pd.DatetimeIndex(month_ends), columns=mcap.columns)
    return decided.reindex(mcap.index).ffill().shift(1).fillna(0.0).astype(bool)


def large_cap_universe(
    close: pd.DataFrame,
    splits: pd.DataFrame,
    shares: pd.DataFrame,
    listings: pd.DataFrame,
    membership: pd.Series | None = None,
    threshold: float = LARGE_CAP,
    confirm_months: int = CONFIRM_MONTHS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimated market caps and the large-cap universe mask.

    Stocks with no market-cap information at all (delisted, no SEC record) are
    treated as large caps while they were in the S&P 500, if `membership` is given.
    """
    mcap = estimate_market_caps(close, splits, shares, listings)
    mask = large_cap_mask(mcap, threshold, confirm_months)
    if membership is not None:
        unknown = mcap.columns[mcap.isna().all()]
        if len(unknown):
            fallback = membership_mask(membership, mcap.index, unknown) & close[unknown].notna()
            mask.loc[:, unknown] = fallback.to_numpy()
    return mcap, mask
