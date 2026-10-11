"""'Twin' setups: setups outside the top 10 that share every trait of the two big runs the
top-10 list caught late, META (setup on Mar 2, 2023, ranked 161st) and TER (Sep 17, 2025,
ranked 216th). Used by --twin-days (strategy.Rules.twin_days).

At the setup close (squeeze on, red momentum bar, ATR trail in a down-trend), a twin:
  - is a valid entry: the next day's high reaches the buy-stop (the flip level), as META's
    did on Mar 3 and TER's on Sep 18,
  - has an SCTR of 65 to 90 (so it is not on the top-10 list, which needs more than 90),
  - had an SCTR of 5 or less at some point in the last 126 trading days (about 6 months),
  - is 50+ SCTR points higher than 60 trading days ago,
  - opened at least 15% above the previous close at least once in the last 63 trading
    days (an earnings jump),
  - has a fresh golden cross: the 50-day SMA has been above the 200-day SMA for 1 to 20
    trading days,
  - closed 10% to 30% below its highest close of the last 252 trading days.
META: SCTR 77.7 (low 0.2, +71 in 60 days), gap +20%, golden cross 1 day old, 25% below
its high. TER: SCTR 74.6 (low 0.7, +67), gap +17%, 13 days, 18% below.
"""

from __future__ import annotations

import pandas as pd

SCTR_RANGE = (65.0, 90.0)
LOW_SCTR, LOW_DAYS = 5.0, 126
RISE, RISE_DAYS = 50.0, 60
GAP, GAP_DAYS = 0.15, 63
CROSS_DAYS = 20
OFF_HIGH, HIGH_DAYS = (0.10, 0.30), 252


def golden_cross_days(closes: pd.DataFrame) -> pd.DataFrame:
    """Trading days in a row, today included, that the 50-day SMA has closed above the 200-day SMA (0 = not above)."""
    above = closes.rolling(50).mean() > closes.rolling(200).mean()
    count = above.astype(int).cumsum()
    return count - count.where(~above).ffill().fillna(0)


def find_twins(closes: pd.DataFrame, opens: pd.DataFrame, highs: pd.DataFrame, raw_closes: pd.DataFrame,
               sctr_table: pd.DataFrame, setup: pd.DataFrame, trend: pd.DataFrame, flip_level: pd.DataFrame) -> pd.DataFrame:
    """Every twin setup close, as a table (date, ticker, sctr), oldest first.

    All inputs are wide frames (dates x tickers), aligned to `sctr_table`:
    closes:      adjusted closes, as used for the indicators (SMAs, 52-week high)
    opens, highs, raw_closes: the chart prices, for the gap up and the next day's fill
    sctr_table:  the 0-99.9 SCTR (NaN outside the universe)
    setup, trend, flip_level: signals.ticker_signals columns
    """
    index, columns = sctr_table.index, sctr_table.columns
    align = lambda frame: frame.reindex(index=index, columns=columns)
    sc, closes = sctr_table, align(closes)
    low = sc.rolling(LOW_DAYS, min_periods=60).min()
    rise = sc - sc.shift(RISE_DAYS)
    gap = (align(opens) / align(raw_closes).shift(1) - 1).rolling(GAP_DAYS, min_periods=20).max()
    cross = golden_cross_days(closes)
    off_high = 1 - closes / closes.rolling(HIGH_DAYS, min_periods=200).max()
    filled = align(highs).shift(-1) >= align(flip_level).shift(-1)  # the buy-stop is reached the next day
    hit = (align(setup).fillna(False).astype(bool) & (align(trend) != 1) & filled
           & sc.ge(SCTR_RANGE[0]) & sc.le(SCTR_RANGE[1]) & low.le(LOW_SCTR) & rise.ge(RISE) & gap.ge(GAP)
           & cross.ge(1) & cross.le(CROSS_DAYS) & off_high.ge(OFF_HIGH[0]) & off_high.le(OFF_HIGH[1]))
    days, tickers = hit.to_numpy().nonzero()
    table = pd.DataFrame({"date": index[days], "ticker": columns[tickers], "sctr": sc.to_numpy()[days, tickers]})
    return table.sort_values(["date", "ticker"], ignore_index=True)
