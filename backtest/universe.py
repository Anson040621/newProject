"""Point-in-time stock universe.

StockCharts' Large-Cap SCTR universe is "US stocks with a market cap over $10
billion". Free historical market-cap data does not exist, so we approximate it
with the S&P 500 *as it was on each date* (from github.com/fja05680/sp500).
Using historical membership instead of today's list avoids survivorship bias:
we only rank stocks that were actually in the index at the time.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MEMBERSHIP_FILE = DATA_DIR / "sp500_membership.csv"
MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)


def download_membership(path: Path = MEMBERSHIP_FILE, url: str = MEMBERSHIP_URL) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp:
        path.write_bytes(resp.read())
    return path


def load_membership(path: Path = MEMBERSHIP_FILE) -> pd.Series:
    """Series indexed by change date; each value is the set of member tickers from that date on."""
    if not path.exists():
        download_membership(path)
    raw = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    return pd.Series(
        [frozenset(t.strip() for t in tickers.split(",") if t.strip()) for tickers in raw["tickers"]],
        index=pd.DatetimeIndex(raw["date"]),
        name="tickers",
    )


def members_on(membership: pd.Series, date) -> frozenset[str]:
    """Index members on `date` (the most recent membership list on or before it)."""
    date = pd.Timestamp(date)
    eligible = membership.loc[:date]
    if eligible.empty:
        raise ValueError(f"No membership data on or before {date.date()}")
    return eligible.iloc[-1]


def tickers_between(membership: pd.Series, start, end) -> list[str]:
    """Every ticker that was a member at any time between `start` and `end`."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    snapshots = [members_on(membership, start)] + list(membership.loc[start:end])
    return sorted(set().union(*snapshots))


def membership_mask(membership: pd.Series, dates: pd.DatetimeIndex, tickers) -> pd.DataFrame:
    """Boolean DataFrame (dates x tickers): True where the ticker was a member that day."""
    tickers = list(tickers)
    changes = pd.DataFrame(
        [[t in members for t in tickers] for members in membership],
        index=membership.index,
        columns=tickers,
    )
    return changes.reindex(dates, method="ffill").fillna(False).astype(bool)
