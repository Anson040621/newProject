"""Download everything the large-cap SCTR universe needs.

    SEC_USER_AGENT="your-app-name your@email.com" python -m backtest.download_data --start 2015-01-01

Steps:
  1. S&P 500 membership history (adds back large caps that have since been delisted)
  2. Current US listings with market caps (Nasdaq screener)
  3. Daily prices + split history (Yahoo Finance) for every stock that is worth
     at least --min-cap today or was in the S&P 500 since --start
  4. Share-count history from SEC filings (needs SEC_USER_AGENT, see README)

Prices start 18 months before --start so the 200-day EMA, 125-day ROC and the
3-month universe rule are warmed up by the time the backtest begins.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from .listings import LISTINGS_FILE, download_listings
from .prices import PRICES_DIR, download_prices
from .universe import MEMBERSHIP_FILE, download_membership, load_membership, tickers_between


def candidate_tickers(listings: pd.DataFrame, membership: pd.Series, start, end, min_cap: float) -> list[str]:
    """Stocks that could have been large caps since `start`.

    Today's listings above `min_cap` (a stock worth $1B today may have been worth
    $10B+ a few years ago), plus every S&P 500 member since `start`.
    """
    listed = listings.index[listings["market_cap"] >= min_cap]
    return sorted(set(listed) | set(tickers_between(membership, start, end)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2015-01-01", help="first date you want SCTR rankings for")
    parser.add_argument("--end", default=None, help="last date (default: today)")
    parser.add_argument("--min-cap", type=float, default=1e9, help="minimum market cap today to download (default 1e9)")
    parser.add_argument("--skip-prices", action="store_true", help="don't re-download prices")
    parser.add_argument("--skip-sec", action="store_true", help="don't download SEC share counts")
    args = parser.parse_args(argv)

    print(f"1. S&P 500 membership history -> {MEMBERSHIP_FILE}")
    download_membership()
    membership = load_membership()

    print(f"2. Nasdaq listings -> {LISTINGS_FILE}")
    listings = download_listings()
    print(f"   {len(listings)} listed stocks, {int((listings['market_cap'] > 10e9).sum())} worth over $10B")

    end = args.end or pd.Timestamp.today().normalize()
    tickers = candidate_tickers(listings, membership, args.start, end, args.min_cap)

    if not args.skip_prices:
        price_start = (pd.Timestamp(args.start) - pd.DateOffset(years=1, months=6)).date()
        print(f"3. Prices for {len(tickers)} tickers from {price_start} -> {PRICES_DIR}")
        missing = download_prices(tickers, start=price_start, end=args.end)
        print(f"   {len(tickers) - len(missing)} downloaded, {len(missing)} with no Yahoo data (mostly delisted):")
        print("   " + ", ".join(missing))

    if not args.skip_sec:
        user_agent = os.environ.get("SEC_USER_AGENT")
        if not user_agent:
            print("4. Skipping SEC share counts: set SEC_USER_AGENT (see README). "
                  "Market caps will use today's share counts instead.")
        else:
            from .sec import SHARES_FILE, download_shares

            print(f"4. SEC share counts for {len(tickers)} tickers -> {SHARES_FILE}")
            shares = download_shares(tickers, user_agent)
            print(f"   share history found for {shares['ticker'].nunique()} tickers")


if __name__ == "__main__":
    main()
