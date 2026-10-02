"""Download S&P 500 membership history and daily prices for every historical member.

    python -m backtest.download_data --start 2010-01-01

Prices start one year before --start so the 200-day EMA and 125-day ROC are
warmed up by the time the backtest begins.
"""

from __future__ import annotations

import argparse

import pandas as pd

from .prices import PRICES_DIR, download_prices
from .universe import MEMBERSHIP_FILE, download_membership, load_membership, tickers_between


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2010-01-01", help="first date you want SCTR rankings for")
    parser.add_argument("--end", default=None, help="last date (default: today)")
    args = parser.parse_args(argv)

    print(f"Downloading S&P 500 membership history -> {MEMBERSHIP_FILE}")
    download_membership()
    membership = load_membership()

    end = args.end or pd.Timestamp.today().normalize()
    tickers = tickers_between(membership, args.start, end)
    price_start = (pd.Timestamp(args.start) - pd.DateOffset(years=1, months=6)).date()
    print(f"Downloading prices for {len(tickers)} tickers from {price_start} -> {PRICES_DIR}")
    missing = download_prices(tickers, start=price_start, end=args.end)

    print(f"\nDone. {len(tickers) - len(missing)} tickers downloaded, {len(missing)} with no Yahoo data.")
    if missing:
        print("No data (mostly delisted or renamed companies Yahoo no longer carries):")
        print("  " + ", ".join(missing))


if __name__ == "__main__":
    main()
