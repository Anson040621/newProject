"""Step 1 - stock selection: top N large-cap US stocks by SCTR.

Show the top 10 (with the six SCTR components) on one date:
    python -m backtest.pick_stocks --date 2025-09-30

Save the daily top 10 for a whole period (input for the backtest):
    python -m backtest.pick_stocks --start 2015-01-01 --end 2025-09-30 --out top10_daily.csv

By default the universe is the S&P 500 as it was on each date. Use
--universe all to rank every ticker in data/prices/ instead (e.g. your own CSVs).
"""

from __future__ import annotations

import argparse

import pandas as pd

from . import sctr
from .prices import load_closes
from .universe import load_membership, membership_mask, tickers_between


def build_inputs(start, end, universe: str):
    """Closing prices (with warm-up history) and the eligibility mask."""
    warmup_start = pd.Timestamp(start) - pd.DateOffset(years=1, months=6)
    if universe == "all":
        closes = load_closes(start=warmup_start, end=end)
        return closes, None
    membership = load_membership()
    closes = load_closes(tickers_between(membership, start, end), start=warmup_start, end=end)
    return closes, membership_mask(membership, closes.index, closes.columns)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", help="show the ranking on this date (uses the last trading day on or before it)")
    parser.add_argument("--start", help="first date of a daily top-N history")
    parser.add_argument("--end", help="last date (default: latest data)")
    parser.add_argument("--top", type=int, default=10, help="how many stocks to pick (default 10)")
    parser.add_argument("--universe", choices=["sp500", "all"], default="sp500")
    parser.add_argument("--out", help="CSV file to write the daily top-N history to")
    args = parser.parse_args(argv)

    if bool(args.date) == bool(args.start):
        parser.error("give either --date or --start")

    if args.date:
        closes, mask = build_inputs(args.date, args.date, args.universe)
        day = closes.index[closes.index <= pd.Timestamp(args.date)][-1]
        table = sctr.sctr_table(closes, day, mask)
        print(f"SCTR ranking on {day.date()} ({len(table)} ranked stocks). Top {args.top}:\n")
        with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 120):
            print(table.head(args.top).to_string())
        return

    closes, mask = build_inputs(args.start, args.end, args.universe)
    history = sctr.daily_top_n(closes, mask, n=args.top)
    history = history[history["date"] >= pd.Timestamp(args.start)]
    out = args.out or f"top{args.top}_daily.csv"
    history.to_csv(out, index=False, float_format="%.4f")
    days = history["date"].nunique()
    print(f"Wrote top {args.top} for {days} trading days to {out}")


if __name__ == "__main__":
    main()
