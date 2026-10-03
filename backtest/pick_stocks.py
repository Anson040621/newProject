"""Step 1 - stock selection: top N large-cap US stocks by SCTR.

Show the top 10 (with the six SCTR components) on one date:
    python -m backtest.pick_stocks --date 2025-09-30

Save the daily top 10 for a whole period (input for the backtest):
    python -m backtest.pick_stocks --start 2015-01-01 --end 2025-09-30 --out top10_daily.csv

Universes (--universe):
  largecap  US stocks worth over $10B, StockCharts-style (default)
  sp500     the S&P 500 as it was on each date
  all       every ticker in data/prices/ (e.g. your own CSVs)
"""

from __future__ import annotations

import argparse

import pandas as pd

from . import sctr
from .listings import load_listings
from .marketcap import large_cap_universe
from .prices import load_prices
from .sec import load_shares
from .universe import load_membership, membership_mask, tickers_between

WARMUP = pd.DateOffset(years=1, months=6)


def build_inputs(start, end, universe: str):
    """Closing prices for the indicators, the eligibility mask, and market caps (largecap only).

    Only tickers that are in the universe at some point between `start` and
    `end` are kept, to keep the SCTR calculation fast.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end) if end else None
    mcap = None
    if universe == "all":
        closes = load_prices(start=start - WARMUP, end=end)["adj_close"]
        return closes, None, None
    membership = load_membership()
    if universe == "sp500":
        prices = load_prices(tickers_between(membership, start, end or pd.Timestamp.today()), end=end)
        mask = membership_mask(membership, prices["adj_close"].index, prices["adj_close"].columns)
    else:
        prices = load_prices(end=end)
        mcap, mask = large_cap_universe(prices["close"], prices["splits"], load_shares(), load_listings(), membership)

    closes = prices["adj_close"]
    in_period = mask.loc[start:end].any()
    keep = in_period.index[in_period]
    closes, mask = closes[keep].loc[start - WARMUP :], mask[keep].loc[start - WARMUP :]
    if mcap is not None:
        mcap = mcap[keep].loc[start - WARMUP :]
    return closes, mask, mcap


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", help="show the ranking on this date (uses the last trading day on or before it)")
    parser.add_argument("--start", help="first date of a daily top-N history")
    parser.add_argument("--end", help="last date (default: latest data)")
    parser.add_argument("--top", type=int, default=10, help="how many stocks to pick (default 10)")
    parser.add_argument("--universe", choices=["largecap", "sp500", "all"], default="largecap")
    parser.add_argument("--out", help="CSV file to write the daily top-N history to")
    args = parser.parse_args(argv)

    if bool(args.date) == bool(args.start):
        parser.error("give either --date or --start")

    if args.date:
        closes, mask, mcap = build_inputs(args.date, args.date, args.universe)
        day = closes.index[closes.index <= pd.Timestamp(args.date)][-1]
        table = sctr.sctr_table(closes, day, mask)
        if mcap is not None:
            table.insert(2, "mcap_$B", mcap.loc[day, table.index] / 1e9)
        print(f"SCTR ranking on {day.date()} ({len(table)} ranked stocks). Top {args.top}:\n")
        with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 140):
            print(table.head(args.top).to_string())
        return

    closes, mask, _ = build_inputs(args.start, args.end, args.universe)
    history = sctr.daily_top_n(closes, mask, n=args.top)
    history = history[history["date"] >= pd.Timestamp(args.start)]
    out = args.out or f"top{args.top}_daily.csv"
    history.to_csv(out, index=False, float_format="%.4f")
    days = history["date"].nunique()
    print(f"Wrote top {args.top} for {days} trading days to {out}")


if __name__ == "__main__":
    main()
