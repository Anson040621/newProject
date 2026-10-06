"""Step 1 - stock selection: top N large-cap US stocks by SCTR.

Show the top 10 on one date, laid out like StockCharts' SCTR report
(add --details for the six indicator values behind each score):
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
from typing import NamedTuple

import pandas as pd

from . import sctr
from .listings import LISTINGS_FILE, load_listings, short_name
from .marketcap import large_cap_universe
from .prices import load_prices
from .sec import load_shares
from .universe import load_membership, membership_mask, tickers_between

WARMUP = pd.DateOffset(years=1, months=6)


class Inputs(NamedTuple):
    closes: pd.DataFrame  # adjusted for splits and dividends: used for the indicators
    prices: pd.DataFrame  # adjusted for splits only: the price as shown on a chart
    mask: pd.DataFrame | None  # which tickers are in the universe on each day
    mcap: pd.DataFrame | None  # estimated market caps (largecap universe only)


def build_inputs(start, end, universe: str) -> Inputs:
    """Prices, the universe mask and market caps for `start`..`end` plus warm-up history.

    Only tickers that are in the universe at some point between `start` and
    `end` are kept, to keep the SCTR calculation fast.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end) if end else None
    mcap = None
    if universe == "all":
        prices = load_prices(start=start - WARMUP, end=end)
        return Inputs(prices["adj_close"], prices["close"], None, None)
    membership = load_membership()
    if universe == "sp500":
        prices = load_prices(tickers_between(membership, start, end or pd.Timestamp.today()), end=end)
        mask = membership_mask(membership, prices["adj_close"].index, prices["adj_close"].columns)
    else:
        # Market caps are calibrated to today's share counts (Nasdaq), so they need the
        # full price, split and filing history even when only an earlier period is wanted.
        prices = load_prices()
        mcap, mask = large_cap_universe(prices["close"], prices["splits"], load_shares(), load_listings(), membership)

    in_period = mask.loc[start:end].any()
    keep = in_period.index[in_period]

    def trim(frame):
        return frame[keep].loc[start - WARMUP : end]

    return Inputs(trim(prices["adj_close"]), trim(prices["close"]), trim(mask), trim(mcap) if mcap is not None else None)


def ranking_table(inputs: Inputs, day, top: int, details: bool = False) -> pd.DataFrame:
    """The top stocks on `day`, StockCharts-style: name, sector, SCTR, CHG, close, market cap."""
    full = sctr.sctr_table(inputs.closes, day, inputs.mask)
    best = full.head(top)
    table = pd.DataFrame(index=best.index)
    if LISTINGS_FILE.exists():
        listings = load_listings().reindex(best.index)
        table["name"] = listings["name"].fillna("").map(short_name).str.slice(0, 30)
        table["sector"] = listings["sector"].fillna("")
    table["sctr"] = best["sctr"]
    table["chg"] = best["chg"]
    table["close"] = inputs.prices.loc[day, best.index]
    if inputs.mcap is not None:
        table["mcap_$B"] = inputs.mcap.loc[day, best.index] / 1e9
    if details:
        table = table.join(best.drop(columns=["sctr", "chg"]))
    table.insert(0, "#", range(1, len(table) + 1))
    table.attrs["ranked"] = len(full)
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", help="show the ranking on this date (uses the last trading day on or before it)")
    parser.add_argument("--start", help="first date of a daily top-N history")
    parser.add_argument("--end", help="last date (default: latest data)")
    parser.add_argument("--top", type=int, default=10, help="how many stocks to pick (default 10)")
    parser.add_argument("--universe", choices=["largecap", "sp500", "all"], default="largecap")
    parser.add_argument("--out", help="CSV file to write the daily top-N history to")
    parser.add_argument("--details", action="store_true", help="with --date: also show the six indicator values")
    args = parser.parse_args(argv)

    if bool(args.date) == bool(args.start):
        parser.error("give either --date or --start")

    if args.date:
        inputs = build_inputs(args.date, args.date, args.universe)
        day = inputs.closes.index[inputs.closes.index <= pd.Timestamp(args.date)][-1]
        table = ranking_table(inputs, day, args.top, args.details)
        print(f"SCTR ranking on {day.date()} ({table.attrs['ranked']} ranked stocks). Top {args.top}:\n")
        formats = {
            "name": "{:<30}".format, "sector": "{:<22}".format, "sctr": "{:.1f}".format,
            "chg": "{:+.1f}".format, "close": "{:,.2f}".format, "mcap_$B": "{:,.1f}".format,
        }
        with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 200):
            print(table.to_string(index_names=False, formatters=formats, justify="left"))
        return

    inputs = build_inputs(args.start, args.end, args.universe)
    history = sctr.daily_top_n(inputs.closes, inputs.mask, n=args.top)
    history = history[history["date"] >= pd.Timestamp(args.start)]
    out = args.out or f"top{args.top}_daily.csv"
    history.to_csv(out, index=False, float_format="%.4f")
    days = history["date"].nunique()
    print(f"Wrote top {args.top} for {days} trading days to {out}")


if __name__ == "__main__":
    main()
