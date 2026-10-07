"""Run the SCTR top-10 strategy backtest and compare it with the S&P 500 (SPY).

    python -m backtest.run_backtest --start 2015-01-01

Rules are described in backtest/strategy.py; every number can be changed with
the options below. Writes the trade log and the daily equity curve as CSV.
"""

from __future__ import annotations

import argparse

import pandas as pd

from . import sctr, signals, strategy
from .pick_stocks import build_inputs
from .prices import load_ohlc
from .universe import DATA_DIR



def load_etf(symbol: str, refresh: bool = False) -> pd.DataFrame:
    """Daily open and close of an ETF, adjusted for dividends (total return), downloaded once."""
    path = DATA_DIR / "benchmark" / f"{symbol}.csv"
    if not refresh and path.exists():
        data = pd.read_csv(path, index_col=0, parse_dates=True)
    if refresh or not path.exists() or "Open" not in data.columns:
        import yfinance as yf

        data = yf.download(symbol, start="2010-01-01", auto_adjust=False, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        path.parent.mkdir(parents=True, exist_ok=True)
        data[["Open", "Close", "Adj Close"]].to_csv(path)
    factor = data["Adj Close"] / data["Close"]
    return pd.DataFrame({"open": data["Open"] * factor, "close": data["Adj Close"]})


def load_benchmark(start, symbol: str = "SPY") -> pd.Series:
    """Total-return closes of the benchmark ETF from `start`."""
    return load_etf(symbol)["close"].loc[start:]


def prepare(start, end, top_n: int = 10, require_squeeze: bool = True):
    """Daily top N, the trading calendar, OHLC prices and signals for every ticker that ever made the top N."""
    inputs = build_inputs(start, end, "largecap")
    top = sctr.daily_top_n(inputs.closes, inputs.mask, n=top_n)
    top = top[top["date"] >= pd.Timestamp(start)]
    dates = inputs.closes.loc[start:end].index

    tickers = sorted(top["ticker"].unique())
    ohlc = load_ohlc(tickers)
    prices, sigs = {}, {}
    for ticker, frame in ohlc.items():
        sigs[ticker] = signals.ticker_signals(frame["high"], frame["low"], frame["close"], require_squeeze,
                                              open_=frame["open"]).reindex(dates)
        prices[ticker] = frame.reindex(dates)
    for frame in sigs.values():
        for col in ("setup", "entry_setup", "setup_dim_green", "atr_buy", "atr_exit"):
            frame[col] = frame[col].fillna(False).astype(bool)
    return dates, prices, sigs, top


def yearly_returns(equity: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    b = benchmark.reindex(equity.index).ffill()
    by_year = pd.DataFrame({"strategy": equity, "SPY": b}).groupby(equity.index.year)
    # Each year runs from the previous year's last close (the first year from its first close).
    base = by_year.last().shift(1).fillna(by_year.first())
    return ((by_year.last() / base - 1) * 100).round(1)


def main(argv=None):
    defaults = strategy.Rules()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--capital", type=float, default=defaults.capital)
    parser.add_argument("--size", type=float, default=defaults.position_size, help="fraction of equity per trade")
    parser.add_argument("--max-positions", type=int, default=defaults.max_positions)
    parser.add_argument("--top", type=int, default=defaults.top_n, help="only stocks in the daily top N")
    parser.add_argument("--min-sctr", type=float, default=defaults.min_sctr)
    parser.add_argument("--take-profit", type=float, default=defaults.take_profit, help="e.g. 0.08 = +8%%")
    parser.add_argument("--ema-exit-days", type=int, default=defaults.ema_exit_days,
                        help="consecutive closes below the 20 EMA before selling the rest (default 3)")
    parser.add_argument("--no-squeeze", action="store_true",
                        help="drop the squeeze (blue cross) entry condition")
    parser.add_argument("--atr-exit-on-touch", action=argparse.BooleanOptionalAction, default=defaults.atr_exit_on_touch,
                        help="before +8%%: sell when the price touches the ATR stop (default on); "
                             "--no-atr-exit-on-touch = version G (sell at the next open after a close below the line)")
    parser.add_argument("--close-exit", action=argparse.BooleanOptionalAction, default=defaults.close_exit,
                        help="with the intraday stop: also sell at the next open after a close below the line (default off)")
    parser.add_argument("--sticky-stop", action=argparse.BooleanOptionalAction, default=defaults.sticky_stop,
                        help="keep the last stop when the green line disappears, never lower it (default on)")
    parser.add_argument("--breakeven-exit", action=argparse.BooleanOptionalAction, default=defaults.breakeven_exit,
                        help="after +8%%: sell the rest at the next open after a close at or below the entry (default on)")
    parser.add_argument("--failed-breakout-exit", action=argparse.BooleanOptionalAction,
                        default=defaults.failed_breakout_exit,
                        help="sell at the next open if the entry day closes back below the flip level (default off)")
    parser.add_argument("--rest-exit", choices=["sma50", "ema20"], default=defaults.rest_exit,
                        help="after +8%%: sma50 = close more than --sma-exit-buffer below the 50 SMA (default); "
                             "ema20 = --ema-exit-days closes in a row below the 20 EMA")
    parser.add_argument("--sma-exit-buffer", type=float, default=defaults.sma_exit_buffer,
                        help="with --rest-exit sma50: how far below the 50 SMA the close must be (default 0.03)")
    parser.add_argument("--entry-buffer", type=float, default=defaults.entry_buffer,
                        help="buy-stop this far above the ATR flip level, e.g. 0.02 = 2%% (default)")
    parser.add_argument("--atr-stop-buffer", type=float, default=defaults.atr_stop_buffer,
                        help="put the intraday stop this far below the ATR line (default 0.02 = 2%%)")
    parser.add_argument("--rebuy-shakeouts", action="store_true",
                        help="after an intraday ATR stop, buy back at the next open if the close is still in an up-trend")
    parser.add_argument("--entry-on-touch", action=argparse.BooleanOptionalAction, default=defaults.entry_on_touch,
                        help="buy intraday when the price touches the ATR flip level (default on); "
                             "--no-entry-on-touch buys at the next open after a close-based flip")
    parser.add_argument("--emergency-stop", type=float, default=defaults.emergency_stop,
                        help="hard stop: sell everything at this loss from entry, e.g. 0.12 = -12%% (default; 0 = off)")
    parser.add_argument("--reentry-days", type=int, default=defaults.reentry_days,
                        help="re-buy if the ATR flips back to BUY within N days of an early exit (default 5, 0 = off)")
    parser.add_argument("--entry-check", choices=["prev", "live", "either"], default=defaults.entry_check,
                        help="prev: buy-stop only after a setup at the previous close; live: buy-stop for every top-10 "
                             "stock in a down-trend, bought where the squeeze and red bar show as the price rises "
                             "through the line; either: both")
    parser.add_argument("--breakeven-buffer", type=float, default=defaults.breakeven_buffer,
                        help="after +8%%: the break-even exit needs a close this far below the entry (default 0.03)")
    parser.add_argument("--close-entry", action=argparse.BooleanOptionalAction, default=defaults.close_entry,
                        help="also buy at the next open when the squeeze, red bar and ATR flip to BUY all show on "
                             "the same bar (the buy-stop needs the setup at the previous close) (default off)")
    parser.add_argument("--dim-green-days", type=int, default=defaults.dim_green_days,
                        help="for N trading days after a full-setup entry (e.g. 42 = about 2 months), after we have "
                             "exited, also buy the same setup with a dim green momentum bar (default 0 = off)")
    parser.add_argument("--park", default=None, metavar="ETF",
                        help="keep all money not in a trade in this ETF, e.g. QQQ (default: cash)")
    parser.add_argument("--park-cost", type=float, default=defaults.park_cost,
                        help="cost per move in/out of the parking ETF (default 0.0005 = 0.05%%)")
    parser.add_argument("--cost", type=float, default=defaults.cost, help="cost per buy/sell, e.g. 0.001 = 0.1%%")
    parser.add_argument("--trades-out", default="backtest_trades.csv")
    parser.add_argument("--equity-out", default="backtest_equity.csv")
    args = parser.parse_args(argv)

    rules = strategy.Rules(
        top_n=args.top, min_sctr=args.min_sctr, position_size=args.size, max_positions=args.max_positions,
        take_profit=args.take_profit, cost=args.cost, capital=args.capital, ema_exit_days=args.ema_exit_days,
        atr_exit_on_touch=args.atr_exit_on_touch, entry_on_touch=args.entry_on_touch,
        emergency_stop=args.emergency_stop, reentry_days=args.reentry_days,
        atr_stop_buffer=args.atr_stop_buffer, rebuy_shakeouts=args.rebuy_shakeouts,
        close_exit=args.close_exit, breakeven_exit=args.breakeven_exit,
        rest_exit=args.rest_exit, entry_check=args.entry_check, breakeven_buffer=args.breakeven_buffer, close_entry=args.close_entry, dim_green_days=args.dim_green_days, failed_breakout_exit=args.failed_breakout_exit, sma_exit_buffer=args.sma_exit_buffer, sticky_stop=args.sticky_stop, entry_buffer=args.entry_buffer, park_cost=args.park_cost,
    )
    dates, prices, sigs, top = prepare(args.start, args.end, args.top, not args.no_squeeze)
    park = load_etf(args.park).reindex(dates).ffill() if args.park else None
    equity, trades = strategy.run(dates, prices, sigs, top, rules, park=park)
    spy = load_benchmark(args.start)

    stats = strategy.summary(equity, trades, spy)
    print("\n".join(f"{k:>28}: {v:,.2f}" if isinstance(v, float) else f"{k:>28}: {v}" for k, v in stats.items()))
    print("\nReturn by year (%):")
    print(yearly_returns(equity, spy).to_string())
    trades.to_csv(args.trades_out, index=False, float_format="%.4f")
    equity.rename("equity").to_csv(args.equity_out, float_format="%.2f")
    print(f"\nTrade log -> {args.trades_out}, equity curve -> {args.equity_out}")


if __name__ == "__main__":
    main()
