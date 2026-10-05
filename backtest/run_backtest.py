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

BENCHMARK_FILE = DATA_DIR / "benchmark" / "SPY.csv"


def load_benchmark(start, refresh: bool = False) -> pd.Series:
    """SPY closes adjusted for dividends (total return), downloaded once."""
    if refresh or not BENCHMARK_FILE.exists():
        import yfinance as yf

        data = yf.download("SPY", start="2010-01-01", auto_adjust=False, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        BENCHMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
        data[["Close", "Adj Close"]].to_csv(BENCHMARK_FILE)
    spy = pd.read_csv(BENCHMARK_FILE, index_col=0, parse_dates=True)["Adj Close"]
    return spy.loc[start:]


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
        sigs[ticker] = signals.ticker_signals(frame["high"], frame["low"], frame["close"], require_squeeze).reindex(dates)
        prices[ticker] = frame.reindex(dates)
    for frame in sigs.values():
        for col in ("entry_setup", "atr_exit"):
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
                        help="consecutive closes below the 20 EMA before selling the rest (default 1)")
    parser.add_argument("--no-squeeze", action="store_true",
                        help="drop the squeeze (blue cross) entry condition")
    parser.add_argument("--cost", type=float, default=defaults.cost, help="cost per buy/sell, e.g. 0.001 = 0.1%%")
    parser.add_argument("--trades-out", default="backtest_trades.csv")
    parser.add_argument("--equity-out", default="backtest_equity.csv")
    args = parser.parse_args(argv)

    rules = strategy.Rules(
        top_n=args.top, min_sctr=args.min_sctr, position_size=args.size, max_positions=args.max_positions,
        take_profit=args.take_profit, cost=args.cost, capital=args.capital, ema_exit_days=args.ema_exit_days,
    )
    dates, prices, sigs, top = prepare(args.start, args.end, args.top, not args.no_squeeze)
    equity, trades = strategy.run(dates, prices, sigs, top, rules)
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
