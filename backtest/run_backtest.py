"""Run the SCTR top-10 strategy backtest and compare it with the S&P 500 (SPY).

    python -m backtest.run_backtest --start 2015-01-01

Rules are described in backtest/strategy.py; every number can be changed with
the options below. Writes the trade log and the daily equity curve as CSV.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import sctr, signals, strategy, twins
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


def load_index(symbol: str = "^GSPC", refresh: bool = False) -> pd.DataFrame:
    """Daily close and volume of a market index (default: the S&P 500), downloaded once."""
    path = DATA_DIR / "benchmark" / f"{symbol.lstrip('^')}.csv"
    if refresh or not path.exists():
        import yfinance as yf

        data = yf.download(symbol, start="2010-01-01", auto_adjust=False, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        path.parent.mkdir(parents=True, exist_ok=True)
        data[["Close", "Volume"]].to_csv(path)
    data = pd.read_csv(path, index_col=0, parse_dates=True)
    return pd.DataFrame({"close": data["Close"], "volume": data["Volume"]})


def market_window(close: pd.Series, volume: pd.Series, dd: int = 0, dd_days: int = 21, sma: int = 0,
                  dd_volume_ma: int = 0) -> pd.Series:
    """True on the evenings new buys are allowed by the index (Rules.market_dd / market_sma):
    fewer than `dd` distribution days (a close 0.2%+ below the day before, on higher volume than
    the day before and, with `dd_volume_ma`, above the average volume of the last that many
    sessions, today included) in the last `dd_days` sessions, and a close at or above the
    `sma`-day average."""
    ok = pd.Series(True, index=close.index)
    if dd:
        distribution = (close.pct_change() <= -0.002) & (volume > volume.shift(1))
        if dd_volume_ma:
            distribution &= volume > volume.rolling(dd_volume_ma).mean()
        ok &= distribution.astype(int).rolling(dd_days, min_periods=1).sum() < dd
    if sma:
        ok &= close >= close.rolling(sma).mean()
    return ok


def load_benchmark(start, symbol: str = "SPY") -> pd.Series:
    """Total-return closes of the benchmark ETF from `start`."""
    return load_etf(symbol)["close"].loc[start:]


def prepare(start, end, top_n: int = 10, require_squeeze: bool = True, find_twins: bool = False, template: str = ""):
    """Daily top N, the trading calendar, OHLC prices and signals for every ticker that ever made the top N.

    template: "full" or "trend" adds the "template" signal column (signals.trend_template; "trend" =
    only the 50-day above the 200-day and within 25% of the 52-week high).

    With find_twins, also the twin setups (backtest/twins.py) of every stock in the universe, with
    their prices and signals; that needs every stock's chart, so it takes a few minutes. Otherwise None.
    """
    inputs = build_inputs(start, end, "largecap")
    top = sctr.daily_top_n(inputs.closes, inputs.mask, n=top_n)
    top = top[top["date"] >= pd.Timestamp(start)].reset_index(drop=True)
    # Volatility over the last year (252 trading days), annualised, in % - for --min-vol / --max-vol.
    vol = inputs.closes.pct_change().rolling(252, min_periods=200).std() * np.sqrt(252) * 100
    lookup = lambda frame, table: frame.to_numpy()[frame.index.get_indexer(table["date"]), frame.columns.get_indexer(table["ticker"])]
    top["vol"] = lookup(vol, top)
    dates = inputs.closes.loc[start:end].index

    tickers = sorted(top["ticker"].unique())
    if find_twins:
        tickers = sorted(set(tickers) | {t for t in inputs.closes.columns if inputs.mask[t].loc[start:end].any()})
    ohlc = load_ohlc(tickers)
    prices, sigs, full = {}, {}, {}
    for ticker, frame in ohlc.items():
        full[ticker] = signals.ticker_signals(frame["high"], frame["low"], frame["close"], require_squeeze,
                                              open_=frame["open"])
        if template:
            full[ticker]["template"] = signals.trend_template(frame["high"], frame["close"], frame["volume"],
                                                              full=template != "trend")
        sigs[ticker] = full[ticker].reindex(dates)
        prices[ticker] = frame.reindex(dates)
    for frame in sigs.values():
        for col in ("setup", "entry_setup", "setup_dim_green", "atr_buy", "atr_exit", "template"):
            if col in frame.columns:
                frame[col] = frame[col].fillna(False).astype(bool)

    twin_setups = None
    if find_twins:
        wide = lambda get: pd.DataFrame({t: get(t) for t in ohlc}).reindex(inputs.closes.index)
        table = sctr.rank_scores(sctr.indicator_score(inputs.closes), inputs.mask)
        twin_setups = twins.find_twins(inputs.closes, wide(lambda t: ohlc[t]["open"]), wide(lambda t: ohlc[t]["high"]),
                                       wide(lambda t: ohlc[t]["close"]), table, wide(lambda t: full[t]["setup"]),
                                       wide(lambda t: full[t]["trend"]), wide(lambda t: full[t]["flip_level"]))
        twin_setups = twin_setups[twin_setups["date"].between(dates[0], dates[-1])].reset_index(drop=True)
        twin_setups["vol"] = lookup(vol, twin_setups)
        keep = set(top["ticker"]) | set(twin_setups["ticker"])  # nobody else can be bought
        prices = {t: f for t, f in prices.items() if t in keep}
        sigs = {t: f for t, f in sigs.items() if t in keep}
    return dates, prices, sigs, top, twin_setups


def market_ok(dates, days: int = 200) -> pd.Series:
    """True on days when SPY and QQQ both closed above their `days`-day average (total-return closes)."""
    ok = pd.Series(True, index=dates)
    for symbol in ("SPY", "QQQ"):
        close = load_etf(symbol)["close"]
        above = (close > close.rolling(days).mean()).reindex(dates).ffill()
        ok &= above.fillna(False).astype(bool)
    return ok


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
    parser.add_argument("--take-profit-fraction", type=float, default=defaults.take_profit_fraction,
                        help="share of the position sold at --take-profit (default 1/3; 0 = none, the rest exit still "
                             "takes over there)")
    parser.add_argument("--take-profit2", type=float, default=defaults.take_profit2,
                        help="also sell part of the position when the price reaches this gain, e.g. 0.25 (default 0 = off)")
    parser.add_argument("--take-profit2-fraction", type=float, default=defaults.take_profit2_fraction,
                        help="share of the original position sold at --take-profit2 (default 0.25)")
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
                        help="keep the last stop when the green line disappears, never lower it")
    parser.add_argument("--breakeven-exit", action=argparse.BooleanOptionalAction, default=defaults.breakeven_exit,
                        help="after +8%%: also sell the rest after a close 3%% below the entry (--breakeven-buffer); off since #23")
    parser.add_argument("--failed-breakout-exit", action=argparse.BooleanOptionalAction,
                        default=defaults.failed_breakout_exit,
                        help="sell at the next open if the entry day closes back below the flip level (default off)")
    parser.add_argument("--rest-exit", choices=["sma50", "ema20", "violation"], default=defaults.rest_exit,
                        help="after +8%%: sma50 = close more than --sma-exit-buffer below the 50 SMA (default); "
                             "ema20 = --ema-exit-days closes in a row below the 20 EMA; violation = 50MA violation: "
                             "the moment the price trades below the low of the last close below the 50 SMA")
    parser.add_argument("--violation-armed", action=argparse.BooleanOptionalAction, default=defaults.violation_armed,
                        help="with --rest-exit violation: only once the stock has closed above its 50 SMA since the "
                             "entry (default off)")
    parser.add_argument("--frozen-line", action=argparse.BooleanOptionalAction, default=defaults.frozen_line,
                        help="no-line entries: use the green line the buy day would have had as a fixed stop "
                             "until the trend turns up (default off)")
    parser.add_argument("--rest-exit-delay", type=int, default=defaults.rest_exit_delay,
                        help="after +8%%: the 50 SMA exit only from this many trading days after the entry "
                             "(default 21 = about a month; 0 = from the start, version #20)")
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
    parser.add_argument("--stop-loss", type=float, default=defaults.stop_loss,
                        help="before --take-profit: sell everything the moment the price touches this loss from the "
                             "entry, e.g. 0.05 = -5%% (default 0 = off)")
    parser.add_argument("--stop-loss-reentry", type=float, default=defaults.stop_loss_reentry,
                        help="for --reentry-days after a --stop-loss sale, also re-buy when the price gets back to the "
                             "old entry price plus this, e.g. 0.02 = +2%% (default 0 = off)")
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
    parser.add_argument("--reentry-top", type=int, default=defaults.reentry_top,
                        help="re-entries only while the stock is in the daily SCTR top N (default 0 = any rank)")
    parser.add_argument("--reentry-max-losses", type=int, default=defaults.reentry_max_losses,
                        help="stop re-buying a stock after N trades in a row (its setup buy and the re-buys after it) "
                             "that made no more than --reentry-win (default 3 since #26; 0 = off = #25)")
    parser.add_argument("--late-entry-days", type=int, default=defaults.late_entry_days,
                        help="buy a top-10 stock at the next open if it gave a valid entry in the last N trading days "
                             "while it was not on the list and is still in its ATR up-trend (default 0 = off)")
    parser.add_argument("--list-memory", type=int, default=defaults.list_memory,
                        help="also look for the setup in stocks that were on the list on any of the last N trading "
                             "days, e.g. 21 = about a month (default 0 = only today's list)")
    parser.add_argument("--list-memory-min-sctr", type=float, default=defaults.list_memory_min_sctr,
                        help="with --list-memory: forget a stock once its SCTR closes below this, e.g. 50 (default 0 = off)")
    parser.add_argument("--list-memory-min-run", type=int, default=defaults.list_memory_min_run,
                        help="with --list-memory: only stocks that were on the list this many trading days in a row "
                             "before they left, e.g. 42 = about 2 months (default 0 = any)")
    parser.add_argument("--list-memory-gap", type=int, default=defaults.list_memory_gap,
                        help="breaks of up to this many trading days don't end a run on the list (default 5)")
    parser.add_argument("--list-memory-above-ma", type=int, default=defaults.list_memory_above_ma,
                        help="with --list-memory: only while the setup close is above this SMA, 50 or 100 (default 0 = off)")
    parser.add_argument("--twin-days", type=int, default=defaults.twin_days,
                        help="twin setups outside the top 10 (META Mar 2023 / TER Sep 2025 look-alikes, see "
                             "backtest/twins.py) put the stock on the list for N trading days, e.g. 20 (default 0 = off)")
    parser.add_argument("--profit-floor", type=float, default=defaults.profit_floor,
                        help="after +8%%: once the trade has been up --profit-floor-after at a close, sell the rest "
                             "after a close below this gain, e.g. 0.10 = +10%% (default 0 = off)")
    parser.add_argument("--profit-floor-after", type=float, default=defaults.profit_floor_after,
                        help="with --profit-floor: the gain the trade must have reached first (default 0.20)")
    parser.add_argument("--reentry-win", type=float, default=defaults.reentry_win,
                        help="with --reentry-max-losses: a trade must make more than this to count as a win "
                             "(default 0.20 = +20%%)")
    parser.add_argument("--extra-slots", type=int, default=defaults.extra_slots,
                        help="also trade stocks above --min-sctr outside the top N, at most this many at a time")
    parser.add_argument("--min-vol", type=float, default=defaults.min_vol,
                        help="only stocks whose 1-year volatility (%% a year) is above this")
    parser.add_argument("--max-vol", type=float, default=defaults.max_vol,
                        help="only stocks whose 1-year volatility (%% a year) is below this (0 = off)")
    parser.add_argument("--climax-gain", type=float, default=defaults.climax_gain,
                        help="climax exit once a trade has been up this much (default 0.40; 0 = off = #23)")
    parser.add_argument("--climax-stretch", type=float, default=defaults.climax_stretch,
                        help="... after a close this far above the --climax-ma SMA (default 0.50)")
    parser.add_argument("--climax-days", type=int, default=defaults.climax_days,
                        help="... within this many trading days before the gap down (default 10)")
    parser.add_argument("--climax-reentry", action=argparse.BooleanOptionalAction, default=defaults.climax_reentry,
                        help="allow the usual re-entry after a climax exit (default off)")
    parser.add_argument("--climax-ma", type=int, choices=[50, 100], default=defaults.climax_ma,
                        help="... measured against the 50 or 100-day SMA (default 100)")
    parser.add_argument("--climax-gap", type=float, default=defaults.climax_gap,
                        help="... gap down = open at least this far below the previous low (default 0.05)")
    parser.add_argument("--climax-drop", type=float, default=defaults.climax_drop,
                        help="... or a big red candle: a close this far below the previous close, e.g. 0.08 (0 = off)")
    parser.add_argument("--climax-body", type=float, default=defaults.climax_body,
                        help="... or a close this far below the day's open (default 0.06; 0 = off = #24)")
    parser.add_argument("--climax-drop-atr", type=float, default=defaults.climax_drop_atr,
                        help="... or a fall from the previous close of N x the 14-day ATR, e.g. 2.5 (0 = off)")
    parser.add_argument("--market-ma", type=int, default=defaults.market_ma,
                        help="no new buys unless SPY and QQQ close above their N-day average, e.g. 200 (0 = off)")
    parser.add_argument("--market-dd", type=int, default=defaults.market_dd,
                        help="no new buys while the S&P 500 had N or more distribution days (a close 0.2%%+ lower on "
                             "higher volume than the day before) in the last --market-dd-days, e.g. 4 (default 0 = off)")
    parser.add_argument("--market-dd-days", type=int, default=defaults.market_dd_days,
                        help="with --market-dd: the window in trading days (default 21, about a month)")
    parser.add_argument("--market-dd-volume-ma", type=int, default=defaults.market_dd_volume_ma,
                        help="with --market-dd: a distribution day also needs volume above its N-day average, e.g. 50 "
                             "(default 0 = off)")
    parser.add_argument("--market-sma", type=int, default=defaults.market_sma,
                        help="no new buys after an S&P 500 close below its N-day average, e.g. 20 (default 0 = off)")
    parser.add_argument("--market-reentry", action=argparse.BooleanOptionalAction, default=defaults.market_reentry,
                        help="the market filters stop re-entries too (default); --no-market-reentry: new setup buys only")
    parser.add_argument("--template", nargs="?", const="full", choices=["full", "trend"], default=None,
                        help="only buy setups in stocks that meet the uptrend template that evening: above the 50-day "
                             "SMA, 50-day above the 200-day, more volume on up days than down days over 20 sessions, "
                             "within 25%% of the 52-week high (default off); --template trend: only the 50-day above "
                             "the 200-day and within 25%% of the 52-week high")
    parser.add_argument("--template-reentry", action=argparse.BooleanOptionalAction, default=defaults.template_reentry,
                        help="with --template: re-entries must meet it too (default off)")
    parser.add_argument("--red-line-stop", type=float, default=defaults.red_line_stop,
                        help="no green line yet: sell this far below the previous close's red line, e.g. 0.03 (0 = off)")
    parser.add_argument("--reentry-after-tp", action=argparse.BooleanOptionalAction, default=defaults.reentry_after_tp,
                        help="also re-enter after the rest was sold after +8%% (default since #23; --no-reentry-after-tp = #22)")
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
        take_profit=args.take_profit, take_profit_fraction=args.take_profit_fraction, take_profit2=args.take_profit2,
        take_profit2_fraction=args.take_profit2_fraction, cost=args.cost, capital=args.capital, ema_exit_days=args.ema_exit_days,
        atr_exit_on_touch=args.atr_exit_on_touch, entry_on_touch=args.entry_on_touch,
        emergency_stop=args.emergency_stop, stop_loss=args.stop_loss, stop_loss_reentry=args.stop_loss_reentry, reentry_days=args.reentry_days, reentry_top=args.reentry_top,
        extra_slots=args.extra_slots, min_vol=args.min_vol, max_vol=args.max_vol, reentry_after_tp=args.reentry_after_tp,
        red_line_stop=args.red_line_stop, climax_gain=args.climax_gain, climax_stretch=args.climax_stretch,
        climax_days=args.climax_days, climax_ma=args.climax_ma, climax_gap=args.climax_gap, market_ma=args.market_ma,
        climax_drop=args.climax_drop, climax_body=args.climax_body, climax_drop_atr=args.climax_drop_atr,
        climax_reentry=args.climax_reentry, reentry_max_losses=args.reentry_max_losses, reentry_win=args.reentry_win,
        market_dd=args.market_dd, market_dd_days=args.market_dd_days, market_dd_volume_ma=args.market_dd_volume_ma,
        market_sma=args.market_sma,
        market_reentry=args.market_reentry, template=bool(args.template), template_reentry=args.template_reentry,
        profit_floor=args.profit_floor, profit_floor_after=args.profit_floor_after, list_memory=args.list_memory,
        late_entry_days=args.late_entry_days, twin_days=args.twin_days,
        list_memory_min_sctr=args.list_memory_min_sctr, list_memory_min_run=args.list_memory_min_run,
        list_memory_gap=args.list_memory_gap, list_memory_above_ma=args.list_memory_above_ma,
        atr_stop_buffer=args.atr_stop_buffer, rebuy_shakeouts=args.rebuy_shakeouts,
        close_exit=args.close_exit, breakeven_exit=args.breakeven_exit,
        rest_exit=args.rest_exit, violation_armed=args.violation_armed, rest_exit_delay=args.rest_exit_delay, frozen_line=args.frozen_line, entry_check=args.entry_check, breakeven_buffer=args.breakeven_buffer, close_entry=args.close_entry, dim_green_days=args.dim_green_days, failed_breakout_exit=args.failed_breakout_exit, sma_exit_buffer=args.sma_exit_buffer, sticky_stop=args.sticky_stop, entry_buffer=args.entry_buffer, park_cost=args.park_cost,
    )
    ranks = 10**4 if args.extra_slots or args.list_memory_min_sctr else max(args.top, args.reentry_top)
    dates, prices, sigs, top, twin_setups = prepare(args.start, args.end, ranks, not args.no_squeeze, bool(args.twin_days),
                                                    args.template or "")
    park = load_etf(args.park).reindex(dates).ffill() if args.park else None
    market = None
    if args.market_ma or args.market_dd or args.market_sma:
        market = pd.Series(True, index=dates)
        if args.market_ma:
            market &= market_ok(dates, args.market_ma)
        if args.market_dd or args.market_sma:
            spx = load_index("^GSPC")
            window = market_window(spx["close"], spx["volume"], args.market_dd, args.market_dd_days, args.market_sma,
                                   args.market_dd_volume_ma)
            market &= window.reindex(dates).ffill().fillna(False).astype(bool)
    equity, trades = strategy.run(dates, prices, sigs, top, rules, park=park, market=market, twins=twin_setups)
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
