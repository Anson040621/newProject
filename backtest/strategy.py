"""Step 2-4 - the trading rules and a day-by-day portfolio simulation.

Rules (defaults = version #23 = #22 without the break-even exit and with re-entry after
any exit; #22 = #21 with a 75% volatility limit on the stocks):
  Stocks  the day's SCTR top 10 among US stocks over $10B (all score above 90),
          skipping stocks whose volatility over the last year (252 trading days,
          annualised) is 75% or more (--max-vol, 0 = off = version #21; --min-vol
          skips the calmest ones).
  Entry   setup at a close: the stock is in the top 10, the squeeze is on (blue
          crosses), the momentum bar is red and the 2x ATR trail is in a
          down-trend. Next day: buy-stop at the ATR flip level (that close's
          trail line); filled when the price touches it, or at the open if it
          gaps above.
          Option --failed-breakout-exit: if the entry day closes back below the
          flip level (no BUY signal, no green line yet), sell at the next open.
  Size    15% of account equity per trade, at most 6 positions. When more
          signals than free slots appear, the highest SCTR goes first.
  Exit    before +8%, whichever comes first:
            a) a close below the green ATR line (the SELL label): sell
               everything at the next open;
            b) safety net: the price touches 2% below the previous close's green
               line: sell everything at once (at the open if it gaps below).
          at +8%:     sell 1/3 at the +8% price (or at the open if it gaps above).
          after +8%:  sell the rest at the next open after a close more than 3%
                      below the 50-day SMA, once price has closed above that level
                      since entry and from 21 trading days after the entry or
                      re-entry on (--rest-exit-delay; 0 = from the start) (--rest-exit ema20 for the old rule: 3 closes in
                      a row below the 20 EMA). Nothing else sells the rest: between
                      the +8% sale and day 21 it is simply held.
                      Option --breakeven-exit (#22): also sell the rest after a close
                      more than 3% below the entry price (--breakeven-buffer).
  Re-entry after any exit (before or after +8%), if the ATR trail flips back to BUY
          within 5 trading days of the exit: buy-stop at the flip level (no other
          conditions; --reentry-top N: only while the stock is in the top N).
          Option --no-reentry-after-tp (#22): only after exits before +8%.
  Option --entry-check live: every evening a buy order waits at the flip level for
          each top-10 stock in an ATR down-trend; the next day it buys at the first
          price, on the way up through the line, at which the squeeze is on and the
          bar is red (as the indicators read live). --entry-check either: the old
          order (setup at the previous close) or the live check.
  Option --climax-gain 0.40: climax exit for big winners. Once a trade has been up 40%
          (at a close) and within the last 10 trading days closed 30% or more above
          its 50-day SMA (--climax-stretch, --climax-days, --climax-ma 100 for the
          100-day SMA: a steep, compressed run), a gap down (open below the previous
          day's low; --climax-gap 0.05: at least 5% below it) sells everything at the
          next open, with no re-entry.
  Option --market-ma 200: market filter - no new buys (setups or re-entries) unless
          SPY and QQQ both closed above their 200-day average that evening.
  Option --red-line-stop 0.03: while a position has no green line yet (the buy day
          closed back below the flip line), sell everything the moment the price touches
          3% below the previous close's red line (the down-trend line it was bought at),
          at the open if it gaps below.
  Option --frozen-line: when the buy day closes back below the flip line (no green
          line), the green line that day would have had is frozen and used like a
          normal green line (2% stop below it, sell after a close below it) until
          the trend really turns up; then the real green line takes over.
  Option --close-entry: when the whole setup (squeeze on, red bar, ATR flips to
          BUY) only shows on the BUY bar itself - so no buy-stop was waiting - buy at
          the next open.
  Option --dim-green-days N: for N trading days after an entry on the full
          setup (42 = about 2 months), once we have sold, the same setup with a
          dim green momentum bar (above zero, not rising) also counts.
  Option --extra-slots N: also trade the setup on any stock with an SCTR above 90
          outside the top 10, at most N such positions at a time (their re-entries
          count too). The top 10 go first when signals arrive on the same day.
  Costs   0.1% of the traded value on every buy and sell (commission + slippage).

Tested alternatives (options): --entry-buffer 0.02 (buy-stop 2% above the flip),
--no-close-exit with --sticky-stop and --atr-stop-buffer 0.03 (stop-only exit,
"version #14"), --emergency-stop 0.12 (hard stop, "version #15"),
--no-atr-exit-on-touch (version G: close-based exit only), --rebuy-shakeouts,
--no-entry-on-touch, --reentry-days N, --ema-exit-days N, --park QQQ.

Prices are split-adjusted, not dividend-adjusted (like a TradingView chart);
dividends received are ignored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Rules:
    top_n: int = 10
    min_sctr: float = 90.0
    position_size: float = 0.15
    max_positions: int = 6
    take_profit: float = 0.08
    take_profit_fraction: float = 1 / 3
    ema_exit_days: int = 3  # consecutive closes below the 20 EMA before selling the rest
    atr_exit_on_touch: bool = True  # before +8%: sell the moment price touches the ATR stop (see buffer)
    entry_on_touch: bool = True  # buy intraday when price touches the ATR flip level
    emergency_stop: float = 0.0  # hard stop: sell everything at -X from entry, any time (e.g. 0.12; 0 = off)
    reentry_days: int = 5  # re-buy if the ATR flips back to BUY within N days of an early exit (0 = off)
    reentry_top: int = 0  # ... only while the stock is in the daily top N at the signal close (0 = any rank)
    reentry_after_tp: bool = True  # ... also after exits that came after the +8% partial
    extra_slots: int = 0  # also buy stocks above min_sctr outside the top N, at most this many at a time (0 = off)
    min_vol: float = 0.0  # only stocks whose 1-year volatility (% a year, column "vol") is above this (0 = off)
    max_vol: float = 75.0  # ... and below this (0 = off)
    park_cost: float = 0.0005  # cost per move in/out of the parking ETF (with park=...)
    atr_stop_buffer: float = 0.02  # with atr_exit_on_touch: stop this far below the trail line (0.02 = 2%)
    sticky_stop: bool = False  # keep the last stop when the green line disappears; never lower it
    close_exit: bool = True  # before +8%: also sell at the next open after a close below the line
    entry_buffer: float = 0.0  # buy-stop this far above the ATR flip level (entries and re-entries)
    breakeven_exit: bool = False  # after +8%: sell the rest at the next open after a close at/below the entry
    breakeven_buffer: float = 0.03  # ... more than this far below the entry (0.03 = a close 3% below it)
    failed_breakout_exit: bool = False  # entry day closes back below the flip level -> sell at the next open
    rest_exit: str = "sma50"  # after +8%: "sma50" (close below the 50 SMA minus buffer) or "ema20" (N closes below)
    frozen_line: bool = False  # no-line entries: the green line the buy day would have had is a fixed stop
    red_line_stop: float = 0.0  # no-line positions: sell this far below the previous close's red line (0 = off)
    climax_gain: float = 0.0  # climax exit once a trade has been up this much at a close (0.40 = +40%; 0 = off) ...
    climax_stretch: float = 0.30  # ... and closed this far above its 50-day SMA ...
    climax_days: int = 10  # ... within this many trading days: a gap down sells at the next open, no re-entry
    climax_reentry: bool = False  # ... unless this is on (the usual re-entry rule then applies)
    climax_ma: int = 50  # the average the stretch is measured against (50 or 100-day SMA)
    climax_gap: float = 0.0  # gap down = open at least this far below the previous day's low (0 = any)
    market_ma: int = 0  # market filter: no new buys unless SPY and QQQ close above their N-day average (0 = off)
    rest_exit_delay: int = 21  # ... the rest exit only from this many trading days after the entry (0 = at once)
    sma_exit_buffer: float = 0.03  # with rest_exit="sma50": the close must be this far below the 50 SMA
    entry_check: str = "prev"  # when the squeeze / red bar must show: "prev" close, "live" as the price
    #                            crosses the flip line, or "either" (see the module docstring)
    close_entry: bool = False  # also buy at the next open when the whole setup shows on the BUY bar itself
    dim_green_days: int = 0  # for N trading days after a full-setup entry, also buy the setup on a dim green bar
    rebuy_shakeouts: bool = False  # after an intraday ATR stop, buy back at the next open if the trend is still up
    cost: float = 0.001
    capital: float = 100_000.0


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    initial_shares: int
    sctr: float
    tp_taken: bool = False
    armed: bool = False  # has closed above the rest-exit line (20 EMA or 50 SMA level) since entry
    closes_below_ema: int = 0  # consecutive closes below that line
    days_without_price: int = 0
    intraday_entry: bool = False
    entry_day: int = 0  # trading-day number of the entry
    frozen_line: float = float("nan")  # fixed stand-in for the green line while there is none (frozen_line)
    best_close: float = 0.0  # highest close since entry (climax exit)
    stretch_day: int = -10**9  # last trading day it closed far above its 50-day SMA (climax exit)
    stop_line: float = float("nan")  # green line the intraday stop is based on
    reentry: bool = False
    kind: str = "setup"  # how the entry came about: "setup", "reentry" or "dim-green"
    extra: bool = False  # bought from outside the top N (extra_slots), or a re-entry of such a trade
    proceeds: float = 0.0  # cash received from sales so far (after costs)
    cost_basis: float = 0.0  # cash paid at entry (incl. costs)
    fills: list = field(default_factory=list)


def run(
    dates: pd.DatetimeIndex,
    prices: dict[str, pd.DataFrame],
    signals: dict[str, pd.DataFrame],
    top: pd.DataFrame,
    rules: Rules = Rules(),
    park: pd.DataFrame | None = None,
    market: pd.Series | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Simulate the strategy.

    prices:  ticker -> DataFrame(open, high, low, close) on the full trading calendar
    signals: ticker -> output of signals.ticker_signals on the same calendar
    top:     daily top-N table (date, rank, ticker, sctr)
    park:    optional DataFrame(open, close) of a total-return ETF (e.g. QQQ) on the
             same calendar. All money not in a trade is held in it: buying a stock
             sells the ETF, selling a stock buys it back (rules.park_cost per move).
             Moves during the day happen at the day's ETF open-to-close average return
             (we only have daily data), i.e. the idle balance earns overnight and
             open-to-close returns.
    Returns the daily equity curve and the trade log.
    """
    listed = 10**9 if rules.extra_slots else rules.top_n  # extra_slots: every stock above min_sctr
    top = top[(top["rank"] <= max(listed, rules.reentry_top)) & (top["sctr"] > rules.min_sctr)]
    if rules.market_ma and market is None:
        raise ValueError("market_ma needs the market series (True = SPY and QQQ above their average)")
    if (rules.min_vol or rules.max_vol) and "vol" not in top.columns:
        raise ValueError("min_vol / max_vol need a 'vol' column in the top table (see run_backtest.prepare)")
    if rules.min_vol:
        top = top[top["vol"] > rules.min_vol]
    if rules.max_vol:
        top = top[top["vol"] < rules.max_vol]
    top_by_day = {d: g[g["rank"] <= listed].sort_values("rank") for d, g in top.groupby("date")}
    reentry_ok = {d: set(g.loc[g["rank"] <= rules.reentry_top, "ticker"]) for d, g in top.groupby("date")}

    cash = rules.capital
    positions: dict[str, Position] = {}
    pending_exits: dict[str, str] = {}  # ticker -> reason, sold at the next open
    pending_entries: list[tuple[str, float, str]] = []  # (ticker, sctr, kind), filled at the open
    touch_orders: list[tuple[str, float, str]] = []  # buy-stop orders at the ATR flip level, for today
    early_exits: dict[str, tuple[int, float, str]] = {}  # ticker -> (day number of an exit before +8%, sctr, how)
    setup_entries: dict[str, int] = {}  # ticker -> day number of the last full-setup entry
    entered_today: set[str] = set()
    extra_orders: set[str] = set()  # tonight's orders that count against extra_slots
    extra_tickers: set[str] = set()  # tickers whose last trade did (so their re-entries do too)
    last_close: dict[str, float] = {}
    equity = pd.Series(np.nan, index=dates)
    trades = []

    def px(ticker, day, field_):
        value = prices[ticker].at[day, field_]
        return None if pd.isna(value) else float(value)

    def sell(pos: Position, shares: int, price: float, day, reason: str):
        nonlocal cash
        value = shares * price
        cash += value * (1 - rules.cost) * (1 - park_cost)
        pos.proceeds += value * (1 - rules.cost)
        pos.shares -= shares
        pos.fills.append((day, reason, shares, price))

    def close_out(pos: Position, day):
        trades.append({
            "ticker": pos.ticker,
            "entry_date": pos.entry_date,
            "entry_price": pos.entry_price,
            "shares": pos.initial_shares,
            "sctr_at_signal": pos.sctr,
            "reentry": pos.reentry,
            "entry_kind": pos.kind,
            "extra": pos.extra,
            "exit_date": day,
            "exits": "; ".join(f"{d.date()} {r} {s}@{p:.2f}" for d, r, s, p in pos.fills),
            "took_profit": pos.tp_taken,
            "pnl": pos.proceeds - pos.cost_basis,
            "return_pct": (pos.proceeds / pos.cost_basis - 1) * 100,
            "days_held": (day - pos.entry_date).days,
        })
        del positions[pos.ticker]
        climax = bool(pos.fills) and pos.fills[-1][1] == "climax-exit" and not rules.climax_reentry
        if rules.reentry_days and (not pos.tp_taken or rules.reentry_after_tp) and not climax:
            early_exits[pos.ticker] = (day_number, pos.sctr, pos.fills[-1][1] if pos.fills else "")

    def buy(ticker, price, day, sctr, kind, intraday) -> bool:
        nonlocal cash
        if len(positions) >= rules.max_positions or ticker in positions:
            return False
        extra = ticker in extra_orders
        if extra and sum(p.extra for p in positions.values()) >= rules.extra_slots:
            return False
        shares = math.floor(rules.position_size * prev_equity / (price * (1 + rules.cost)))
        if shares <= 0 or shares * price * (1 + rules.cost) * (1 + park_cost) > cash:
            return False
        paid = shares * price * (1 + rules.cost)
        cash -= paid * (1 + park_cost)
        positions[ticker] = Position(ticker, day, price, shares, shares, sctr, cost_basis=paid,
                                     intraday_entry=intraday, reentry=kind == "reentry", kind=kind,
                                     entry_day=day_number, extra=extra)
        (extra_tickers.add if extra else extra_tickers.discard)(ticker)
        early_exits.pop(ticker, None)
        entered_today.add(ticker)
        if kind.startswith("setup"):
            setup_entries[ticker] = day_number
        return True

    park_cost = rules.park_cost if park is not None else 0.0
    if park is not None:
        cash *= 1 - park_cost  # initial purchase of the parking ETF
        park_open, park_close = park["open"].to_numpy(dtype=float), park["close"].to_numpy(dtype=float)

    prev_equity = cash
    for day_number, day in enumerate(dates):
        entered_today.clear()
        # 0. Idle money in the parking ETF earns its overnight return (yesterday's close -> today's open).
        if park is not None and day_number > 0 and park_open[day_number] > 0 and park_close[day_number - 1] > 0:
            cash *= park_open[day_number] / park_close[day_number - 1]

        # 1. Orders from yesterday's close are filled at today's open.
        for ticker, reason in sorted(pending_exits.items()):
            pos = positions.get(ticker)
            price = px(ticker, day, "open") or px(ticker, day, "close")
            if pos is not None and price is not None:
                sell(pos, pos.shares, price, day, reason)
                close_out(pos, day)
        pending_exits.clear()

        for ticker, sctr, kind in pending_entries:
            price = px(ticker, day, "open") or px(ticker, day, "close")
            if price is not None:
                buy(ticker, price, day, sctr, kind, intraday=False)
        pending_entries = []

        # 1b. Buy-stop orders at the ATR flip level (--entry-on-touch): filled when the
        #     high reaches yesterday's trail line, at the open if it gaps above.
        for ticker, sctr, kind in touch_orders:
            level = signals[ticker].at[day, "flip_level"] * (1 + rules.entry_buffer)
            high, open_ = px(ticker, day, "high"), px(ticker, day, "open")
            if pd.isna(level) or high is None or high < level:
                continue
            fill = max(level, open_) if open_ is not None else level
            if kind == "watch":
                # Bought at the first price on the way up at which the squeeze is on and the bar red.
                live = signals[ticker].at[day, "touch_price"]
                if pd.isna(live):
                    continue
                fill, kind = max(fill, live), "setup-live"
            buy(ticker, fill, day, sctr, kind, intraday=True)
        touch_orders = []

        # 2. Hard stop: sell everything at -X% from the entry (at the open if it gaps below).
        if rules.emergency_stop:
            for pos in list(positions.values()):
                stop = pos.entry_price * (1 - rules.emergency_stop)
                low, open_ = px(pos.ticker, day, "low"), px(pos.ticker, day, "open")
                if pos.intraday_entry and pos.entry_date == day:
                    # Bought during the day: the low may have come before the purchase. Only a
                    # close below the stop proves the price fell through it after we bought.
                    low = px(pos.ticker, day, "close")
                if low is None or low > stop:
                    continue
                fill = min(stop, open_) if open_ is not None and day != pos.entry_date else stop
                sell(pos, pos.shares, fill, day, "emergency-stop")
                close_out(pos, day)

        # 2a. Intraday ATR stop: before +8%, sell everything the moment the price touches
        #     the stop (buffer below yesterday's green line), at the open if it gaps below.
        if rules.atr_exit_on_touch:
            for pos in list(positions.values()):
                line = signals[pos.ticker].at[day, "atr_stop"]
                frozen = pd.isna(line) and pd.notna(pos.frozen_line)
                if frozen:
                    line = pos.frozen_line  # no green line yet: the one the buy day would have had
                if not rules.sticky_stop:
                    pos.stop_line = line
                elif pd.notna(line):
                    pos.stop_line = line if pd.isna(pos.stop_line) else max(pos.stop_line, line)
                stop = pos.stop_line * (1 - rules.atr_stop_buffer)
                low, open_ = px(pos.ticker, day, "low"), px(pos.ticker, day, "open")
                if pos.tp_taken or pd.isna(stop) or low is None or low > stop:
                    continue
                fill = min(stop, open_) if open_ is not None and day != pos.entry_date else stop
                sell(pos, pos.shares, fill, day, "no-line-stop" if frozen else "atr-stop")
                close_out(pos, day)

        # 2a'. Red-line stop: while a position has no green line yet (the buy day closed back
        #      below the flip line), sell everything the moment the price touches the stop below
        #      yesterday's red line, at the open if it gaps below.
        if rules.red_line_stop:
            for pos in list(positions.values()):
                if pos.tp_taken or day == pos.entry_date or pd.notna(signals[pos.ticker].at[day, "atr_stop"]):
                    continue
                red = signals[pos.ticker].at[day, "flip_level"]
                low, open_ = px(pos.ticker, day, "low"), px(pos.ticker, day, "open")
                stop = red * (1 - rules.red_line_stop)
                if pd.isna(stop) or low is None or low > stop:
                    continue
                sell(pos, pos.shares, min(stop, open_) if open_ is not None else stop, day, "red-line-stop")
                close_out(pos, day)

        # 2b. Intraday: take 1/3 profit when the high reaches +8%.
        for pos in list(positions.values()):
            high, open_ = px(pos.ticker, day, "high"), px(pos.ticker, day, "open")
            target = pos.entry_price * (1 + rules.take_profit)
            if not pos.tp_taken and high is not None and high >= target:
                fill = max(target, open_) if open_ is not None and day != pos.entry_date else target
                qty = max(1, round(pos.initial_shares * rules.take_profit_fraction))
                sell(pos, min(qty, pos.shares), fill, day, "take-profit")
                pos.tp_taken = True
                if pos.shares == 0:
                    close_out(pos, day)

        # 2c. The parking ETF's open -> close return on whatever is still idle at the close
        #     (cash from intraday trades counts as idle for the whole day).
        if park is not None and park_open[day_number] > 0:
            cash *= park_close[day_number] / park_open[day_number]

        # 3. At the close: exit signals for open positions.
        for pos in list(positions.values()):
            sig = signals[pos.ticker]
            close = px(pos.ticker, day, "close")
            if close is None:
                # Delisted or halted: after 10 trading days without a price, close at the last price.
                pos.days_without_price += 1
                if pos.days_without_price > 10:
                    sell(pos, pos.shares, last_close.get(pos.ticker, pos.entry_price), day, "no price")
                    close_out(pos, day)
                    pending_exits.pop(pos.ticker, None)
                continue
            pos.days_without_price = 0
            gap_down = False
            if rules.climax_gain:
                pos.best_close = max(pos.best_close, close)
                ma = sig.at[day, f"sma{rules.climax_ma}"]
                if pd.notna(ma) and close >= ma * (1 + rules.climax_stretch):
                    pos.stretch_day = day_number
                open_, prev_low = px(pos.ticker, day, "open"), px(pos.ticker, dates[day_number - 1], "low")
                gap_down = open_ is not None and prev_low is not None and open_ < prev_low * (1 - rules.climax_gap)
            last_close[pos.ticker] = close
            # The line the rest (after +8%) is sold on: 3% below the 50-day SMA, or the 20 EMA.
            if rules.rest_exit == "sma50" and pd.notna(sig.at[day, "sma50"]):
                rest_line, closes_needed = sig.at[day, "sma50"] * (1 - rules.sma_exit_buffer), 1
            else:
                rest_line, closes_needed = sig.at[day, "ema20"], rules.ema_exit_days
            pos.closes_below_ema = pos.closes_below_ema + 1 if close < rest_line else 0
            if rules.frozen_line and pos.intraday_entry and pos.entry_date == day and sig.at[day, "trend"] != 1:
                # Bought on the touch, but the day closed back below the flip line: no green line.
                # Freeze the line the day would have had, had it closed above (until a real one shows).
                pos.frozen_line = sig.at[day, "long_line"]
            elif sig.at[day, "trend"] == 1:
                pos.frozen_line = float("nan")  # broke out: the real green line takes over tomorrow
            if (rules.climax_gain and gap_down and day != pos.entry_date
                    and pos.best_close >= pos.entry_price * (1 + rules.climax_gain)
                    and day_number - pos.stretch_day <= rules.climax_days):
                # A long run, then a steep (compressed) leg far above the 50-day SMA, then a gap down.
                pending_exits[pos.ticker] = "climax-exit"
            elif (rules.failed_breakout_exit and pos.intraday_entry and pos.entry_date == day
                    and sig.at[day, "trend"] != 1 and pos.shares > 0):
                # Bought on the touch of the flip level, but the day closed back below it:
                # no BUY signal and no green line (so no stop). Get out at the next open.
                pending_exits[pos.ticker] = "failed-breakout"
            elif not pos.tp_taken:
                # A close below the ATR line (the trail flips to SELL) always exits at the
                # next open, also with the intraday stop: with --atr-stop-buffer the stop
                # sits below the line, so a close can land between the two.
                if (rules.close_exit or not rules.atr_exit_on_touch) and bool(sig.at[day, "atr_exit"]):
                    pending_exits[pos.ticker] = "exit"
                elif ((rules.close_exit or not rules.atr_exit_on_touch) and pd.notna(pos.frozen_line)
                        and day != pos.entry_date and close < pos.frozen_line):
                    pending_exits[pos.ticker] = "no-line-exit"  # a close below the frozen line
            elif (pos.armed and pos.closes_below_ema >= closes_needed
                  and day_number - pos.entry_day >= rules.rest_exit_delay):
                pending_exits[pos.ticker] = "exit"
            elif rules.breakeven_exit and close <= pos.entry_price * (1 - rules.breakeven_buffer):
                pending_exits[pos.ticker] = "breakeven"  # gave back the whole gain
            if close > rest_line:
                pos.armed = True

        # 4. At the close: new entry signals among today's top 10.
        today = top_by_day.get(day)
        candidates, close_setups = [], []
        if today is not None:
            for row in today.itertuples():
                sig = signals.get(row.ticker)
                if sig is None or row.ticker in positions:
                    continue
                # Follow-up entry: we bought this stock on the full setup within the last
                # N days and have since sold; now the same setup with a dim green bar counts too.
                since = day_number - setup_entries.get(row.ticker, -10**9)
                kind = "dim-green" if rules.dim_green_days and since <= rules.dim_green_days else None
                if not rules.entry_on_touch and bool(sig.at[day, "entry_setup"]):
                    candidates.append((row.ticker, row.sctr, "setup"))
                elif rules.entry_on_touch and sig.at[day, "trend"] != 1 and (
                        rules.entry_check != "prev" or bool(sig.at[day, "setup"])):
                    # Order for tomorrow at the flip level. "watch": the squeeze and red bar are
                    # checked live tomorrow, as the price rises through the line.
                    ready = bool(sig.at[day, "setup"]) and rules.entry_check != "live"
                    candidates.append((row.ticker, row.sctr, "setup" if ready else "watch"))
                elif (rules.entry_on_touch and rules.close_entry and bool(sig.at[day, "entry_setup"])
                        and row.ticker not in entered_today):
                    # The squeeze / red bar only appeared on the BUY bar itself, so no order was
                    # waiting at the flip level (HOOD, 9 Apr 2025): buy at the next open instead.
                    close_setups.append((row.ticker, row.sctr, "setup-close"))
                elif kind and bool(sig.at[day, "setup_dim_green"]) and (
                        sig.at[day, "trend"] != 1 if rules.entry_on_touch else bool(sig.at[day, "atr_buy"])):
                    candidates.append((row.ticker, row.sctr, kind))

        # 5. Re-entry after an early exit (or any exit with reentry_after_tp): the ATR
        #    trail flips back to BUY within N days.
        for ticker, (exit_day, sctr, how) in list(early_exits.items()):
            sig = signals[ticker]
            if ticker in positions:
                continue
            # With reentry_top: no order tonight unless the stock is in the top N (the window keeps running).
            listed = not rules.reentry_top or ticker in reentry_ok.get(day, ())
            # Shaken out: the intraday stop sold us, but the close is still in an
            # up-trend (it never flipped to SELL) -> buy back at the next open.
            if (rules.rebuy_shakeouts and listed and how == "atr-stop" and sig.at[day, "trend"] == 1
                    and day_number - exit_day <= rules.reentry_days):
                pending_entries.append((ticker, sctr, "reentry"))
                del early_exits[ticker]
                continue
            if rules.entry_on_touch:
                if day_number + 1 - exit_day > rules.reentry_days:
                    del early_exits[ticker]
                elif listed and sig.at[day, "trend"] != 1:
                    candidates.insert(0, (ticker, sctr, "reentry"))
            elif day_number - exit_day > rules.reentry_days:
                del early_exits[ticker]
            elif listed and bool(sig.at[day, "atr_buy"]):
                candidates.insert(0, (ticker, sctr, "reentry"))

        if rules.market_ma and not bool(market.get(day, False)):
            candidates, close_setups = [], []  # market filter: SPY or QQQ below its average, no new buys
        seen = set()
        for order in candidates:
            if order[0] not in seen:
                seen.add(order[0])
                (touch_orders if rules.entry_on_touch else pending_entries).append(order)
        for order in close_setups:
            if order[0] not in seen:
                seen.add(order[0])
                pending_entries.append(order)
        # extra_slots: orders for stocks outside the top N, and re-entries of such trades.
        extra_orders.clear()
        if rules.extra_slots and today is not None:
            outside = set(today.loc[today["rank"] > rules.top_n, "ticker"])
            extra_orders.update(t for t, _, kind in touch_orders + pending_entries
                                if (t in extra_tickers if kind == "reentry" else t in outside))

        prev_equity = cash + sum(p.shares * last_close.get(p.ticker, p.entry_price) for p in positions.values())
        equity[day] = prev_equity

    # Positions still open at the end are valued at the last close.
    last_day = dates[-1]
    for pos in list(positions.values()):
        pos.proceeds += pos.shares * last_close.get(pos.ticker, pos.entry_price)
        pos.fills.append((last_day, "open at end", pos.shares, last_close.get(pos.ticker, pos.entry_price)))
        close_out(pos, last_day)
    return equity, pd.DataFrame(trades)


def summary(equity: pd.Series, trades: pd.DataFrame, benchmark: pd.Series | None = None) -> dict:
    """Headline statistics of an equity curve and its trades."""
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    daily = equity.pct_change().dropna()
    drawdown = equity / equity.cummax() - 1
    stats = {
        "start": equity.index[0].date(),
        "end": equity.index[-1].date(),
        "final_equity": equity.iloc[-1],
        "total_return_pct": (equity.iloc[-1] / equity.iloc[0] - 1) * 100,
        "cagr_pct": ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100,
        "max_drawdown_pct": drawdown.min() * 100,
        "sharpe": daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else float("nan"),
        "trades": len(trades),
    }
    if len(trades):
        wins, losses = trades[trades["pnl"] > 0], trades[trades["pnl"] <= 0]
        stats.update({
            "win_rate_pct": len(wins) / len(trades) * 100,
            "avg_win_pct": wins["return_pct"].mean() if len(wins) else 0.0,
            "avg_loss_pct": losses["return_pct"].mean() if len(losses) else 0.0,
            "profit_factor": wins["pnl"].sum() / -losses["pnl"].sum() if losses["pnl"].sum() < 0 else float("inf"),
            "took_profit_pct": trades["took_profit"].mean() * 100,
            "avg_days_held": trades["days_held"].mean(),
        })
        held = pd.Series(0, index=equity.index)
        for trade in trades.itertuples():
            held[(held.index >= trade.entry_date) & (held.index < trade.exit_date)] += 1
        stats["avg_positions_held"] = held.mean()
        stats["days_with_no_position_pct"] = (held == 0).mean() * 100
    if benchmark is not None:
        b = benchmark.reindex(equity.index).ffill()
        stats["benchmark_cagr_pct"] = ((b.iloc[-1] / b.iloc[0]) ** (1 / years) - 1) * 100
        stats["benchmark_max_drawdown_pct"] = (b / b.cummax() - 1).min() * 100
    return stats
