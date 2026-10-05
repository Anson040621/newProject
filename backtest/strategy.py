"""Step 2-4 - the trading rules and a day-by-day portfolio simulation.

Rules (defaults):
  Entry   the stock is in the day's SCTR top 10 (SCTR > 90) AND, on the same
          day, the squeeze is on, the momentum bar is red and the 2x ATR trail
          flips to BUY. Buy at the next day's open.
  Size    15% of account equity per trade, at most 6 positions. When more
          signals than free slots appear, the highest SCTR goes first.
  Exit    before +8%: sell everything at the next open after the ATR trail
                      flips to exit (or, with --atr-exit-on-touch, the moment
                      the price touches the trail line during the day).
          at +8%:     sell 1/3 at the +8% price (or at the open if it gaps above).
          after +8%:  sell the rest at the next open after a close below the
                      20 EMA (or N consecutive closes, --ema-exit-days), once
                      price has closed above the 20 EMA since entry.
  Costs   0.1% of the traded value on every buy and sell (commission + slippage).

Optional rules (off by default):
  --entry-on-touch     buy during the day the moment the price touches the ATR
                       flip level (yesterday's trail line), if yesterday the stock
                       was in the top 10 with the squeeze on and a red bar.
  --emergency-stop X   sell everything the moment the price is X below the entry
                       (e.g. 0.09 = -9%), at any time.
  --reentry-days N     after an exit before +8% (ATR exit or emergency stop), buy
                       again if the ATR trail flips back to BUY within N trading
                       days (no other conditions).

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
    ema_exit_days: int = 1  # consecutive closes below the 20 EMA before selling the rest
    atr_exit_on_touch: bool = False  # before +8%: sell the moment price touches the ATR trail
    entry_on_touch: bool = False  # buy intraday when price touches the ATR flip level
    emergency_stop: float = 0.0  # e.g. 0.09: sell everything at -9% from entry (0 = off)
    reentry_days: int = 0  # re-buy if the ATR flips back to BUY within N days of an early exit (0 = off)
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
    armed: bool = False  # has closed above the 20 EMA since entry
    closes_below_ema: int = 0  # consecutive closes below the 20 EMA
    days_without_price: int = 0
    intraday_entry: bool = False
    reentry: bool = False
    proceeds: float = 0.0  # cash received from sales so far (after costs)
    cost_basis: float = 0.0  # cash paid at entry (incl. costs)
    fills: list = field(default_factory=list)


def run(
    dates: pd.DatetimeIndex,
    prices: dict[str, pd.DataFrame],
    signals: dict[str, pd.DataFrame],
    top: pd.DataFrame,
    rules: Rules = Rules(),
) -> tuple[pd.Series, pd.DataFrame]:
    """Simulate the strategy.

    prices:  ticker -> DataFrame(open, high, low, close) on the full trading calendar
    signals: ticker -> output of signals.ticker_signals on the same calendar
    top:     daily top-N table (date, rank, ticker, sctr)
    Returns the daily equity curve and the trade log.
    """
    top = top[(top["rank"] <= rules.top_n) & (top["sctr"] > rules.min_sctr)]
    top_by_day = {d: g.sort_values("rank") for d, g in top.groupby("date")}

    cash = rules.capital
    positions: dict[str, Position] = {}
    pending_exits: set[str] = set()
    pending_entries: list[tuple[str, float, bool]] = []  # (ticker, sctr, is re-entry), filled at the open
    touch_orders: list[tuple[str, float, bool]] = []  # buy-stop orders at the ATR flip level, for today
    early_exits: dict[str, tuple[int, float]] = {}  # ticker -> (day number of an exit before +8%, sctr)
    last_close: dict[str, float] = {}
    equity = pd.Series(np.nan, index=dates)
    trades = []

    def px(ticker, day, field_):
        value = prices[ticker].at[day, field_]
        return None if pd.isna(value) else float(value)

    def sell(pos: Position, shares: int, price: float, day, reason: str):
        nonlocal cash
        value = shares * price
        cash += value * (1 - rules.cost)
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
            "exit_date": day,
            "exits": "; ".join(f"{d.date()} {r} {s}@{p:.2f}" for d, r, s, p in pos.fills),
            "took_profit": pos.tp_taken,
            "pnl": pos.proceeds - pos.cost_basis,
            "return_pct": (pos.proceeds / pos.cost_basis - 1) * 100,
            "days_held": (day - pos.entry_date).days,
        })
        del positions[pos.ticker]
        if not pos.tp_taken and rules.reentry_days:
            early_exits[pos.ticker] = (day_number, pos.sctr)

    def buy(ticker, price, day, sctr, reentry, intraday) -> bool:
        nonlocal cash
        if len(positions) >= rules.max_positions or ticker in positions:
            return False
        shares = math.floor(rules.position_size * prev_equity / (price * (1 + rules.cost)))
        if shares <= 0 or shares * price * (1 + rules.cost) > cash:
            return False
        paid = shares * price * (1 + rules.cost)
        cash -= paid
        positions[ticker] = Position(ticker, day, price, shares, shares, sctr, cost_basis=paid,
                                     intraday_entry=intraday, reentry=reentry)
        early_exits.pop(ticker, None)
        return True

    prev_equity = cash
    for day_number, day in enumerate(dates):
        # 1. Orders from yesterday's close are filled at today's open.
        for ticker in sorted(pending_exits):
            pos = positions.get(ticker)
            price = px(ticker, day, "open") or px(ticker, day, "close")
            if pos is not None and price is not None:
                sell(pos, pos.shares, price, day, "exit")
                close_out(pos, day)
        pending_exits.clear()

        for ticker, sctr, reentry in pending_entries:
            price = px(ticker, day, "open") or px(ticker, day, "close")
            if price is not None:
                buy(ticker, price, day, sctr, reentry, intraday=False)
        pending_entries = []

        # 1b. Buy-stop orders at the ATR flip level (--entry-on-touch): filled when the
        #     high reaches yesterday's trail line, at the open if it gaps above.
        for ticker, sctr, reentry in touch_orders:
            level = signals[ticker].at[day, "flip_level"]
            high, open_ = px(ticker, day, "high"), px(ticker, day, "open")
            if pd.isna(level) or high is None or high < level:
                continue
            buy(ticker, max(level, open_) if open_ is not None else level, day, sctr, reentry, intraday=True)
        touch_orders = []

        # 2. Emergency stop: sell everything at -X% from the entry (at the open if it gaps below).
        if rules.emergency_stop:
            for pos in list(positions.values()):
                if pos.intraday_entry and pos.entry_date == day:
                    continue  # bought during the day: we can't tell whether the low came first
                stop = pos.entry_price * (1 - rules.emergency_stop)
                low, open_ = px(pos.ticker, day, "low"), px(pos.ticker, day, "open")
                if low is None or low > stop:
                    continue
                fill = min(stop, open_) if open_ is not None and day != pos.entry_date else stop
                sell(pos, pos.shares, fill, day, "emergency-stop")
                close_out(pos, day)

        # 2a. Intraday ATR stop (optional): before +8%, sell everything the moment the
        #     price touches yesterday's trail line (at the open if it gaps below).
        if rules.atr_exit_on_touch:
            for pos in list(positions.values()):
                stop = signals[pos.ticker].at[day, "atr_stop"]
                low, open_ = px(pos.ticker, day, "low"), px(pos.ticker, day, "open")
                if pos.tp_taken or pd.isna(stop) or low is None or low > stop:
                    continue
                fill = min(stop, open_) if open_ is not None and day != pos.entry_date else stop
                sell(pos, pos.shares, fill, day, "atr-stop")
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

        # 3. At the close: exit signals for open positions.
        for pos in list(positions.values()):
            sig = signals[pos.ticker]
            close, ema20 = px(pos.ticker, day, "close"), sig.at[day, "ema20"]
            if close is None:
                # Delisted or halted: after 10 trading days without a price, close at the last price.
                pos.days_without_price += 1
                if pos.days_without_price > 10:
                    sell(pos, pos.shares, last_close.get(pos.ticker, pos.entry_price), day, "no price")
                    close_out(pos, day)
                    pending_exits.discard(pos.ticker)
                continue
            pos.days_without_price = 0
            last_close[pos.ticker] = close
            pos.closes_below_ema = pos.closes_below_ema + 1 if close < ema20 else 0
            if not pos.tp_taken:
                if not rules.atr_exit_on_touch and bool(sig.at[day, "atr_exit"]):
                    pending_exits.add(pos.ticker)
            elif pos.armed and pos.closes_below_ema >= rules.ema_exit_days:
                pending_exits.add(pos.ticker)
            if close > ema20:
                pos.armed = True

        # 4. At the close: new entry signals among today's top 10.
        today = top_by_day.get(day)
        candidates = []
        if today is not None:
            for row in today.itertuples():
                sig = signals.get(row.ticker)
                if sig is None or row.ticker in positions:
                    continue
                if not rules.entry_on_touch and bool(sig.at[day, "entry_setup"]):
                    candidates.append((row.ticker, row.sctr, False))
                elif rules.entry_on_touch and bool(sig.at[day, "setup"]) and sig.at[day, "trend"] != 1:
                    candidates.append((row.ticker, row.sctr, False))  # order for tomorrow at the flip level

        # 5. Re-entry after an early exit: the ATR trail flips back to BUY within N days.
        for ticker, (exit_day, sctr) in list(early_exits.items()):
            sig = signals[ticker]
            if ticker in positions:
                continue
            if rules.entry_on_touch:
                if day_number + 1 - exit_day > rules.reentry_days:
                    del early_exits[ticker]
                elif sig.at[day, "trend"] != 1:
                    candidates.insert(0, (ticker, sctr, True))
            elif day_number - exit_day > rules.reentry_days:
                del early_exits[ticker]
            elif bool(sig.at[day, "atr_buy"]):
                candidates.insert(0, (ticker, sctr, True))

        seen = set()
        for order in candidates:
            if order[0] not in seen:
                seen.add(order[0])
                (touch_orders if rules.entry_on_touch else pending_entries).append(order)

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
