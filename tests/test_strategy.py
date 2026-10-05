import numpy as np
import pandas as pd
import pytest

from backtest import signals, strategy


def series(values, start="2024-01-01"):
    return pd.Series(np.asarray(values, dtype=float), index=pd.bdate_range(start, periods=len(values)))


# --- indicators -------------------------------------------------------------

def test_rma_matches_wilder_definition():
    x = series(np.arange(1.0, 31.0))
    out = signals.rma(x, 14)
    assert out.iloc[:13].isna().all()
    assert out.iloc[13] == pytest.approx(np.mean(np.arange(1.0, 15.0)))  # seeded with the SMA
    assert out.iloc[14] == pytest.approx(out.iloc[13] + (15.0 - out.iloc[13]) / 14)


def test_linreg_matches_polyfit():
    rng = np.random.default_rng(0)
    x = series(rng.normal(size=60).cumsum())
    out = signals.linreg(x, 20)
    for i in (19, 35, 59):
        window = x.iloc[i - 19 : i + 1].to_numpy()
        slope, intercept = np.polyfit(np.arange(20), window, 1)
        assert out.iloc[i] == pytest.approx(intercept + slope * 19)


def test_squeeze_counts_consecutive_bars():
    rng = np.random.default_rng(1)
    calm = 100 + rng.normal(scale=0.05, size=60).cumsum()
    close = series(np.r_[calm, calm[-1] * np.linspace(1, 1.5, 20)])  # quiet range, then a breakout
    high, low = close + 1.0, close - 1.0  # wide daily ranges -> Keltner wider than Bollinger while quiet
    sq = signals.squeeze(high, low, close)
    assert sq["squeeze_on"].iloc[40:60].all()
    assert (sq["blue_count"].iloc[40:60].diff().iloc[1:] == 1).all()
    assert not sq["squeeze_on"].iloc[-1] and sq["blue_count"].iloc[-1] == 0


def test_atr_trail_flips():
    # The trend only flips when a close jumps past the previous bar's hl2 +/- 1.2 x ATR.
    close = series([100.0] * 30 + [105.0] * 11 + [99.0] * 5)
    high, low = close + 0.5, close - 0.5
    trail = signals.atr_trail(high, low, close)
    assert (trail["trend"].iloc[:30] == 0).all()
    assert list(np.flatnonzero(trail["atr_buy"])) == [30]  # the jump to 105 flips it up
    assert (trail["trend"].iloc[30:41] == 1).all()
    assert list(np.flatnonzero(trail["atr_exit"])) == [41]  # the drop to 99 flips it down
    assert trail["trend"].iloc[-1] == -1


# --- portfolio simulation ----------------------------------------------------

def make_world(closes, entry_day, exit_flags=None, ema=None):
    """One ticker with a flat open = close, highs 1% above the close."""
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    close = pd.Series(closes, index=dates, dtype=float)
    prices = {"AAA": pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close})}
    sig = pd.DataFrame(index=dates)
    sig["entry_setup"] = [i == entry_day for i in range(len(dates))]
    sig["atr_exit"] = exit_flags if exit_flags is not None else [False] * len(dates)
    sig["ema20"] = ema if ema is not None else close * 0.9
    top = pd.DataFrame({"date": dates, "rank": 1, "ticker": "AAA", "sctr": 99.9, "score": 50.0})
    return dates, prices, {"AAA": sig}, top


def test_buys_next_open_and_exits_on_atr_flip_before_target():
    closes = [100, 100, 101, 102, 99, 98, 98]
    exits = [False, False, False, False, True, False, False]
    dates, prices, sigs, top = make_world(closes, entry_day=1, exit_flags=exits)
    rules = strategy.Rules(cost=0.0)
    equity, trades = strategy.run(dates, prices, sigs, top, rules)
    trade = trades.iloc[0]
    assert trade["entry_date"] == dates[2] and trade["entry_price"] == 101  # next day's open
    assert trade["shares"] == 148  # 15% of $100,000 at $101
    assert trade["exit_date"] == dates[5] and not trade["took_profit"]  # sold at the open after the flip
    assert trade["pnl"] == pytest.approx(148 * (98 - 101))
    assert equity.iloc[-1] == pytest.approx(100_000 + 148 * (98 - 101))


def test_take_profit_then_ema_exit_once_armed():
    #        signal  entry  +8%   above EMA   below EMA -> exit next open
    closes = [100, 100, 100, 109, 112, 111, 105, 104]
    ema = [101, 101, 101, 110, 110, 110, 110, 110]  # 109 < 110 (not armed yet), 112 > 110 arms it
    dates, prices, sigs, top = make_world(closes, entry_day=1, ema=ema)
    equity, trades = strategy.run(dates, prices, sigs, top, strategy.Rules(cost=0.0))
    trade = trades.iloc[0]
    assert trade["took_profit"]
    fills = trade["exits"].split("; ")
    assert fills[0].startswith(f"{dates[3].date()} take-profit 50@109.00")  # gap above +8%: filled at the open
    assert fills[1].startswith(f"{dates[7].date()} exit 100@104.00")  # close < EMA on day 6 -> next open
    assert trade["pnl"] == pytest.approx(50 * 9 + 100 * 4)


def test_ignores_atr_exit_after_take_profit_and_respects_max_positions():
    closes = [100, 100, 100, 110, 111, 112]
    exits = [False, False, False, False, True, False]
    dates, prices, sigs, top = make_world(closes, entry_day=1, exit_flags=exits)
    _, trades = strategy.run(dates, prices, sigs, top, strategy.Rules(cost=0.0))
    assert trades.iloc[0]["exits"].endswith("open at end 100@112.00")  # the ATR flip no longer applies

    # Seven tickers signal on the same day; only six slots: the lowest SCTR is skipped.
    dates, prices, sigs, top = make_world([100] * 5, entry_day=1)
    prices = {f"T{i}": prices["AAA"] for i in range(7)}
    sigs = {f"T{i}": sigs["AAA"] for i in range(7)}
    top = pd.concat([top.assign(ticker=f"T{i}", rank=i + 1, sctr=99.9 - i) for i in range(7)])
    _, trades = strategy.run(dates, prices, sigs, top, strategy.Rules(cost=0.0))
    assert sorted(trades["ticker"]) == [f"T{i}" for i in range(6)]


def test_ema_exit_waits_for_consecutive_closes_below():
    #        signal  entry  +8%  above  below below  above  below below below -> exit next open
    closes = [100, 100, 100, 109, 112, 105, 105, 112, 105, 104, 103, 102]
    ema = [101] * 3 + [110] * 9
    dates, prices, sigs, top = make_world(closes, entry_day=1, ema=ema)
    _, trades = strategy.run(dates, prices, sigs, top, strategy.Rules(cost=0.0, ema_exit_days=3))
    fills = trades.iloc[0]["exits"].split("; ")
    assert fills[1].startswith(f"{dates[11].date()} exit 100@102.00")  # 3rd close below on day 10, sold day 11
