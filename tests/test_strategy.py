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

def basic_rules(**overrides):
    """Next-open entries, 1-day EMA exit, no re-entry; each test switches on what it checks."""
    settings = dict(entry_on_touch=False, ema_exit_days=1, reentry_days=0, atr_exit_on_touch=False,
                    atr_stop_buffer=0.0, sticky_stop=False, close_exit=False, entry_buffer=0.0, emergency_stop=0.0,
                    breakeven_exit=False, rest_exit="ema20", failed_breakout_exit=False)
    settings.update(overrides)
    return strategy.Rules(**settings)


def test_default_rules_are_version_19():
    rules = strategy.Rules()
    assert (rules.top_n, rules.position_size, rules.max_positions, rules.take_profit) == (10, 0.15, 6, 0.08)
    assert (rules.entry_on_touch, rules.entry_buffer, rules.ema_exit_days, rules.reentry_days) == (True, 0.0, 3, 5)
    assert (rules.atr_exit_on_touch, rules.atr_stop_buffer, rules.close_exit) == (True, 0.02, True)
    assert not rules.sticky_stop and rules.emergency_stop == 0.0 and not rules.rebuy_shakeouts
    assert rules.breakeven_exit and (rules.rest_exit, rules.sma_exit_buffer) == ("sma50", 0.03)
    assert not rules.failed_breakout_exit


def make_world(closes, entry_day, exit_flags=None, ema=None):
    """One ticker with a flat open = close, highs 1% above the close."""
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    close = pd.Series(closes, index=dates, dtype=float)
    prices = {"AAA": pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close})}
    sig = pd.DataFrame(index=dates)
    sig["entry_setup"] = [i == entry_day for i in range(len(dates))]
    sig["atr_exit"] = exit_flags if exit_flags is not None else [False] * len(dates)
    sig["ema20"] = ema if ema is not None else close * 0.9
    sig["sma50"] = np.nan
    sig["atr_stop"] = np.nan
    sig["setup"] = sig["entry_setup"]
    sig["atr_buy"] = sig["entry_setup"]
    sig["trend"] = 1
    sig["flip_level"] = np.nan
    top = pd.DataFrame({"date": dates, "rank": 1, "ticker": "AAA", "sctr": 99.9, "score": 50.0})
    return dates, prices, {"AAA": sig}, top


def test_buys_next_open_and_exits_on_atr_flip_before_target():
    closes = [100, 100, 101, 102, 99, 98, 98]
    exits = [False, False, False, False, True, False, False]
    dates, prices, sigs, top = make_world(closes, entry_day=1, exit_flags=exits)
    rules = basic_rules(cost=0.0)
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
    equity, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0))
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
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0))
    assert trades.iloc[0]["exits"].endswith("open at end 100@112.00")  # the ATR flip no longer applies

    # Seven tickers signal on the same day; only six slots: the lowest SCTR is skipped.
    dates, prices, sigs, top = make_world([100] * 5, entry_day=1)
    prices = {f"T{i}": prices["AAA"] for i in range(7)}
    sigs = {f"T{i}": sigs["AAA"] for i in range(7)}
    top = pd.concat([top.assign(ticker=f"T{i}", rank=i + 1, sctr=99.9 - i) for i in range(7)])
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0))
    assert sorted(trades["ticker"]) == [f"T{i}" for i in range(6)]


def test_ema_exit_waits_for_consecutive_closes_below():
    #        signal  entry  +8%  above  below below  above  below below below -> exit next open
    closes = [100, 100, 100, 109, 112, 105, 105, 112, 105, 104, 103, 102]
    ema = [101] * 3 + [110] * 9
    dates, prices, sigs, top = make_world(closes, entry_day=1, ema=ema)
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, ema_exit_days=3))
    fills = trades.iloc[0]["exits"].split("; ")
    assert fills[1].startswith(f"{dates[11].date()} exit 100@102.00")  # 3rd close below on day 10, sold day 11


def test_entry_without_squeeze_requirement():
    close = series([100.0] * 30 + [105.0] * 5)
    high, low = close + 0.5, close - 0.5
    strict = signals.ticker_signals(high, low, close)
    loose = signals.ticker_signals(high, low, close, require_squeeze=False)
    expected = (loose["momentum"] < 0) & loose["atr_buy"]
    assert (loose["entry_setup"] == expected).all()
    assert (strict["entry_setup"] == expected & strict["squeeze_on"]).all()


def test_atr_stop_on_touch_sells_intraday():
    closes = [100, 100, 100, 99, 99, 99]
    dates, prices, sigs, top = make_world(closes, entry_day=1)
    sigs["AAA"]["atr_stop"] = [np.nan, np.nan, 95.0, 98.5, 98.5, 98.5]  # day 3 low (98.01) touches 98.5
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, atr_exit_on_touch=True))
    trade = trades.iloc[0]
    assert trade["exit_date"] == dates[3]
    assert trade["exits"] == f"{dates[3].date()} atr-stop 150@98.50"
    # A gap below the line fills at the open instead.
    prices["AAA"].loc[dates[3], ["open", "low"]] = [97.0, 96.0]
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, atr_exit_on_touch=True))
    assert trades.iloc[0]["exits"].endswith("atr-stop 150@97.00")


def test_ema_exit_never_applies_before_take_profit():
    # Price closes above the EMA, then below it for days, but never reaches +8%:
    # with the intraday ATR stop mode only the ATR stop may close the trade.
    closes = [100, 100, 100, 102, 99, 98, 97, 96]
    ema = [101, 101, 99, 99, 101, 101, 101, 101]
    dates, prices, sigs, top = make_world(closes, entry_day=1, ema=ema)
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, atr_exit_on_touch=True))
    assert "open at end" in trades.iloc[0]["exits"]


def test_emergency_stop_sells_at_minus_9_percent():
    closes = [100, 100, 100, 95, 93, 93]
    dates, prices, sigs, top = make_world(closes, entry_day=1)
    prices["AAA"].loc[dates[4], "low"] = 90.0  # touches 100 x 0.91 = 91 during the day
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, emergency_stop=0.09))
    assert trades.iloc[0]["exits"] == f"{dates[4].date()} emergency-stop 150@91.00"


def test_reentry_when_atr_flips_back_within_window():
    closes = [100, 100, 100, 97, 96, 99, 101, 102, 103, 104]
    exits = [False, False, False, True, False, False, False, False, False, False]
    dates, prices, sigs, top = make_world(closes, entry_day=1, exit_flags=exits)
    sigs["AAA"]["atr_buy"] = [False] * 6 + [True] + [False] * 3  # flips back on day 6
    rules = basic_rules(cost=0.0, reentry_days=5)
    _, trades = strategy.run(dates, prices, sigs, top, rules)
    assert list(trades["entry_date"]) == [dates[2], dates[7]]  # exit day 4, flip day 6 -> buy day 7
    assert list(trades["reentry"]) == [False, True]
    # The same flip outside the window is ignored.
    sigs["AAA"]["atr_buy"] = [False] * 9 + [True]
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, reentry_days=3))
    assert len(trades) == 1


def test_entry_on_touch_buys_at_flip_level():
    closes = [100, 100, 100, 104, 105, 106]
    dates, prices, sigs, top = make_world(closes, entry_day=-1)
    sig = sigs["AAA"]
    sig["setup"] = [False, True, False, False, False, False]  # squeeze + red bar on day 1
    sig["trend"] = [-1, -1, 1, 1, 1, 1]
    sig["flip_level"] = [np.nan, np.nan, 100.5, np.nan, np.nan, np.nan]  # day 1's down-trend trail
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, entry_on_touch=True))
    trade = trades.iloc[0]
    assert trade["entry_date"] == dates[2] and trade["entry_price"] == 100.5  # day 2 high 101 touches 100.5


def test_idle_money_parked_in_etf():
    closes = [100.0] * 6
    dates, prices, sigs, top = make_world(closes, entry_day=-1)  # never trades
    park = pd.DataFrame({"open": [10, 10.5, 11, 11, 12, 12], "close": [10, 11, 11, 12, 12, 13]}, index=dates, dtype=float)
    equity, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, park_cost=0.0), park=park)
    assert trades.empty
    assert equity.iloc[-1] == pytest.approx(100_000 * 13 / 10)  # fully in the ETF the whole time

    # Buying a stock takes the money out of the ETF; it stops earning the ETF's return.
    dates, prices, sigs, top = make_world([100.0] * 6, entry_day=1)
    equity, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, park_cost=0.0), park=park)
    # Day 2: the ETF opens at 11 (account $110,000), buy 15% = 165 shares at 100.
    in_etf_after_buy = 100_000 * 11 / 10 - 165 * 100
    assert trades.iloc[0]["shares"] == 165
    assert equity.iloc[-1] == pytest.approx(in_etf_after_buy * 13 / 11 + 165 * 100)


def test_atr_stop_buffer_and_rebuy_after_shakeout():
    closes = [100, 100, 100, 99, 100, 101, 101]
    dates, prices, sigs, top = make_world(closes, entry_day=1)
    sigs["AAA"]["atr_stop"] = [np.nan, np.nan, 95.0, 99.0, 97.0, 97.0, 97.0]
    prices["AAA"].loc[dates[3], "low"] = 97.5  # dips 1.5% below the line during the day
    # Without a buffer the dip sells; with a 3% buffer (stop 96.03) it does not.
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, atr_exit_on_touch=True))
    assert "atr-stop 150@99.00" in trades.iloc[0]["exits"]
    _, trades = strategy.run(dates, prices, sigs, top,
                             basic_rules(cost=0.0, atr_exit_on_touch=True, atr_stop_buffer=0.03))
    assert len(trades) == 1 and "open at end" in trades.iloc[0]["exits"]
    # Shaken out but the trend is still up at the close -> buy back at the next open.
    rules = basic_rules(cost=0.0, atr_exit_on_touch=True, reentry_days=5, rebuy_shakeouts=True)
    _, trades = strategy.run(dates, prices, sigs, top, rules)
    assert list(trades["entry_date"]) == [dates[2], dates[4]]
    assert list(trades["reentry"]) == [False, True]


def test_close_below_line_still_sells_with_buffered_stop():
    # EPAM, Sep 2022: the close falls below the ATR line but stays above the
    # buffered intraday stop -> sell at the next open anyway.
    closes = [100, 100, 100, 98, 95, 90]
    exits = [False, False, False, True, False, False]
    dates, prices, sigs, top = make_world(closes, entry_day=1, exit_flags=exits)
    sigs["AAA"]["atr_stop"] = [np.nan, np.nan, 99.0, 99.0, np.nan, np.nan]  # stop 3% below = 96.03
    rules = basic_rules(cost=0.0, atr_exit_on_touch=True, atr_stop_buffer=0.03, close_exit=True)
    _, trades = strategy.run(dates, prices, sigs, top, rules)
    assert trades.iloc[0]["exits"] == f"{dates[4].date()} exit 150@95.00"


def test_without_close_exit_the_sticky_stop_still_protects():
    # EPAM-like: the close falls below the line (trend flips down, the green line
    # disappears) without touching the 3% stop. With the default rules nothing sells
    # on the close, but the stop stays at its last level and sells on the way down.
    closes = [100, 100, 100, 98, 97, 94, 90]
    exits = [False, False, False, True, False, False, False]
    dates, prices, sigs, top = make_world(closes, entry_day=1, exit_flags=exits)
    sigs["AAA"]["atr_stop"] = [np.nan, np.nan, 99.0, 99.0, np.nan, np.nan, np.nan]
    rules = basic_rules(cost=0.0, atr_exit_on_touch=True, atr_stop_buffer=0.03, sticky_stop=True)
    _, trades = strategy.run(dates, prices, sigs, top, rules)
    assert trades.iloc[0]["exits"] == f"{dates[4].date()} atr-stop 150@96.03"  # 99 x 0.97; day 4 low 96.03
    # Without the sticky stop there is no protection once the line is gone.
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, atr_exit_on_touch=True, atr_stop_buffer=0.03))
    assert "open at end" in trades.iloc[0]["exits"]


def test_sticky_stop_never_moves_down():
    closes = [100, 100, 100, 101, 101, 101, 101]
    dates, prices, sigs, top = make_world(closes, entry_day=1)
    sigs["AAA"]["atr_stop"] = [np.nan, np.nan, 95.0, 97.0, 90.0, np.nan, np.nan]  # line drops to 90 on day 4
    prices["AAA"].loc[dates[5], "low"] = 94.0  # below 97 x 0.97 = 94.09
    rules = basic_rules(cost=0.0, atr_exit_on_touch=True, atr_stop_buffer=0.03, sticky_stop=True)
    _, trades = strategy.run(dates, prices, sigs, top, rules)
    assert trades.iloc[0]["exits"] == f"{dates[5].date()} atr-stop 150@94.09"


def test_entry_buffer_above_flip_level():
    closes = [100, 100, 100, 104, 105, 106]
    dates, prices, sigs, top = make_world(closes, entry_day=-1)
    sig = sigs["AAA"]
    sig["setup"] = [False, True, False, False, False, False]
    sig["trend"] = [-1, -1, 1, 1, 1, 1]
    sig["flip_level"] = [np.nan, np.nan, 98.0, np.nan, np.nan, np.nan]
    # 2% above the flip level = 99.96; day 2 high is 101 -> filled at 99.96 (open 100 is above it -> at the open)
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, entry_on_touch=True, entry_buffer=0.02))
    assert trades.iloc[0]["entry_price"] == 100.0
    sig["flip_level"] = [np.nan, np.nan, 99.0, np.nan, np.nan, np.nan]  # 2% above = 100.98 < high 101
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, entry_on_touch=True, entry_buffer=0.02))
    assert trades.iloc[0]["entry_price"] == pytest.approx(100.98)
    sig["flip_level"] = [np.nan, np.nan, 99.5, np.nan, np.nan, np.nan]  # 2% above = 101.49 > high 101: no fill
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, entry_on_touch=True, entry_buffer=0.02))
    assert trades.empty


def test_hard_stop_on_intraday_entry_day_needs_a_close_below():
    closes = [100, 100, 85, 85, 85]
    dates, prices, sigs, top = make_world(closes, entry_day=-1)
    sig = sigs["AAA"]
    sig["setup"] = [True, False, False, False, False]
    sig["trend"] = [-1, 1, 1, 1, 1]
    sig["flip_level"] = [np.nan, 99.0, np.nan, np.nan, np.nan]
    # Day 1: opens at 100, above the 99 level -> bought at the open; the close (100) is above
    # the 12% stop (88) -> kept, although the day's low (80) is below it.
    prices["AAA"].loc[dates[1], "low"] = 80.0  # this low may have come before the purchase
    rules = basic_rules(cost=0.0, entry_on_touch=True, emergency_stop=0.12)
    _, trades = strategy.run(dates, prices, sigs, top, rules)
    assert trades.iloc[0]["exits"] == f"{dates[2].date()} emergency-stop 150@85.00"  # day 2 gaps below: at the open
    # Same day crash below the stop at the close -> stopped on the entry day at the stop price.
    prices["AAA"].loc[dates[1], "close"] = 86.0
    _, trades = strategy.run(dates, prices, sigs, top, rules)
    assert trades.iloc[0]["exits"] == f"{dates[1].date()} emergency-stop 150@88.00"


def test_breakeven_exit_after_take_profit():
    # MAAS-like: +8% hit, the price never closes above the 20 EMA, then falls back to the entry.
    closes = [100, 100, 100, 109, 104, 99.5, 90, 80]
    ema = [101, 101, 101, 112, 112, 112, 112, 112]  # never closes above it -> the EMA exit never arms
    dates, prices, sigs, top = make_world(closes, entry_day=1, ema=ema)
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, breakeven_exit=True))
    fills = trades.iloc[0]["exits"].split("; ")
    assert fills[1] == f"{dates[6].date()} breakeven 100@90.00"  # close 99.5 <= 100 on day 5 -> next open
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0))
    assert "open at end" in trades.iloc[0]["exits"]  # without it: held all the way down


def test_rest_exit_on_close_3pct_below_50_sma():
    #        signal entry +8%  above  dips 2% below SMA  3.5% below -> sell next open
    closes = [100, 100, 100, 109, 112, 107.8, 106.15, 106]
    dates, prices, sigs, top = make_world(closes, entry_day=1)
    sigs["AAA"]["sma50"] = 110.0  # level = 110 x 0.97 = 106.70
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, rest_exit="sma50", sma_exit_buffer=0.03))
    fills = trades.iloc[0]["exits"].split("; ")
    assert fills[1] == f"{dates[7].date()} exit 100@106.00"  # close 106.15 < 106.70 on day 6


def test_failed_breakout_sells_next_open():
    # COIN, Apr 2024: bought on the touch of the flip level, the day closes back below it.
    closes = [100, 100, 99, 95, 90, 85]
    dates, prices, sigs, top = make_world(closes, entry_day=-1)
    sig = sigs["AAA"]
    sig["setup"] = [True, False, False, False, False, False]
    sig["trend"] = [-1, -1, -1, -1, -1, -1]  # never flips up
    sig["flip_level"] = [np.nan, 100.5, 100.5, 100.5, 100.5, 100.5]  # day 1 high 101 touches it
    rules = basic_rules(cost=0.0, entry_on_touch=True, failed_breakout_exit=True)
    _, trades = strategy.run(dates, prices, sigs, top, rules)
    assert trades.iloc[0]["exits"] == f"{dates[2].date()} failed-breakout 149@99.00"
    # A confirmed breakout (close above the level, trend flips up) is kept.
    sig["trend"] = [-1, 1, 1, 1, 1, 1]
    _, trades = strategy.run(dates, prices, sigs, top, rules)
    assert "failed-breakout" not in trades.iloc[0]["exits"]


def test_dim_green_follow_up_entry_within_window():
    # Bought on the full setup (day 2), sold at the open of day 5; on day 6 the same
    # setup shows up with a dim green momentum bar instead of red.
    closes = [100, 100, 100, 101, 99, 98, 99, 100, 101, 102]
    exits = [False, False, False, True, False, False, False, False, False, False]
    dates, prices, sigs, top = make_world(closes, entry_day=1, exit_flags=exits)
    sigs["AAA"]["setup_dim_green"] = [False] * 6 + [True] + [False] * 3
    sigs["AAA"]["atr_buy"] = [False, True] + [False] * 4 + [True] + [False] * 3
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, dim_green_days=42))
    assert list(trades["entry_date"]) == [dates[2], dates[7]]
    assert list(trades["entry_kind"]) == ["setup", "dim-green"]
    # Off by default, and ignored once the setup entry is older than the window.
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0))
    assert len(trades) == 1
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, dim_green_days=3))
    assert len(trades) == 1


def test_close_entry_buys_next_open_when_setup_shows_on_the_buy_bar():
    # HOOD, 9 Apr 2025: the first blue cross, a red bar and the ATR flip to BUY all
    # appear on the same day, so no buy-stop was waiting from the day before.
    closes = [100, 100, 100, 104, 105, 106]
    dates, prices, sigs, top = make_world(closes, entry_day=2)  # setup + atr_buy on day 2
    rules = basic_rules(cost=0.0, entry_on_touch=True)
    _, trades = strategy.run(dates, prices, sigs, top, rules)
    assert trades.empty  # the trend is already up at the close: no flip-level order
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, entry_on_touch=True, close_entry=True))
    trade = trades.iloc[0]
    assert trade["entry_date"] == dates[3] and trade["entry_price"] == 104  # next day's open
    assert trade["entry_kind"] == "setup-close"
