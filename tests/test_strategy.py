import numpy as np
import pandas as pd
import pytest

from backtest import signals, strategy, twins


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
                    breakeven_exit=False, breakeven_buffer=0.0, rest_exit="ema20", failed_breakout_exit=False,
                    rest_exit_delay=0, max_vol=0, reentry_after_tp=False, climax_gain=0.0, climax_body=0.0,
                    reentry_max_losses=0)
    settings.update(overrides)
    return strategy.Rules(**settings)


def test_default_rules_are_version_26():
    rules = strategy.Rules()
    assert (rules.top_n, rules.position_size, rules.max_positions, rules.take_profit) == (10, 0.15, 6, 0.08)
    assert (rules.entry_on_touch, rules.entry_buffer, rules.ema_exit_days, rules.reentry_days) == (True, 0.0, 3, 5)
    assert (rules.atr_exit_on_touch, rules.atr_stop_buffer, rules.close_exit) == (True, 0.02, True)
    assert not rules.sticky_stop and rules.emergency_stop == 0.0 and not rules.rebuy_shakeouts
    assert not rules.breakeven_exit and (rules.rest_exit, rules.sma_exit_buffer) == ("sma50", 0.03)
    assert not rules.failed_breakout_exit
    assert rules.breakeven_buffer == 0.03 and rules.entry_check == "prev" and rules.rest_exit_delay == 21
    assert (rules.min_vol, rules.max_vol) == (0, 75) and rules.reentry_after_tp and not rules.red_line_stop
    assert (rules.climax_gain, rules.climax_stretch, rules.climax_ma, rules.climax_gap, rules.climax_days) == (0.40, 0.50, 100, 0.05, 10)
    assert not rules.climax_reentry and not rules.market_ma
    assert (rules.reentry_max_losses, rules.reentry_win) == (3, 0.20) and not rules.profit_floor and not rules.list_memory and not rules.late_entry_days
    assert not rules.twin_days
    assert (rules.take_profit_fraction, rules.take_profit2) == (1 / 3, 0.0)
    assert (rules.climax_body, rules.climax_drop, rules.climax_drop_atr) == (0.06, 0.0, 0.0)


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


def test_second_partial_sale_and_nothing_sold_at_the_first_target():
    closes = [100, 100, 100, 109, 126, 127]
    exits = [False, False, False, False, True, False]  # a close below the green line after +8%: ignored
    dates, prices, sigs, top = make_world(closes, entry_day=1, exit_flags=exits)
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, take_profit2=0.25))
    assert trades.iloc[0]["exits"].split("; ")[:2] == [f"{dates[3].date()} take-profit 50@109.00",  # 1/3 of 150
                                                        f"{dates[4].date()} take-profit-2 38@126.00"]  # 1/4 of 150
    # Nothing sold at +8%, but the green line no longer sells after it: the rest exit has taken over.
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, take_profit_fraction=0.0, take_profit2=0.25))
    assert trades.iloc[0]["exits"] == f"{dates[4].date()} take-profit-2 38@126.00; {dates[5].date()} open at end 112@127.00"
    assert trades.iloc[0]["took_profit"]


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


def test_breakeven_exit_waits_for_a_close_3pct_below_entry():
    closes = [100, 100, 100, 109, 104, 99.5, 90, 80]
    ema = [101, 101, 101, 112, 112, 112, 112, 112]  # the rest-exit line never arms
    dates, prices, sigs, top = make_world(closes, entry_day=1, ema=ema)
    rules = basic_rules(cost=0.0, breakeven_exit=True, breakeven_buffer=0.03)
    _, trades = strategy.run(dates, prices, sigs, top, rules)
    # 99.5 is only 0.5% below the entry: hold. 90 is 10% below: sell at the next open.
    assert trades.iloc[0]["exits"].split("; ")[1] == f"{dates[7].date()} breakeven 100@80.00"


def test_live_entry_buys_where_the_squeeze_shows_on_the_way_up():
    closes = [100, 100, 100, 104, 105, 106]
    dates, prices, sigs, top = make_world(closes, entry_day=-1)
    sig = sigs["AAA"]
    sig["setup"] = False  # no squeeze / red bar at any close before the BUY day
    sig["trend"] = [-1, -1, 1, 1, 1, 1]
    sig["flip_level"] = [np.nan, np.nan, 100.5, np.nan, np.nan, np.nan]
    sig["touch_price"] = [np.nan, np.nan, 100.8, np.nan, np.nan, np.nan]  # squeeze shows at 100.8
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, entry_on_touch=True))
    assert trades.empty  # old rule: no setup at the previous close, no order
    rules = basic_rules(cost=0.0, entry_on_touch=True, entry_check="live")
    _, trades = strategy.run(dates, prices, sigs, top, rules)
    trade = trades.iloc[0]
    assert (trade["entry_date"], trade["entry_price"], trade["entry_kind"]) == (dates[2], 100.8, "setup-live")
    sig["touch_price"] = np.nan  # the squeeze never shows that day: no buy
    _, trades = strategy.run(dates, prices, sigs, top, rules)
    assert trades.empty


def test_live_squeeze_reading_matches_the_end_of_day_bar():
    # Walking the price up to the real close (low = open) must give the end-of-day values.
    rng = np.random.default_rng(3)
    close = series(100 + rng.normal(scale=1.0, size=80).cumsum())
    open_ = close.shift(1).fillna(close.iloc[0]) + rng.normal(scale=0.3, size=80)
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    day = close.index[60]
    open_[day] = close[day] - 2.0
    high[day], low[day] = close[day], open_[day]
    level = pd.Series(np.nan, index=close.index)
    level[day] = close[day]
    live = signals.at_touch(open_, high, low, close, level).loc[day]
    full = signals.squeeze(high, low, close).loc[day]
    assert bool(live["touch_squeeze"]) == bool(full["squeeze_on"])
    assert live["touch_momentum"] == pytest.approx(full["momentum"])


def test_reentry_after_the_rest_was_sold_after_take_profit():
    #        signal  entry  +8%   above EMA   below EMA -> exit day 7   flip back on day 9
    closes = [100, 100, 100, 109, 112, 111, 105, 104, 106, 108, 110, 111]
    ema = [101, 101, 101] + [110] * 9
    dates, prices, sigs, top = make_world(closes, entry_day=1, ema=ema)
    sigs["AAA"]["atr_buy"] = [False] * 9 + [True] + [False] * 2
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, reentry_days=5))
    assert len(trades) == 1 and trades.iloc[0]["took_profit"]  # no re-entry after a +8% trade
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, reentry_days=5, reentry_after_tp=True))
    assert list(trades["entry_date"]) == [dates[2], dates[10]] and list(trades["reentry"]) == [False, True]


def test_red_line_stop_sells_a_false_breakout():
    closes = [100, 100, 100, 99, 95, 96]
    dates, prices, sigs, top = make_world(closes, entry_day=-1)
    sig = sigs["AAA"]
    sig["setup"] = [False, True, False, False, False, False]  # squeeze + red bar on day 1
    sig["trend"] = -1  # the buy day closes back below the line: no green line
    sig["flip_level"] = [np.nan, np.nan, 100.5, 100.5, 100.5, 100.5]
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, entry_on_touch=True))
    assert len(trades) == 1 and "red-line-stop" not in trades.iloc[0]["exits"]  # held without the rule
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, entry_on_touch=True, red_line_stop=0.03))
    trade = trades.iloc[0]
    assert trade["entry_date"] == dates[2] and trade["entry_price"] == 100.5
    # Day 3's low (98.01) stays above 97.49 (3% below 100.5); day 4 opens at 95, below the stop.
    assert trade["exits"] == f"{dates[4].date()} red-line-stop 149@95.00"


def test_climax_exit_after_a_steep_run_and_a_gap_down():
    #         signal entry  +20%  +40%  +50%  gap down -> sold next open; flip back later: no re-entry
    closes = [100, 100, 100, 120, 140, 150, 142, 141, 141, 143]
    dates, prices, sigs, top = make_world(closes, entry_day=1, ema=[90] * 10)
    sigs["AAA"]["sma50"] = 100.0  # 140 and 150 are 30%+ above it: a stretched, compressed run
    sigs["AAA"]["atr_buy"] = [False, True] + [False] * 6 + [True, False]
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, reentry_days=5, reentry_after_tp=True))
    assert len(trades) == 1 and "climax-exit" not in trades.iloc[0]["exits"]  # without the rule: held
    rules = basic_rules(cost=0.0, reentry_days=5, reentry_after_tp=True, climax_gain=0.40, climax_stretch=0.30,
                        climax_ma=50, climax_gap=0.0)
    _, trades = strategy.run(dates, prices, sigs, top, rules)
    assert len(trades) == 1  # no re-entry after the climax exit
    # Day 6 opens at 142, below day 5's low (148.5): sold at day 7's open.
    assert trades.iloc[0]["exits"].endswith(f"{dates[7].date()} climax-exit 100@141.00")
    # A gap of at least 5% is required: 142 is only 4.4% below 148.5, so no climax exit.
    rules = basic_rules(cost=0.0, climax_gain=0.40, climax_stretch=0.30, climax_ma=50, climax_gap=0.05)
    _, trades = strategy.run(dates, prices, sigs, top, rules)
    assert "climax-exit" not in trades.iloc[0]["exits"]
    # ... but a big red candle also counts: day 6 closes 5.3% below day 5's close (150 -> 142).
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, climax_gain=0.40, climax_stretch=0.30,
                                                                  climax_ma=50, climax_gap=0.05, climax_drop=0.05))
    assert trades.iloc[0]["exits"].endswith(f"{dates[7].date()} climax-exit 100@141.00")


def test_market_filter_blocks_new_buys():
    dates, prices, sigs, top = make_world([100, 100, 101, 102, 103], entry_day=1)
    off = pd.Series([True, False, True, True, True], index=dates)  # SPY or QQQ below average on day 1
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, market_ma=200), market=off)
    assert trades.empty
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, market_ma=200), market=off | True)
    assert len(trades) == 1
    with pytest.raises(ValueError):
        strategy.run(dates, prices, sigs, top, basic_rules(market_ma=200))


def test_volatility_limit_needs_the_vol_column():
    dates, prices, sigs, top = make_world([100] * 5, entry_day=1)
    with pytest.raises(ValueError):
        strategy.run(dates, prices, sigs, top, basic_rules(max_vol=75))


def test_reentry_only_while_in_the_top_n():
    closes = [100, 100, 100, 97, 96, 99, 101, 102, 103, 104]
    exits = [False, False, False, True, False, False, False, False, False, False]
    dates, prices, sigs, top = make_world(closes, entry_day=1, exit_flags=exits)
    sigs["AAA"]["atr_buy"] = [False] * 6 + [True] + [False] * 3  # flips back on day 6
    top.loc[6, "rank"] = 15  # ranked #15 on the evening of the flip
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, reentry_days=5))
    assert len(trades) == 2  # no requirement: re-bought on day 7
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, reentry_days=5, reentry_top=10))
    assert len(trades) == 1  # outside the top 10 that evening: no re-entry
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, reentry_days=5, reentry_top=20))
    assert list(trades["entry_date"]) == [dates[2], dates[7]]


def test_reentry_stops_after_three_trades_in_a_row_without_a_win():
    # Each exit flag sells at the next open; each flip back to BUY re-buys at the next open.
    closes = [100] * 26
    dates, prices, sigs, top = make_world(closes, entry_day=1, exit_flags=[d in (3, 7, 11, 15, 19, 23) for d in range(26)])
    sigs["AAA"]["atr_buy"] = [d in (5, 9, 13, 21) for d in range(26)]
    sigs["AAA"]["entry_setup"] = [d in (1, 17) for d in range(26)]
    sigs["AAA"]["setup"] = sigs["AAA"]["entry_setup"]
    run = lambda **kw: strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, reentry_days=5, **kw))[1]
    assert list(run()["entry_kind"]) == ["setup", "reentry", "reentry", "reentry", "setup", "reentry"]
    # The setup buy and two re-buys all end flat (not above +20%): the re-buy on day 14 is skipped.
    # The new setup buy on day 18 starts the count again, so its re-buy on day 22 goes ahead.
    trades = run(reentry_max_losses=3)
    assert list(trades["entry_date"]) == [dates[2], dates[6], dates[10], dates[18], dates[22]]
    assert len(run(reentry_max_losses=3, reentry_win=-1.0)) == 6  # every trade a "win": nothing skipped


def test_profit_floor_sells_the_rest_after_a_round_trip():
    #        signal  entry  +25% (gap)  +20%  +9% -> sold at the next open
    closes = [100, 100, 100, 125, 120, 109, 108, 107]
    dates, prices, sigs, top = make_world(closes, entry_day=1, ema=[50] * 8)
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, profit_floor=0.10))
    fills = trades.iloc[0]["exits"].split("; ")
    assert fills[0].startswith(f"{dates[3].date()} take-profit 50@125.00")
    assert fills[1].startswith(f"{dates[6].date()} profit-floor 100@108.00")  # close 109 < +10% on day 5
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0))
    assert "profit-floor" not in trades.iloc[0]["exits"]  # off by default
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, profit_floor=0.10, profit_floor_after=0.30))
    assert "profit-floor" not in trades.iloc[0]["exits"]  # never up 30%


def test_list_memory_allows_stocks_that_left_the_list_recently():
    dates, prices, sigs, top = make_world([100] * 10, entry_day=6)
    top = top[top["date"] <= dates[2]]  # on the list on days 0-2 only; the setup shows on day 6
    run = lambda **kw: strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, **kw))[1]
    assert len(run()) == 0  # not on today's list
    assert len(run(list_memory=3)) == 0  # last on the list 4 trading days earlier
    trades = run(list_memory=5)
    assert list(trades["entry_date"]) == [dates[7]] and trades.iloc[0]["entry_kind"] == "setup-recent"


def test_list_memory_forgets_a_stock_whose_sctr_falls_below_the_floor():
    dates, prices, sigs, top = make_world([100] * 10, entry_day=6)
    top.loc[3:, "rank"] = 50  # off the list from day 3 on, with an SCTR of 60 ...
    top.loc[3:, "sctr"] = 60.0
    top.loc[4, "sctr"] = 40.0  # ... that dips to 40 on day 4
    run = lambda **kw: strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, list_memory=5, **kw))[1]
    assert len(run()) == 1 and len(run(list_memory_min_sctr=30)) == 1
    assert len(run(list_memory_min_sctr=50)) == 0  # fell below 50 on day 4: forgotten
    with pytest.raises(ValueError):  # needs the SCTR of the stocks outside the top 10
        strategy.run(dates, prices, sigs, top[top["rank"] == 1], basic_rules(list_memory=5, list_memory_min_sctr=50))


def test_list_memory_needs_a_long_enough_run_on_the_list():
    dates, prices, sigs, top = make_world([100] * 12, entry_day=8)
    top = top[top["date"].isin(dates[[0, 1, 2, 4, 5]])]  # on the list on days 0-2 and 4-5, then off
    run = lambda **kw: strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, list_memory=5, **kw))[1]
    assert len(run(list_memory_min_run=6)) == 1  # days 0-5 count as one 6-day run (the 1-day break is short)
    assert len(run(list_memory_min_run=7)) == 0
    assert len(run(list_memory_min_run=6, list_memory_gap=0)) == 0  # no breaks allowed: the run is days 4-5


def test_list_memory_can_require_a_close_above_the_100_day_average():
    dates, prices, sigs, top = make_world([100] * 10, entry_day=6)
    top = top[top["date"] <= dates[2]]  # on the list on days 0-2; the setup shows on day 6
    sigs["AAA"]["sma100"] = 90.0
    run = lambda **kw: strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, list_memory=5, **kw))[1]
    assert len(run(list_memory_above_ma=100)) == 1  # close 100 above the average of 90
    sigs["AAA"]["sma100"] = 110.0
    assert len(run(list_memory_above_ma=100)) == 0 and len(run()) == 1  # below its average: skipped


def test_late_entry_after_a_valid_entry_made_off_the_list():
    dates, prices, sigs, top = make_world([100] * 10, entry_day=-1)
    sig = sigs["AAA"]
    sig["trend"] = [-1, -1, -1] + [1] * 7  # down-trend until day 2 ...
    sig.loc[dates[2], "setup"] = True  # ... setup at that close ...
    sig.loc[dates[3], "flip_level"] = 99.0  # ... and day 3's high (101) reaches the flip level: a valid entry
    top = top[top["date"] >= dates[5]]  # but the stock only joins the list on day 5
    run = lambda **kw: strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, **kw))[1]
    assert len(run()) == 0
    assert len(run(late_entry_days=1)) == 0  # day 5 is 2 trading days after the valid entry
    trades = run(late_entry_days=2)
    assert list(trades["entry_date"]) == [dates[6]] and trades.iloc[0]["entry_kind"] == "setup-late"  # next open


def twin_world(cross_days=5):
    """One stock: flat at 200, a slide to 100, a base, a jump to 120 and a climb. Its setup close
    comes `cross_days` days into the golden cross that follows, 10 days after an open 16% above the
    previous close; its SCTR was 3 about 3-4 months earlier and is 75 now."""
    closes = np.r_[[200.0] * 150, np.linspace(200, 100, 150), [100.0] * 40, np.linspace(120, 175, 80)]
    dates = pd.bdate_range("2020-01-01", periods=len(closes))
    close = pd.DataFrame({"AAA": closes}, index=dates)
    evening = int(np.flatnonzero(twins.golden_cross_days(close)["AAA"].to_numpy() == cross_days)[0])
    opens = close.shift(1).fillna(200.0)  # no gaps ...
    opens.iloc[evening - 10] = close.iloc[evening - 11] * 1.16  # ... but one
    sctr_table = pd.DataFrame({"AAA": 50.0}, index=dates)
    sctr_table.iloc[evening - 80 : evening - 59] = 3.0
    sctr_table.iloc[evening - 59 :] = 75.0
    setup = pd.DataFrame({"AAA": False}, index=dates)
    setup.iloc[evening] = True
    flip_level = close * 1.005  # reached by the next day's high (1% above the close)
    world = dict(closes=close, opens=opens, highs=close * 1.01, raw_closes=close, sctr_table=sctr_table, setup=setup,
                 trend=pd.DataFrame({"AAA": -1}, index=dates), flip_level=flip_level)
    assert 0.10 <= 1 - closes[evening] / 200 <= 0.30  # 10-30% below the 52-week high
    return world, dates[evening]


def test_twin_setup_needs_every_trait():
    world, day = twin_world()
    find = lambda **changes: twins.find_twins(**{**world, **changes})
    found = find()
    assert list(found["date"]) == [day] and list(found["ticker"]) == ["AAA"] and found["sctr"].iloc[0] == 75.0
    assert find(trend=world["trend"].replace(-1, 1)).empty  # the ATR trail is already up: no setup
    assert find(sctr_table=world["sctr_table"].replace(75.0, 95.0)).empty  # SCTR above 90: (nearly) on the list
    assert find(sctr_table=world["sctr_table"].replace(3.0, 10.0)).empty  # its low was 10, not near the bottom
    assert find(opens=world["closes"].shift(1).fillna(200.0)).empty  # no gap up
    assert find(flip_level=world["closes"] * 1.05).empty  # the next day's high doesn't reach the buy-stop
    world, day = twin_world(cross_days=20)
    assert list(twins.find_twins(**world)["date"]) == [day]
    world, _ = twin_world(cross_days=21)
    assert twins.find_twins(**world).empty  # the golden cross is too old


def test_twin_setups_join_the_list_for_n_days_ahead_of_the_top_10():
    dates, prices, sigs, top = make_world([100] * 10, entry_day=4)  # setup on day 4, bought at the day-5 open
    found = pd.DataFrame({"date": [dates[2]], "ticker": ["AAA"], "sctr": [75.0], "vol": [50.0]})
    off_list = top.assign(vol=50.0).iloc[0:0]  # never in the top 10
    run = lambda **kw: strategy.run(dates, prices, sigs, off_list, basic_rules(cost=0.0, **kw), twins=found)[1]
    assert len(run()) == 0  # off by default
    assert len(run(twin_days=2)) == 0  # on the list on days 2-3 only
    trades = run(twin_days=3)
    assert list(trades["entry_date"]) == [dates[5]] and trades.iloc[0]["entry_kind"] == "setup-twin"
    assert len(run(twin_days=3, max_vol=40)) == 0  # the volatility limit applies to twins too
    # One free slot, a setup on the same evening in the top 10 (TOP) and in the twin (AAA): the twin goes first.
    prices, sigs = {"TOP": prices["AAA"], "AAA": prices["AAA"]}, {"TOP": sigs["AAA"], "AAA": sigs["AAA"]}
    _, trades = strategy.run(dates, prices, sigs, top.assign(ticker="TOP"), basic_rules(cost=0.0, twin_days=3, max_positions=1),
                             twins=found)
    assert list(trades["ticker"]) == ["AAA"]
    with pytest.raises(ValueError):  # needs the table of twin setups
        strategy.run(dates, prices, sigs, top, basic_rules(twin_days=3))


def test_extra_slots_for_stocks_outside_the_top_n():
    # Three stocks signal on the same day, ranked #1, #15 and #20 (all SCTR above 90).
    dates, prices, sigs, top = make_world([100] * 5, entry_day=1)
    ranks = {"T0": 1, "T1": 15, "T2": 20}
    prices = {t: prices["AAA"] for t in ranks}
    sigs = {t: sigs["AAA"] for t in ranks}
    top = pd.concat([top.assign(ticker=t, rank=r, sctr=99.0 - r / 10) for t, r in ranks.items()])
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0))
    assert list(trades["ticker"]) == ["T0"]  # the top 10 only
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, extra_slots=1))
    assert sorted(trades["ticker"]) == ["T0", "T1"]  # one extra slot: the better ranked of the two
    assert list(trades.sort_values("ticker")["extra"]) == [False, True]
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, extra_slots=2))
    assert sorted(trades["ticker"]) == ["T0", "T1", "T2"]


def test_volatility_limits_skip_the_wildest_and_calmest_stocks():
    dates, prices, sigs, top = make_world([100] * 5, entry_day=1)
    vols = {"CALM": 20.0, "MID": 50.0, "WILD": 110.0}
    prices = {t: prices["AAA"] for t in vols}
    sigs = {t: sigs["AAA"] for t in vols}
    top = pd.concat([top.assign(ticker=t, rank=i + 1, vol=v) for i, (t, v) in enumerate(vols.items())])
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0))
    assert sorted(trades["ticker"]) == ["CALM", "MID", "WILD"]
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, max_vol=75))
    assert sorted(trades["ticker"]) == ["CALM", "MID"]
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(cost=0.0, min_vol=30, max_vol=75))
    assert list(trades["ticker"]) == ["MID"]


def test_rest_exit_waits_until_a_month_after_entry():
    # MSTR, Sep 2024: +8% on the buy day, then a dip below the 50-day line two days later.
    closes = [100, 100, 100, 109, 112, 107.8, 106.15, 106, 105, 104]
    dates, prices, sigs, top = make_world(closes, entry_day=1)  # bought day 2
    sigs["AAA"]["sma50"] = 110.0  # sell line 110 x 0.97 = 106.70
    rules = basic_rules(cost=0.0, rest_exit="sma50", sma_exit_buffer=0.03, rest_exit_delay=5)
    _, trades = strategy.run(dates, prices, sigs, top, rules)
    # Below the line on day 6 (only 4 days after entry): no sale; still below on day 7 -> next open.
    assert trades.iloc[0]["exits"].split("; ")[1] == f"{dates[8].date()} exit 100@105.00"


def test_long_line_is_the_green_line_on_the_flip_bar():
    close = series([100.0] * 30 + [105.0] * 11 + [99.0] * 5)
    trail = signals.atr_trail(close + 0.5, close - 0.5, close)
    assert trail["long_line"].iloc[30] == trail["trail"].iloc[30]  # flips up on bar 30
    assert trail["long_line"].iloc[20:30].notna().all()  # known in the down-trend too


def frozen_world():
    """Bought at the flip level (100.5) on day 2; the day closes back below it: no green line."""
    closes = [100, 100, 100, 99, 95, 90, 90]
    dates, prices, sigs, top = make_world(closes, entry_day=-1)
    sig = sigs["AAA"]
    sig["setup"] = [False, True, False, False, False, False, False]
    sig["trend"] = -1
    sig["flip_level"] = [np.nan, np.nan, 100.5, 100.5, 100.5, 100.5, 100.5]
    sig["long_line"] = [90.0, 90.0, 96.0, 95.0, 92.0, 88.0, 88.0]  # day 2 would have had 96
    return dates, prices, sigs, top


def test_frozen_line_stops_a_no_line_entry():
    dates, prices, sigs, top = frozen_world()
    rules = basic_rules(cost=0.0, entry_on_touch=True, atr_exit_on_touch=True, atr_stop_buffer=0.02)
    _, trades = strategy.run(dates, prices, sigs, top, rules)
    assert "open at end" in trades.iloc[0]["exits"]  # no green line, no stop: held all the way down
    _, trades = strategy.run(dates, prices, sigs, top, basic_rules(
        cost=0.0, entry_on_touch=True, atr_exit_on_touch=True, atr_stop_buffer=0.02, frozen_line=True))
    # Frozen at 96 (not 95 or 92 later): stop 96 x 0.98 = 94.08; day 4 low 94.05 touches it.
    assert trades.iloc[0]["exits"] == f"{dates[4].date()} no-line-stop 149@94.08"


def test_frozen_line_hands_over_to_the_real_green_line_on_a_breakout():
    dates, prices, sigs, top = frozen_world()
    sigs["AAA"]["trend"] = [-1, -1, -1, 1, 1, 1, 1]  # closes above the flip line on day 3
    sigs["AAA"]["atr_stop"] = [np.nan] * 4 + [85.0, 85.0, 85.0]  # the real green line, from day 4
    rules = basic_rules(cost=0.0, entry_on_touch=True, atr_exit_on_touch=True, atr_stop_buffer=0.02, frozen_line=True)
    _, trades = strategy.run(dates, prices, sigs, top, rules)
    assert "no-line-stop" not in trades.iloc[0]["exits"]  # day 4 dip to 94.05 is far above 85 x 0.98
