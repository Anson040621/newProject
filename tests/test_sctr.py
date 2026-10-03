import numpy as np
import pandas as pd
import pytest

from backtest import sctr
from backtest.prices import load_closes
from backtest.universe import load_membership, members_on, membership_mask, tickers_between


def random_walk(n=400, drift=0.0, seed=0, start=100.0):
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(drift + 0.02 * rng.standard_normal(n)))


def frame(**cols):
    n = len(next(iter(cols.values())))
    return pd.DataFrame(cols, index=pd.bdate_range("2020-01-01", periods=n))


# --- reference implementations written as plain loops -------------------------

def ema_loop(values, span):
    alpha = 2.0 / (span + 1)
    out, prev = [], None
    for v in values:
        prev = v if prev is None else alpha * v + (1 - alpha) * prev
        out.append(prev)
    return np.array(out)


def rsi_loop(values, period=14):
    gains = losses = 0.0
    out = [np.nan] * len(values)
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gain, loss = max(change, 0.0), max(-change, 0.0)
        if i == 1:
            gains, losses = gain, loss
        else:
            gains = gains + (gain - gains) / period
            losses = losses + (loss - losses) / period
        if i >= period:
            out[i] = 100.0 if losses == 0 else 100.0 - 100.0 / (1.0 + gains / losses)
    return np.array(out)


def score_loop(values):
    """SCTR indicator score on the last bar, straight from the StockCharts article."""
    close = values[-1]
    pct200 = (close / ema_loop(values, 200)[-1] - 1) * 100
    pct50 = (close / ema_loop(values, 50)[-1] - 1) * 100
    roc125 = (close / values[-1 - 125] - 1) * 100
    roc20 = (close / values[-1 - 20] - 1) * 100
    ema12, ema26 = ema_loop(values, 12), ema_loop(values, 26)
    ppo = (ema12 - ema26) / ema26 * 100
    # The first 25 PPO values are NaN in the pandas version (min_periods), so
    # start the signal line where the slow EMA becomes valid.
    signal = np.full_like(ppo, np.nan)
    signal[25:] = ema_loop(ppo[25:], 9)
    hist = ppo - signal
    slope = (hist[-1] - hist[-4]) / 3
    if slope > 1:
        ppo_pts = 5.0
    elif slope < -1:
        ppo_pts = 0.0
    else:
        ppo_pts = 0.05 * ((slope + 1) * 50)
    rsi14 = rsi_loop(values)[-1]
    return 0.30 * pct200 + 0.30 * roc125 + 0.15 * pct50 + 0.15 * roc20 + ppo_pts + 0.05 * rsi14


# --- indicators ---------------------------------------------------------------

def test_roc_and_pct_from_ema():
    close = frame(A=np.arange(1.0, 301.0))
    assert sctr.roc(close, 20)["A"].iloc[-1] == pytest.approx((300 / 280 - 1) * 100)
    assert sctr.roc(close, 125)["A"].iloc[124] != sctr.roc(close, 125)["A"].iloc[124]  # NaN before enough data
    flat = frame(A=np.full(300, 50.0))
    assert sctr.pct_from_ema(flat, 200)["A"].iloc[-1] == pytest.approx(0.0)


def test_ema_matches_loop_after_warmup():
    values = random_walk(300, seed=1)
    ours = sctr.ema(frame(A=values), 50)["A"]
    assert ours.iloc[:49].isna().all()
    np.testing.assert_allclose(ours.iloc[49:], ema_loop(values, 50)[49:])


def test_rsi_matches_wilder_loop():
    values = random_walk(300, seed=2)
    ours = sctr.rsi(frame(A=values))["A"].to_numpy()
    ref = rsi_loop(values)
    np.testing.assert_allclose(ours[14:], ref[14:], rtol=1e-9)


def test_rsi_extremes():
    assert sctr.rsi(frame(A=np.arange(1.0, 51.0)))["A"].iloc[-1] == pytest.approx(100.0)
    assert sctr.rsi(frame(A=np.arange(50.0, 0.0, -1.0)))["A"].iloc[-1] == pytest.approx(0.0)
    assert sctr.rsi(frame(A=np.full(50, 10.0)))["A"].iloc[-1] == pytest.approx(50.0)


def test_ppo_slope_points_bounds():
    flat = frame(A=np.full(100, 10.0))
    assert sctr.ppo_slope_points(flat)["A"].iloc[-1] == pytest.approx(2.5)  # slope 0 -> half of 5 points
    pts = sctr.ppo_slope_points(frame(A=random_walk(400, seed=3)))["A"].dropna()
    assert pts.between(0.0, 5.0).all()
    # A violent jump makes the histogram rise steeply -> capped at 5 points.
    jump = np.r_[np.full(100, 10.0), np.full(3, 30.0)]
    assert sctr.ppo_slope_points(frame(A=jump))["A"].iloc[-1] == pytest.approx(5.0)


def test_indicator_score_matches_article_formula():
    values = random_walk(400, drift=0.001, seed=4)
    ours = sctr.indicator_score(frame(A=values))["A"].iloc[-1]
    assert ours == pytest.approx(score_loop(values), rel=1e-9)


def test_score_requires_min_history():
    score = sctr.indicator_score(frame(A=random_walk(300, seed=5)))["A"]
    assert score.iloc[: sctr.MIN_HISTORY_DAYS - 1].isna().all()
    assert score.iloc[sctr.MIN_HISTORY_DAYS - 1 :].notna().all()


# --- ranking ------------------------------------------------------------------

def test_rank_scores_spread_0_to_99_9():
    score = frame(A=[1.0], B=[5.0], C=[3.0])
    ranked = sctr.rank_scores(score).iloc[0]
    # C sits exactly in the middle: 49.95, truncated to 49.9 like StockCharts.
    assert ranked.to_dict() == {"A": 0.0, "B": 99.9, "C": 49.9}


def test_rank_scores_truncates_like_stockcharts():
    # With ~912 stocks the 2nd best is 99.79... which StockCharts shows as 99.7.
    score = pd.DataFrame([np.arange(912.0)], index=pd.bdate_range("2024-01-01", periods=1))
    ranked = sctr.rank_scores(score).iloc[0].sort_values(ascending=False)
    assert list(ranked.iloc[:3]) == [99.9, 99.7, 99.6]


def test_rank_scores_respects_universe_mask():
    score = frame(A=[1.0, 1.0], B=[5.0, 5.0], C=[3.0, 3.0])
    mask = frame(A=[True, True], B=[True, False], C=[True, True])
    ranked = sctr.rank_scores(score, mask)
    assert np.isnan(ranked["B"].iloc[1])
    assert ranked.iloc[1][["A", "C"]].to_dict() == {"A": 0.0, "C": 99.9}


def test_daily_top_n_picks_strongest_trends():
    # 15 stocks with increasing drift: the strongest uptrends should rank on top.
    cols = {f"S{i:02d}": random_walk(400, drift=0.0003 * i, seed=100 + i) for i in range(15)}
    close = frame(**cols)
    top = sctr.daily_top_n(close, n=10)
    last = top[top["date"] == close.index[-1]]
    assert len(last) == 10
    assert list(last["rank"]) == list(range(1, 11))
    assert last["sctr"].is_monotonic_decreasing
    assert last["sctr"].iloc[0] == pytest.approx(99.9)
    assert {"S14", "S13", "S12"} <= set(last["ticker"].head(5))
    # Nothing before the warm-up period.
    assert top["date"].min() == close.index[sctr.MIN_HISTORY_DAYS - 1]


def test_sctr_table_columns():
    cols = {f"S{i}": random_walk(300, drift=0.0005 * i, seed=i) for i in range(5)}
    close = frame(**cols)
    table = sctr.sctr_table(close, close.index[-1])
    assert list(table.columns) == ["sctr", "score", "pct_ema200", "roc125", "pct_ema50", "roc20", "ppo_points", "rsi14"]
    assert table["score"].is_monotonic_decreasing


# --- universe and price loading ----------------------------------------------

def test_membership_point_in_time(tmp_path):
    path = tmp_path / "members.csv"
    path.write_text('date,tickers\n2020-01-02,"AAA,BBB"\n2020-06-01,"AAA,CCC"\n')
    membership = load_membership(path)
    assert members_on(membership, "2020-03-15") == {"AAA", "BBB"}
    assert members_on(membership, "2020-06-01") == {"AAA", "CCC"}
    assert tickers_between(membership, "2020-03-01", "2020-12-31") == ["AAA", "BBB", "CCC"]
    dates = pd.to_datetime(["2020-01-01", "2020-05-29", "2020-06-02"])
    mask = membership_mask(membership, dates, ["AAA", "BBB", "CCC"])
    assert mask.to_numpy().tolist() == [[False, False, False], [True, True, False], [True, False, True]]


def test_load_closes_yahoo_and_tradingview_formats(tmp_path):
    (tmp_path / "AAA.csv").write_text("Date,Open,High,Low,Close,Volume\n2024-01-02,1,1,1,10.5,100\n2024-01-03,1,1,1,11,100\n")
    # TradingView export: unix seconds in "time", plus an "Adj Close"-less layout.
    t0 = int(pd.Timestamp("2024-01-02 14:30", tz="UTC").timestamp())
    (tmp_path / "BBB.csv").write_text(f"time,open,high,low,Close\n{t0},1,1,1,20\n{t0 + 86400},1,1,1,21\n")
    (tmp_path / "CCC.csv").write_text("Date,Close,Adj Close\n2024-01-02,50,40\n2024-01-03,52,41\n")
    closes = load_closes(prices_dir=tmp_path)
    assert list(closes.columns) == ["AAA", "BBB", "CCC"]
    assert closes.loc["2024-01-03"].to_dict() == {"AAA": 11.0, "BBB": 21.0, "CCC": 41.0}
