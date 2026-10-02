# SCTR Top-10 Strategy Backtest

A step-by-step backtest of a US large-cap strategy that trades the top 10 stocks
by **StockCharts Technical Rank (SCTR)**.

## Step 1: Stock selection (done)

Each trading day, every stock in the large-cap universe gets an SCTR score using
the [StockCharts methodology](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/stockcharts-technical-rank-sctr):

| Timeframe | Indicator | Weight |
|---|---|---|
| Long | % above/below 200-day EMA | 30% |
| Long | 125-day rate of change | 30% |
| Medium | % above/below 50-day EMA | 15% |
| Medium | 20-day rate of change | 15% |
| Short | 3-day slope of PPO(12,26,9) histogram (0-5 points) | 5% |
| Short | 14-day RSI | 5% |

Scores are ranked within the universe each day and scaled from 0.0 (weakest)
to 99.9 (strongest). The 10 highest are the picks. A stock needs 250 days of
price history before it can be ranked.

**Universe.** StockCharts' Large-Cap universe is "US stocks with a market cap
over $10B". There is no free source of historical market cap, so we use the
**S&P 500 as it was on each date** (point-in-time membership from
[fja05680/sp500](https://github.com/fja05680/sp500)). This avoids survivorship
bias (testing only on today's winners). Our SCTR values will be close to
StockCharts' but not identical, because their universe is somewhat larger.

### Usage

```bash
pip install -r requirements.txt

# 1. Download membership history + daily prices (Yahoo Finance)
python -m backtest.download_data --start 2015-01-01

# 2a. Top 10 on a given date, with the six SCTR components
python -m backtest.pick_stocks --date 2025-09-30

# 2b. Daily top-10 history for the backtest
python -m backtest.pick_stocks --start 2015-01-01 --out top10_daily.csv

# Tests
python -m pytest
```

**Using your own data.** Put one CSV per ticker in `data/prices/<TICKER>.csv`
with a `Date` (or TradingView `time`) column and a `Close` / `Adj Close`
column, then run `pick_stocks` with `--universe all` to rank every file in that
folder.

### Known limitations

- Yahoo Finance does not carry most delisted companies, so some past S&P 500
  members will have no prices. `download_data` lists them. This brings back
  some survivorship bias. For a cleaner test, use a paid source with delisted
  stocks (e.g. Norgate Data or EODData).
- The S&P 500 is a proxy for "market cap > $10B"; see above.

## Next steps

- Step 2: entry rules
- Step 3: exit rules
- Step 4: position sizing, costs, and performance report
