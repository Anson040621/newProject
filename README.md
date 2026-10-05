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

**Universe.** Like StockCharts' Large-Cap universe: every US-listed stock
(including foreign ADRs such as UMC) with a market cap over **$10 billion**,
rebalanced at each month end. A stock must be on the other side of $10B for
**3 consecutive month ends** before it moves in or out.

Market cap on each past date = price x shares outstanding at the time:

- Today's listings and market caps come from the Nasdaq stock screener.
- Share-count history comes from SEC filings (10-K/10-Q cover pages), using
  only filings already published on that date, adjusted for later stock splits.
- The history is scaled so today's value matches Nasdaq (this handles ADRs and
  companies with several share classes).
- Companies with no SEC history use today's share count. Delisted companies with
  no data at all count as large caps while they were in the S&P 500.
- Preferred shares, notes, warrants and units are left out (the screener lists
  them with the parent company's market cap).

**Check against StockCharts.** On 2026-10-02 the top 11 matched StockCharts'
Large-Cap SCTR report exactly, ticker and value (MRNA 99.9, DELL 99.7, UMC
99.6, HPE 99.5, AMD 99.4, OKTA 99.3, CRWD 99.2, ALAB 99.1, MU 99.0, P 98.9,
SMTC 98.8). The one difference was CORT, which had just completed its third
month above $10B: we add it the day after the month end, StockCharts later.

Two other universes are available with `--universe`: `sp500` (the S&P 500 as it
was on each date) and `all` (every file in `data/prices/`).

### Usage

```bash
pip install -r requirements.txt

# 1. Download listings, prices (Yahoo) and share counts (SEC).
#    The SEC asks for a contact email in every request:
export SEC_USER_AGENT="sctr-backtest your@email.com"
python -m backtest.download_data --start 2015-01-01

# 2a. Top 10 on a given date, laid out like StockCharts' SCTR report
#     (name, sector, SCTR, CHG, close, market cap); --details adds the six indicators
python -m backtest.pick_stocks --date 2025-09-30

# 2b. Daily top-10 history for the backtest
python -m backtest.pick_stocks --start 2015-01-01 --out top10_daily.csv

# Tests
python -m pytest
```

The full download covers ~2,900 stocks (everything worth $1B+ today, since some
were worth $10B+ in the past, plus all S&P 500 members since the start date) and
takes about 30 minutes. Without `SEC_USER_AGENT` the SEC step is skipped and
today's share counts are used for every date.

**Using your own data.** Put one CSV per ticker in `data/prices/<TICKER>.csv`
with a `Date` (or TradingView `time`) column and a `Close` / `Adj Close`
column, then run `pick_stocks` with `--universe all` to rank every file in that
folder.

### Known limitations

- Yahoo Finance does not carry most delisted companies, so some past large caps
  have no prices (`download_data` lists them). This brings back some
  survivorship bias. For a cleaner test, use a paid source with delisted stocks
  (e.g. Norgate Data or EODData).
- Market caps before a company's first SEC filing, or for companies that report
  shares irregularly, are estimates. Stocks without SEC filings rely on the
  Nasdaq screener's market cap, which is occasionally wrong (e.g. a small fund
  listed at $13B). A few such stocks near $10B can be misclassified.
- StockCharts does not publish its rebalancing day; we apply changes on the
  first trading day after each month end, which can be a few days early.
- A ticker that changed (e.g. Paramount, now PSKY) only has SEC share history
  under its new company, so its earlier years may be missed.
- Yahoo sometimes leaves the latest daily bar empty for hours after the close;
  `download_data` fills that day from Yahoo's quote data (close, high, low,
  volume).

## Steps 2-4: Trading rules and backtest

`backtest/signals.py` ports the two TradingView scripts (squeeze + momentum
from "SEPA", and the "2x ATR Trailing Exit"); `backtest/strategy.py` simulates
the portfolio day by day.

| Rule | Default |
|---|---|
| Stocks | daily SCTR top 10 (`--top`), SCTR above 90 (`--min-sctr`) |
| Entry | same day: squeeze on (blue crosses), momentum bar red, ATR trail flips to BUY; buy at the next open |
| Size | 15% of equity per trade (`--size`), at most 6 positions (`--max-positions`); highest SCTR first |
| Exit before +8% | everything at the next open after the ATR trail flips to exit |
| Take profit | 1/3 at +8% (`--take-profit`), intraday |
| Exit after +8% | the rest at the next open after a close below the 20 EMA (once price has closed above it) |
| Costs | 0.1% per buy and per sell (`--cost`) |

```bash
python -m backtest.run_backtest --start 2015-01-01
```

Prints the statistics and yearly returns against SPY (with dividends) and
writes `backtest_trades.csv` and `backtest_equity.csv`. Trade prices are
split-adjusted; dividends on the stocks held are ignored.
