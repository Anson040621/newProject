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

The defaults are the current rule set ("version #24" = #23 with the climax exit; #23 = #22 without the break-even exit and with re-entry after any exit; #22 = #21 with a 75% volatility limit on the stocks; #21 = #19 with a 3% cushion on the break-even exit and the 50-day exit switched on one month after entry):

| Rule | Default |
|---|---|
| Stocks | daily SCTR top 10 among US stocks over $10B (`--top`, `--min-sctr 90`), skipping stocks whose volatility over the last year (252 trading days, annualised) is 75% or more (`--max-vol`, 0 = off = #21; `--min-vol` for a minimum) |
| Setup (at a close) | in the top 10, squeeze on (blue crosses), momentum bar red, ATR trail in a down-trend |
| Entry | next day, buy-stop at the ATR flip level; filled when the price touches it, at the open if it gaps above (`--entry-buffer` to place it higher) |
| Size | 15% of equity per trade (`--size`), at most 6 positions (`--max-positions`); highest SCTR first |
| Exit before +8% | a) a close below the green ATR line (SELL label): everything at the next open; b) safety net: the price touches 2% below the previous close's green line (`--atr-stop-buffer`): everything at once, at the open if it gaps below |
| Take profit | 1/3 at +8% (`--take-profit`), intraday |
| Exit after +8% | the rest at the next open after a close more than 3% below the 50-day SMA (`--sma-exit-buffer`), once price has closed above that level and from 21 trading days after the entry or re-entry (`--rest-exit-delay`, 0 = at once) (`--rest-exit ema20` for the old rule: 3 closes below the 20 EMA); nothing else sells the rest, so between the +8% sale and day 21 it is held (`--breakeven-exit` = #22: also sell after a close more than 3% below the entry price, `--breakeven-buffer`) |
| Climax exit | once a trade has been up 40% at a close (`--climax-gain`, 0 = off = #23) and within the last 10 trading days closed 50%+ above its 100-day SMA (`--climax-stretch`, `--climax-ma`, `--climax-days`), an open at least 5% below the previous day's low (`--climax-gap`) sells everything at the next open, with no re-entry (`--climax-reentry` to allow it) |
| Re-entry | after any exit, before or after +8%, if the ATR flips back to BUY within 5 trading days of the exit, buy-stop at the flip level, at any SCTR rank (`--reentry-days`, 0 = off; `--no-reentry-after-tp` = #22: only after exits before +8%; `--reentry-top N` to require the top N) |
| Costs | 0.1% per buy and per sell (`--cost`) |
| Tested alternatives (on #21 or earlier unless noted) | on #24, a big red candle as a second climax trigger (gap down OR ...): `--climax-drop 0.07` (a close 7%+ below the previous close: 20.4% a year, worst drop -25.2%, Sharpe 1.20, Sep 2020 drop -11.2% instead of -15.4%, 2026 drop -12.9%; it sells TSLA on Sep 4 2020 at +182% but HOOD in Dec 2024 at +62% instead of +106%), `--climax-drop 0.10` (20.6%, -25.3%), `--climax-body 0.06` / `0.08` (a close 6% / 8%+ below the day's open: 20.0% / 20.7% a year, worst drop -23.9% / -25.3%, Sharpe 1.21 / 1.18; 6% also sells SE in Jun 2020 at +89% instead of +266%; 5% / 7%: 19.4% / 19.8% a year, worst drop -23.8% / -24.0%, so the body trigger is stable around 5-7%; `--climax-drop 0.06` / `0.08`: 18.9% / 20.3%, worst drop -23.5% / -25.2%), `--climax-drop-atr 2` / `2.5` / `3` (a fall of 2 / 2.5 / 3 x the 14-day ATR: 20.7% / 21.0% / 21.8% a year, worst drop -25.3% / -25.3% / -28.2%, no help in Sep 2020); on #23: `--climax-stretch 0.40 --climax-ma 50 --climax-gap 0` (climax exit: once a trade has been up 40% and closed 40%+ above its 50-day SMA within 10 days, a gap down below the previous low sells everything at the next open, no re-entry: 18.0% a year, worst drop -24.0%, Sharpe 1.09; the 2026 give-back drop shrinks from -27.1% to -14.9% and 2025's from -25.0% to -17.1%, but it sells SE and TSLA in Jul 2020 and APP in Oct 2024 on their first climax; stretch 30%: 16.8% / -24.4%, 25%: 14.7% / -23.6%, 20%: 13.7% / -25.9%; with `--climax-reentry` 17.6% / -24.3%; `--climax-stretch 0.60` (gap of at least 5% below the previous low, after a close 60%+ above the 100-day SMA): 22.1% a year, worst drop -28.4% (the 2021-22 grind, untouched), Sharpe 1.16, only 5 climax exits, all in 2024-2026 (DELL, HOOD, APP, MU, WDC): the 2026 drop shrinks from -27.1% to -16.0% and 2025's from -25.0% to -20.5%; nearby settings: gap 3% 20.3%, gap 7% 22.6%, stretch 50% 21.7% (2025 drop -14.8%), stretch 70% 21.7%), `--market-ma 200` / `100` / `50` (no new buys unless SPY and QQQ are above their average: 11.5% / 5.7% / 1.8% a year, worst drop -22.1% / -18.4% / -18.6%; with 200 days it skips 126 trades worth $335k, e.g. SE and TSLA in spring 2020, PLTR and AS Apr 2025, MU Apr 2026: the setup buys pullbacks, which come with market dips); on #22 (`--breakeven-exit --no-reentry-after-tp`): re-entry after +8% exits too (19.2% a year, worst drop -27.6%, profit factor 2.87, 342 trades; the re-entries include APP Aug 2024 +223%, SOFI Apr 2025 +101%, MU Apr 2026 +92%, PLTR Apr 2025 +69% and WDC Apr 2026 +51%), both changes = #23 (20.9% a year, worst drop -28.4%, profit factor 2.91, 350 trades; more trades give back the +8% and end in a loss, 40 vs 36, e.g. TSLA Mar 2020 -17%, RIVN Sep 2023 -15%; adding `--red-line-stop 0.02` / `0.03` / `0.05` (no green line yet: sell 2% / 3% / 5% below the previous close's red line): 16.3% / 14.3% / 12.3% a year, worst drop -27.7% / -28.3% / -32.1%; with 3%, 71% of the 117 stopped trades were bought back within 5 days (APP, SE, WDC, TLN), but PLTR Mar 2025 (+63%), AS Apr 2025 (+38%) and COHR Apr 2026 (+22%) were not; adding `--emergency-stop 0.12`: 16.0% a year, worst drop -33.9%, 64 stop-outs, 34 of the 50 stopped trades that also exist without the stop would have ended better without it, and -8% / -10% / -15% / -20% stops give 13.5% / 14.8% / 17.0% / 16.8% a year), no break-even exit alone (14.4% a year, worst drop -23.9%), `--reentry-days 7` (12.8% a year, worst drop -29.5%; with re-entry after +8% 17.2% / -39.1%, and without the break-even exit too 20.0% / -40.9%; on #23 the re-entry window 0 / 2 / 3 / 4 / 6 / 7 days gives 13.8% / 14.6% / 16.4% / 18.6% / 19.9% / 20.0% a year with worst drops of -24% to -41%); other volatility limits (#22's 75% limit vs none (#21): 2015-2020 about even (14.8% vs 15.0% a year), 2021-2026 better (11.4% vs 7.3%); it skips LITE Jan 2026 (+84%) but also the XXI, MSTR, NBIS, SMCI and COIN losses; any limit from 60% to 90% gives 12.2-13.3% a year with a worst drop of -21% to -23.5% (50%: 5.8%, 100%: 11.5%), about the same as no limit in 2015-2020 (14.8-15.3% vs 15.0% a year) and better in 2021-2026 (10.0-11.7% vs 7.3%); a minimum (`--min-vol`) of 20-30% changes little (`--min-vol 30 --max-vol 80`: 12.2%, -22.5%), 35-40% costs 2-3% a year; for comparison, a 1-year beta of 1-2.5 on the top 10: 10.4% a year, worst drop -21.5%), `--extra-slots 1` / `2` (the top 10 first, plus at most 1 / 2 positions in other stocks with an SCTR above 90: 9.3% / 11.3% a year, worst drop -41.9% / -42.4%, profit factor 1.53 / 1.60; the extra trades averaged -0.6% / +0.8% and with 2 slots lost $46k in 2021-2022; neither caught ILMN in Aug 2026, the extra slots were busy with ROIV and RVMD), `--top 1000` (every stock with an SCTR above 90, about 90 a day: 8.8% a year, worst drop -42.1%, profit factor 1.37; it catches ILMN in Aug 2026 (+27%), but the 6 slots are nearly always full, so 214 of the 302 top-10 trades are skipped, among them SE Mar 2020, CIEN Nov 2025 and LITE Jan 2026), `--frozen-line` (no-line entries: the green line the buy day would have had, frozen as the stop until the trend really turns up: 9.1% a year, worst drop -32.0%; the frozen stop sits a median 7.9% below the entry and sold 56 trades at -8.6% on average), `--reentry-top 10` / `20` (re-entries only while the stock is in the daily top 10 / 20: 11.1% / 10.8% a year, worst drop -25.1% / -25.2%, vs 11.4% and -26.3% without the requirement), `--entry-check live` (a buy order waits at the flip level for every top-10 stock in an ATR down-trend; it buys at the first price on the way up through the line where the squeeze is on and the bar red, as the indicators read live: 12.2% a year, worst drop -28.1%, profit factor 1.78 with the break-even exit at the entry; 11.7% / -30.5% with the 3% cushion), `--entry-check either` (11.7% / -29.0% at the entry, 11.1% / -31.3% with the cushion), `--close-entry` (also buy at the next open when the squeeze, red bar and ATR flip to BUY all appear on the same bar, e.g. HOOD 9 Apr 2025: 11.9% a year but worst drop -36.8% and profit factor 1.64; the 89 extra trades lost $41k, 43% winners), `--atr-stop-buffer` 0 / 0.03 / 0.05 (safety net on the line / 3% / 5% below it: 10.0% / 11.3% / 11.3% a year, the 5% one is version #17), `--dim-green-days 42` (within about 2 months of a full-setup entry we have exited, also buy the same setup with a dim green momentum bar: 12.3% a year, worst drop -26.3%, but only 7 extra trades and most of the gain is one trade, WDC Apr 2026), `--failed-breakout-exit` (sell at the next open when the entry day closes back below the flip level: 5.4% a year), `--emergency-stop 0.12` on #17 (6.6% a year), `--entry-buffer 0.02`, `--no-close-exit --sticky-stop --atr-stop-buffer 0.03` (#14), `--emergency-stop 0.12` (#15), `--no-atr-exit-on-touch --atr-stop-buffer 0` (version G), `--park QQQ`, `--rebuy-shakeouts` |

2015-01-02 to 2026-10-02 with the defaults (#24): $100k -> $1.00M (21.7% a year,
worst drawdown -28.2%, Sharpe 1.18, 349 trades, 44% winners, profit factor 3.15) vs SPY
13.8% a year (worst drawdown -33.7%). The climax exit fired 7 times (ZM 2020, DELL 2024,
APP, HOOD, CIEN, MU, WDC), never in 2015-2019 or 2021-2023; it cuts the 2026 drop from
-27.1% to -16.2% and 2025's from -25.0% to -14.8%. #23 (`--climax-gain 0`): $934k (20.9%
a year, worst drawdown -28.4%, Sharpe 1.07, 350 trades, 43% winners, profit factor 2.91).
#23 with idle money in QQQ (`--climax-gain 0 --park QQQ`): $1.48M
(25.9% a year, worst drawdown -41.0%, Nov 2021 to Jun 2022, Sharpe 0.96) vs holding QQQ
alone $794k (19.3% a year, worst drawdown -35.1%); 2022 was -26% instead of -6%, while the
calm bull years improved (2019 +37% vs +15%, 2021 +8% vs -6%, 2023 +28% vs +5%).
#22 (`--climax-gain 0 --breakeven-exit --no-reentry-after-tp`):
$427k (13.2% a year, worst drawdown -22.8%, 254 trades, 44% winners, profit factor 2.75).
#21 (#22 without the volatility limit, adding `--max-vol 0`): $347k (11.2% a year, worst drawdown
-27.5%, 302 trades, 42% winners, profit factor 1.82). #20 (#21 with the 50-day exit from the start, also `--rest-exit-delay 0`): $356k (11.4%, -26%). #19 (#20 with the break-even exit at the entry, also `--breakeven-buffer 0`): $375k (11.9%,
-24%, profit factor 1.88). #17 (5% safety net): $352k (11.3%, -26%). #16 (20 EMA exit): $258k (8.4%, -22%); #13 (no break-even exit): $289k (9.5%, -26%).
#14 (+2% entry, stop-only exit) returned 1.0% a year and #15 (#14 + 12% hard
stop) -0.9%: the +2% entry skipped 7 of the 12 biggest winners.

Version G results:
2015-01-02 to 2026-10-02: $100k -> $254k (8.3% a year, worst drawdown -36%,
344 trades, 45% winners) vs SPY 13.8% a year. Most of the gain came in 2020
(+80%); excluding 2020 the strategy returned about 3% a year.

With idle money parked in QQQ (`--park QQQ`): $100k -> $728k (18.5% a year,
worst drawdown -42%) vs holding QQQ alone $100k -> $794k (19.3%, -35%). The
trades earned about the same as QQQ over the days they were held (2.2% vs 2.3%
on average).

```bash
python -m backtest.run_backtest --start 2015-01-01
```

Prints the statistics and yearly returns against SPY (with dividends) and
writes `backtest_trades.csv` and `backtest_equity.csv`. Trade prices are
split-adjusted; dividends on the stocks held are ignored.
