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

The defaults are the current rule set ("version #26" = #25 with no more re-buys of a stock after 3 trades in a row without a win; #25 = #24 with a big red candle as a second climax trigger; #24 = #23 with the climax exit; #23 = #22 without the break-even exit and with re-entry after any exit; #22 = #21 with a 75% volatility limit on the stocks; #21 = #19 with a 3% cushion on the break-even exit and the 50-day exit switched on one month after entry):

| Rule | Default |
|---|---|
| Stocks | daily SCTR top 10 among US stocks over $10B (`--top`, `--min-sctr 90`), skipping stocks whose volatility over the last year (252 trading days, annualised) is 75% or more (`--max-vol`, 0 = off = #21; `--min-vol` for a minimum) |
| Setup (at a close) | in the top 10, squeeze on (blue crosses), momentum bar red, ATR trail in a down-trend |
| Entry | next day, buy-stop at the ATR flip level; filled when the price touches it, at the open if it gaps above (`--entry-buffer` to place it higher) |
| Size | 15% of equity per trade (`--size`), at most 6 positions (`--max-positions`); highest SCTR first |
| Exit before +8% | a) a close below the green ATR line (SELL label): everything at the next open; b) safety net: the price touches 2% below the previous close's green line (`--atr-stop-buffer`): everything at once, at the open if it gaps below |
| Take profit | 1/3 at +8% (`--take-profit`), intraday |
| Exit after +8% | the rest at the next open after a close more than 3% below the 50-day SMA (`--sma-exit-buffer`), once price has closed above that level and from 21 trading days after the entry or re-entry (`--rest-exit-delay`, 0 = at once) (`--rest-exit ema20` for the old rule: 3 closes below the 20 EMA); nothing else sells the rest, so between the +8% sale and day 21 it is held (`--breakeven-exit` = #22: also sell after a close more than 3% below the entry price, `--breakeven-buffer`) |
| Climax exit | once a trade has been up 40% at a close (`--climax-gain`, 0 = off = #23) and within the last 10 trading days closed 50%+ above its 100-day SMA (`--climax-stretch`, `--climax-ma`, `--climax-days`), an open at least 5% below the previous day's low (`--climax-gap`) or a big red candle, a close at least 6% below the day's open (`--climax-body`, 0 = #24), sells everything at the next open, with no re-entry (`--climax-reentry` to allow it) |
| Re-entry | after any exit, before or after +8%, if the ATR flips back to BUY within 5 trading days of the exit, buy-stop at the flip level, at any SCTR rank (`--reentry-days`, 0 = off; `--no-reentry-after-tp` = #22: only after exits before +8%; `--reentry-top N` to require the top N); no more re-buys after 3 trades in a row (the setup buy and the re-buys after it) that made no more than +20% (`--reentry-max-losses`, 0 = off = #25; `--reentry-win`); a new setup buy starts the count again |
| Costs | 0.1% per buy and per sell (`--cost`) |
| Tested alternatives (on #21 or earlier unless noted) | on #26, lists built from the 20-day rise in SCTR (stocks with an SCTR of at least 75 and under 75% volatility, ranked by how much their SCTR rose in 20 trading days; a one-off test, not a command-line option): the 10 fastest risers instead of the top 10: $161k (4.1% a year, 102 trades: fast risers rarely pull back into a setup); the top 5 by SCTR plus the 5 fastest risers: $301k (9.8%); the top 10 plus the 3 / 5 / 10 fastest risers: $855k / $920k / $906k (20.0% / 20.8% / 20.6% a year, worst drop -23.0% / -20.0% / -22.3%, Sharpe 1.24 / 1.27 / 1.25); the top 10 plus 5 with the rise measured over 10 / 40 days: $834k / $856k (worst drop -25.9% / -22.2%); none of them buys META in Mar 2023; on #26, a top-10 list from a faster SCTR (same formula, every look-back window shortened; a one-off test, not a command-line option) so stocks coming back from a crash rank higher sooner: windows 15% / 25% / 40% / 50% / 65% shorter: $381k / $682k / $380k / $189k / $163k (12.1% / 17.8% / 12.0% / 5.6% / 4.2% a year, worst drop -27.9% / -25.0% / -23.4% / -27.4% / -19.7%, Sharpe 0.85 / 1.14 / 0.92 / 0.51 / 0.56); the faster lists lose many of #26's big trades (half as long keeps 8 of its 18 trades of +40% or more); META ranks 7th on its Mar 2 2023 setup day with half-length windows (161st normally) and is bought Mar 3 (-0.7%, then +34.8% on the Mar 14 re-buy), TER ranks 88th-151st on Sep 17 2025 with every version; #26 with META and TER forced onto the list on their setup days: META is not bought (all 6 slots full on Mar 3 2023), TER is bought Sep 18 2025 at $117.78 and makes +23.7% (1/3 at +8% on Sep 22, the rest by the climax exit on Nov 21 at $155.33): $941k (21.0% a year, -23.7%, Sharpe 1.28); on #26, 'twin' setups that share every trait META (Mar 3 2023) and TER (Sep 18 2025) had in common, off the list (`--twin-days`, backtest/twins.py): a valid entry with an SCTR at most 5 in the last 6 months, up 50+ points in 60 days, a gap up of 15%+ in the last 3 months, a golden cross at most 20 days old, 10-30% below the 52-week high, SCTR 65-90: 8 setups in 2015-2026 (BIIB Nov 2019 best +19%, META, ZS Jul 2023 +54%, SAIA Dec 2024 never up and -36%, TER, TECK Oct 2025 +39%, W and PLTR Sep 2026 too recent); added to the list on the setup evening only: $920k (20.8% a year, -23.7%, Sharpe 1.26; META not bought, all 6 slots full); for 10 / 20 / 30 trading days and first in line for a free slot (`--twin-days 10` / `20` / `30`, the same trades): $984k (21.5%, -23.7%, Sharpe 1.29, 329 trades, profit factor 3.38; 2023 +16.6% instead of +7.1%, 2024 +73.8% instead of +78.1%; META bought Mar 14 2023 at $187.58, +34.8%; ZS -1% then +42%, TECK -7% then +15%, BIIB +1% then -3%, SAIA -13%); 5 days (`--twin-days 5`): $920k (the Mar 14 2023 META setup is too late); 20 days but behind the top 10: $920k (TCOM and an SCCO re-buy take the two free slots on Mar 14 2023); 20 days counting every twin setup close, even when the next day's high misses the buy-stop: $900k (20.6%, Sharpe 1.24: BURL Feb 2024, SNAP Jun 2024 and RTO Aug 2024 fill later, -$3k together, and their slots push out HOOD's Jul-Sep 2024 trades); the rule is copied from META and TER, so only the other trades count as evidence; on #26's setups, straight-line SCTR rises (2015-2026, setups off the list, share reaching +50% within 6 months): a one-day SCTR jump of 50+ points from 40 or less in the 3 months before the setup, SCTR still 65+ (META Feb 2 2023, TER Jul 30 2025): 10.0% of 130 setups; a steady climb in the 10 days before the setup (up 15+ points, 7+ days up): 5.6% of 1,000; a steady climb within 10 days after the setup (up 4 of 5 days, 10+ points above the setup evening; META Mar 3-13 2023, TER Sep 18-22 2025): 37% of all setups, 6.8% at the setup price, 5.4% if bought when the climb shows; jump before and climb after: 14.3% of 28; top-10 setups: 25.7% of 334, all setups off the list 5.6%; on #26, a late entry (`--late-entry-days 10`: a stock in today's top 10 that gave a valid entry in the last 10 trading days while it was off the list, and is still in its ATR up-trend, is bought at the next open): $938k (21.0% a year, worst drawdown -27.0% Jul 2019 to Mar 2020, Sharpe 1.08, 551 trades, profit factor 2.39); the 228 late entries won 36%, +1.9% on average, +$144k in total (META Mar 2023 +30%, VRT Nov 2023 +70%, NVDA May 2016 +108%, INSM Jun 2025 +52%, but STX Jul 2026 -15%, HPE Jul 2026 -10%), all 6 slots are full on 49% of days; 5 / 20 days: $806k / $520k (19.4% / 15.1% a year, worst drop -29.5% / -32.8%); on #26, every stock with an SCTR above 90 on each of the last 21 trading days, at any rank, instead of the daily top 10 (about 21 candidates a day): $612k (16.7% a year, worst drawdown -28.4% Feb-Apr 2025, Sharpe 0.90, 569 trades, 41% winners, profit factor 2.06; all 6 slots full on 53% of days; 2020 +55% and 2024 +39% instead of +85% / +78%, 2026 +70% instead of +48%; still no TER in 2025 or SMCI in 2023); above 95 for the month: $673k (17.6%, -31.0%); the top 10 and above 90 for the month: $815k (19.6%, -23.7%, Sharpe 1.26); on #26, the climax exit from +20% instead of +40% (`--climax-gain 0.20`): $873k (20.3% a year, worst drawdown -21.7%, Sharpe 1.28, profit factor 3.52, 25 climax exits instead of 18; the new ones: SE Jul 2020 +14% instead of +67%, TWLO Jun 2020 +16% instead of +8%, UMC Jan 2021, PLTR Jun 2023, TWLO Jan 2025, TER Mar 2026 sold at the same Apr 30 open as the 50-day exit, INTC Jun 2026 +8% instead of 0%); with the quick re-buy after a climax exit too (`--climax-reentry`): $869k (20.2%, -21.8%, Sharpe 1.21; the 16 re-buys after climax exits made +$5k: CIEN Dec 2025 +24%, MU May 2026 +20% but WDC Jun 2026 -32%, ZM Oct 2020 -21%, COHR Jun 2026 -20%); +20% with a stretch of 30% instead of 50% (`--climax-stretch 0.30`) and re-buys: $770k (19.0%, -19.9%, 55 climax exits); #26 with only `--climax-reentry`: $832k (19.8%, -25.4%); on #26, `--list-memory 21` (also look for the setup in stocks that were on the list on any of the last 21 trading days, after today's top 10): $474k (14.2% a year, worst drawdown -51.2% from Feb 2021 to Oct 2023, Sharpe 0.77, 560 trades, 39% winners); 10 / 42 days: $711k / $462k (18.2% / 13.9% a year, worst drop -46.8% / -42.1%); the 245 extra trades (38% winners, +$151k in total, e.g. SMCI Dec 2023 +58%, PLTR Aug 2024 +113%) keep all 6 slots full on 55% of days (#26: about 10%), so 157 of #26's trades are never made (+$349k in #26, e.g. TSLA Apr 2020 +126%, ZM Aug 2020 +79%, MU Nov 2025 +60%), and 2021 / 2022 / 2023 lose 22% / 14% / 13% (VIPS Apr 2021 -36%, CLF Apr 2022 -32%); forgetting such a stock once its SCTR closes below 50 / 70 / 30 (`--list-memory-min-sctr`) hardly changes that: $534k / $519k / $507k (15.3% / 15.0% / 14.8% a year, worst drop -49.7% / -49.1% / -50.1%), because within a month of leaving the top 10 almost no stock falls that far (with 50, 241 of the 245 extra trades remain and all 6 slots are still full on 55% of days); requiring a run of at least 1 / 2 / 3 months on the list before the stock left (`--list-memory-min-run 21` / `42` / `63`, breaks of up to 5 trading days allowed, `--list-memory-gap`): $729k / $752k / $894k (18.4% / 18.7% / 20.5% a year, worst drop -39.6% / -31.1% / -24.0%, Sharpe 0.99 / 1.09 / 1.24); with 2 months only 42 extra trades remain, 29% winners, -$28k in total (SMCI Dec 2023 +58% but ROKU Mar 2021 -17%, MU Jul 2026 -15%, STX Jul 2026 -13%), and 2021 / 2022 / 2026 are -10% / -12% / +27% instead of -5% / -6% / +48%; adding that the setup close must be above the 100-day / 50-day SMA (`--list-memory-above-ma 100` / `50`): $739k / $834k (18.6% / 19.8% a year, worst drop -31.9% / -30.0%, Sharpe 1.10 / 1.18): with 100 days 32 extra trades remain (25% winners, -$39k), MU, STX and ROKU still pass, and SMCI Dec 2023 (+58%) is skipped because it was below its 100-day SMA; with 50 days 17 remain (29% winners, -$11k); the 100-day filter on the plain one-month memory: $658k (17.4%, -47.7%); on #26, a profit floor (`--profit-floor`: after +8%, once a trade has been up 20% at a close (`--profit-floor-after`), sell the rest at the next open after a close below the floor): +5% / +8% / +10% / +12% / +15%: $936k / $935k / $868k / $841k / $797k (21.0% / 21.0% / 20.2% / 19.9% / 19.3% a year, worst drop -24.2% / -20.3% / -19.2% / -19.2% / -19.1%, Sharpe 1.30 / 1.32 / 1.28 / 1.28 / 1.28) vs #26's $908k (20.7%, -23.7%, 1.27); +10% after +15% / +25% / +30%: $534k / $864k / $886k; the smaller 2021-22 drop with a floor of 8% or more comes mostly from selling SE in Aug 2020 at +8% (in #26 it ran to +146%): the Feb 2021 high is lower ($288k vs $314k with +10%) and the Sep 2022 low is no higher in dollars; between +8% and +10% a few trades (NU Oct 2023, CIEN Nov 2025) dipped to +8-10% before big runs, so the result depends on where those dips stopped; with +8%, 2023 is +10.4% (PLTR +12% / +6% / +5% instead of +4% / +4% / +2%) and TER / INTC Jun 2026 are sold at +2% / +8%; on #26, the 50-day exit from day 1 (`--rest-exit-delay 0`) / from day 10: $660k / $824k (17.4% / 19.7% a year, worst drop -24.3% / -22.3%), and without its 3% cushion (`--sma-exit-buffer 0`): $723k (18.4%, -21.0%); on #26, why buy orders go unfilled: of 377 order runs (2015-2026) 234 filled; in the 113 that never filled (stock not owned) the price first fell a median 10.6% below the order level, and when it crossed the red line again (a median 4 days later) no order was waiting: the stock had left the top 10 (85) or the squeeze had ended (20); 3 months later those stocks were a median 11% higher (69% higher, 30% up 20%+), e.g. SMCI Nov 2023 +161%; on #25, `--reentry-max-losses 3` (now the default, #26: stop re-buying a stock after 3 trades in a row, counting its setup buy, that made no more than +20%, `--reentry-win 0.20`; a new setup buy starts the count again): $908k (20.7% a year, worst drawdown -23.7%, Sharpe 1.27, 318 trades, 46% winners, profit factor 3.44); after 2 / 4 such trades: $835k / $811k (19.8% / 19.5% a year, worst drop -21.2% / -24.3%, Sharpe 1.29 / 1.20); a win above +10% / +30% instead of +20%: $911k / $901k; with 3, 2023 is +7.1% instead of +4.6% (SCCO bought 3 times instead of 9) and the worst drop inside 2023 -10.3% instead of -12.1%; it skips 43 of #25's re-buys (together +$17k in #25: AS Apr 2025 +38% and TER Sep 2026 +19% after 4+ losing trades, but also TER Jul 2026 -10% and GEV Oct 2025 -11%) and the freed slots buy NVDA in Jun 2016 (+86%) and INTC in Jul 2026 (+18%); on #25, the re-entry rule: no re-entry (`--reentry-days 0`) / only after exits before +8% (`--no-reentry-after-tp`) / only while the stock is in the daily top 10 (`--reentry-top 10`): $456k / $497k / $545k (13.8% / 14.6% / 15.5% a year, worst drop -22.0% / -21.7% / -22.0%, Sharpe 1.09 / 1.08 / 1.18) vs $855k with #25's re-entry after any exit; better than no re-entry in 9 of 12 years, worse in 2015, 2022 and 2023 (+4.6% vs +9.2%, the SCCO / TCOM / ROST re-buy chains); the 147 #25 re-entries made after the stock had left the top 10 made +$207k in total (APP Aug 2024 +205%, COHR Apr 2026, HOOD Sep 2024), the 34 made while it was still in the top 10 +$90k; on #25, `--climax-reentry` (allow the 5-day re-entry after a climax exit: 19.4% a year, worst drop -27.1%, Sharpe 1.10; the buy-backs made +$20k in total, SE Jun 2020 +71%, CIEN Mar 2026 +41% and MU May 2026 +20% but COHR, ZM, CIEN, MU and TSLA buy-backs lost, and the 2026 drop comes back; normal new setups already re-bought 5 of the 18 climax-exited stocks, e.g. MU in Mar 2026 +60%); on #26, the partial sale at +8% (`--take-profit-fraction`): 1/2 / 1/3 (default) / 1/4 / 1/5 / none (`0`, 1 share): $684k / $908k / $1.04M / $1.12M / $1.48M (17.8% / 20.7% / 22.0% / 22.9% / 25.8% a year, worst drawdown -20.8% / -23.7% / -25.1% / -25.9% / -29.1%, each Feb 2021 to Sep 2022, Sharpe 1.31 / 1.27 / 1.25 / 1.24 / 1.20), the same 318 trades; with 1/4 the trades that end at +50% or more make 9.4 points more on average (APP Aug 2024 +230% instead of +205%, TSLA Apr 2020 +141% instead of +126%), losing trades 0.3 points less (STLD Dec 2022 -2.0% instead of -0.9%), the worst drop inside each year is 1-2 points deeper, 2020 / 2024 / 2026 are +93.6% / +86.3% / +52.4% instead of +85.0% / +78.1% / +47.7% and 2018 -9.3% instead of -7.5%; on #25, the partial sale at +8% (`--take-profit-fraction`): 1/2 / 1/3 (default) / 1/4 / 1/5 / none: 17.3% / 20.0% / 21.3% / 22.0% / 24.9% a year, worst drop -20.7% / -23.9% / -25.3% / -26.2% / -29.6%, Sharpe 1.25 / 1.21 / 1.19 / 1.18 / 1.15; taking the 1/3 later is worse on both counts (`--take-profit 0.12` / `0.15`: 18.3% / 17.6% a year, worst drop -31.2% / -29.3%); on #24, a big red candle as a second climax trigger (gap down OR ...): `--climax-drop 0.07` (a close 7%+ below the previous close: 20.4% a year, worst drop -25.2%, Sharpe 1.20, Sep 2020 drop -11.2% instead of -15.4%, 2026 drop -12.9%; it sells TSLA on Sep 4 2020 at +182% but HOOD in Dec 2024 at +62% instead of +106%), `--climax-drop 0.10` (20.6%, -25.3%), `--climax-body 0.06` / `0.08` (a close 6% / 8%+ below the day's open: 20.0% / 20.7% a year, worst drop -23.9% / -25.3%, Sharpe 1.21 / 1.18; 6% also sells SE in Jun 2020 at +89% instead of +266%; 5% / 7%: 19.4% / 19.8% a year, worst drop -23.8% / -24.0%, so the body trigger is stable around 5-7%; `--climax-drop 0.06` / `0.08`: 18.9% / 20.3%, worst drop -23.5% / -25.2%), `--climax-drop-atr 2` / `2.5` / `3` (a fall of 2 / 2.5 / 3 x the 14-day ATR: 20.7% / 21.0% / 21.8% a year, worst drop -25.3% / -25.3% / -28.2%, no help in Sep 2020); on #23: `--climax-stretch 0.40 --climax-ma 50 --climax-gap 0` (climax exit: once a trade has been up 40% and closed 40%+ above its 50-day SMA within 10 days, a gap down below the previous low sells everything at the next open, no re-entry: 18.0% a year, worst drop -24.0%, Sharpe 1.09; the 2026 give-back drop shrinks from -27.1% to -14.9% and 2025's from -25.0% to -17.1%, but it sells SE and TSLA in Jul 2020 and APP in Oct 2024 on their first climax; stretch 30%: 16.8% / -24.4%, 25%: 14.7% / -23.6%, 20%: 13.7% / -25.9%; with `--climax-reentry` 17.6% / -24.3%; `--climax-stretch 0.60` (gap of at least 5% below the previous low, after a close 60%+ above the 100-day SMA): 22.1% a year, worst drop -28.4% (the 2021-22 grind, untouched), Sharpe 1.16, only 5 climax exits, all in 2024-2026 (DELL, HOOD, APP, MU, WDC): the 2026 drop shrinks from -27.1% to -16.0% and 2025's from -25.0% to -20.5%; nearby settings: gap 3% 20.3%, gap 7% 22.6%, stretch 50% 21.7% (2025 drop -14.8%), stretch 70% 21.7%), `--market-ma 200` / `100` / `50` (no new buys unless SPY and QQQ are above their average: 11.5% / 5.7% / 1.8% a year, worst drop -22.1% / -18.4% / -18.6%; with 200 days it skips 126 trades worth $335k, e.g. SE and TSLA in spring 2020, PLTR and AS Apr 2025, MU Apr 2026: the setup buys pullbacks, which come with market dips); on #22 (`--breakeven-exit --no-reentry-after-tp`): re-entry after +8% exits too (19.2% a year, worst drop -27.6%, profit factor 2.87, 342 trades; the re-entries include APP Aug 2024 +223%, SOFI Apr 2025 +101%, MU Apr 2026 +92%, PLTR Apr 2025 +69% and WDC Apr 2026 +51%), both changes = #23 (20.9% a year, worst drop -28.4%, profit factor 2.91, 350 trades; more trades give back the +8% and end in a loss, 40 vs 36, e.g. TSLA Mar 2020 -17%, RIVN Sep 2023 -15%; adding `--red-line-stop 0.02` / `0.03` / `0.05` (no green line yet: sell 2% / 3% / 5% below the previous close's red line): 16.3% / 14.3% / 12.3% a year, worst drop -27.7% / -28.3% / -32.1%; with 3%, 71% of the 117 stopped trades were bought back within 5 days (APP, SE, WDC, TLN), but PLTR Mar 2025 (+63%), AS Apr 2025 (+38%) and COHR Apr 2026 (+22%) were not; adding `--emergency-stop 0.12`: 16.0% a year, worst drop -33.9%, 64 stop-outs, 34 of the 50 stopped trades that also exist without the stop would have ended better without it, and -8% / -10% / -15% / -20% stops give 13.5% / 14.8% / 17.0% / 16.8% a year), no break-even exit alone (14.4% a year, worst drop -23.9%), `--reentry-days 7` (12.8% a year, worst drop -29.5%; with re-entry after +8% 17.2% / -39.1%, and without the break-even exit too 20.0% / -40.9%; on #23 the re-entry window 0 / 2 / 3 / 4 / 6 / 7 days gives 13.8% / 14.6% / 16.4% / 18.6% / 19.9% / 20.0% a year with worst drops of -24% to -41%); other volatility limits (#22's 75% limit vs none (#21): 2015-2020 about even (14.8% vs 15.0% a year), 2021-2026 better (11.4% vs 7.3%); it skips LITE Jan 2026 (+84%) but also the XXI, MSTR, NBIS, SMCI and COIN losses; any limit from 60% to 90% gives 12.2-13.3% a year with a worst drop of -21% to -23.5% (50%: 5.8%, 100%: 11.5%), about the same as no limit in 2015-2020 (14.8-15.3% vs 15.0% a year) and better in 2021-2026 (10.0-11.7% vs 7.3%); a minimum (`--min-vol`) of 20-30% changes little (`--min-vol 30 --max-vol 80`: 12.2%, -22.5%), 35-40% costs 2-3% a year; for comparison, a 1-year beta of 1-2.5 on the top 10: 10.4% a year, worst drop -21.5%), `--extra-slots 1` / `2` (the top 10 first, plus at most 1 / 2 positions in other stocks with an SCTR above 90: 9.3% / 11.3% a year, worst drop -41.9% / -42.4%, profit factor 1.53 / 1.60; the extra trades averaged -0.6% / +0.8% and with 2 slots lost $46k in 2021-2022; neither caught ILMN in Aug 2026, the extra slots were busy with ROIV and RVMD), `--top 1000` (every stock with an SCTR above 90, about 90 a day: 8.8% a year, worst drop -42.1%, profit factor 1.37; it catches ILMN in Aug 2026 (+27%), but the 6 slots are nearly always full, so 214 of the 302 top-10 trades are skipped, among them SE Mar 2020, CIEN Nov 2025 and LITE Jan 2026), `--frozen-line` (no-line entries: the green line the buy day would have had, frozen as the stop until the trend really turns up: 9.1% a year, worst drop -32.0%; the frozen stop sits a median 7.9% below the entry and sold 56 trades at -8.6% on average), `--reentry-top 10` / `20` (re-entries only while the stock is in the daily top 10 / 20: 11.1% / 10.8% a year, worst drop -25.1% / -25.2%, vs 11.4% and -26.3% without the requirement), `--entry-check live` (a buy order waits at the flip level for every top-10 stock in an ATR down-trend; it buys at the first price on the way up through the line where the squeeze is on and the bar red, as the indicators read live: 12.2% a year, worst drop -28.1%, profit factor 1.78 with the break-even exit at the entry; 11.7% / -30.5% with the 3% cushion), `--entry-check either` (11.7% / -29.0% at the entry, 11.1% / -31.3% with the cushion), `--close-entry` (also buy at the next open when the squeeze, red bar and ATR flip to BUY all appear on the same bar, e.g. HOOD 9 Apr 2025: 11.9% a year but worst drop -36.8% and profit factor 1.64; the 89 extra trades lost $41k, 43% winners), `--atr-stop-buffer` 0 / 0.03 / 0.05 (safety net on the line / 3% / 5% below it: 10.0% / 11.3% / 11.3% a year, the 5% one is version #17), `--dim-green-days 42` (within about 2 months of a full-setup entry we have exited, also buy the same setup with a dim green momentum bar: 12.3% a year, worst drop -26.3%, but only 7 extra trades and most of the gain is one trade, WDC Apr 2026), `--failed-breakout-exit` (sell at the next open when the entry day closes back below the flip level: 5.4% a year), `--emergency-stop 0.12` on #17 (6.6% a year), `--entry-buffer 0.02`, `--no-close-exit --sticky-stop --atr-stop-buffer 0.03` (#14), `--emergency-stop 0.12` (#15), `--no-atr-exit-on-touch --atr-stop-buffer 0` (version G), `--park QQQ`, `--rebuy-shakeouts` |

2015-01-02 to 2026-10-02 with the defaults (#26): $100k -> $908k (20.7% a year,
worst drawdown -23.7%, Feb 2021 to Sep 2022, back at the high in Mar 2024, Sharpe 1.27,
318 trades, 46% winners, profit factor 3.44) vs SPY 13.8% a year (worst drawdown -33.7%);
2023 +7.1%, the Jun-Jul 2026 drop -11.7% (back at the high in Sep 2026), but Jul 2019 to
Mar 2020 -19.2% (#25: -16.8%). #25 (`--reentry-max-losses 0`): $855k (20.0% a year,
worst drawdown -23.9%, Sharpe 1.21, 354 trades, 44% winners, profit factor 2.99). Its
climax exit fires 18 times; the Sep 2020 drop is -10.7% (#24: -15.4%) and 2026's -12.9%.
#24 (gap down only, `--climax-body 0`): $1.00M (21.7% a year, worst drawdown -28.2%,
Sharpe 1.18, 349 trades, 44% winners, profit factor 3.15); its climax exit fired 7 times
(ZM 2020, DELL 2024, APP, HOOD, CIEN, MU, WDC) and cut the 2026 drop from -27.1% to -16.2%
and 2025's from -25.0% to -14.8%. #23 (`--climax-gain 0`): $934k (20.9%
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

Alternative to revisit later: #26 with `--twin-days 20` (also buy setups outside the top
10 that look like META's in Mar 2023 and TER's in Sep 2025, see `backtest/twins.py`; the
stock stays on the list for 20 trading days and goes ahead of the top 10 when slots are
short): $984k (21.5% a year, worst drawdown -23.7%, Sharpe 1.29, 329 trades, profit factor
3.38) vs #26's $908k. It finds 8 twins in 2015-2026 and trades 7 (PLTR Sep 2026: no free
slot): META Mar 2023 +34.8%, TER Sep 2025 +23.7%, W Sep 2026 +3.1% (open), BIIB Nov 2019
+1.2%, ZS Jul 2023 -1.3% (its re-buy +41.7%), TECK Oct 2025 -7.3% (re-buy +14.9%), SAIA
Dec 2024 -13.4%. The rule is copied from META and TER, so only the other trades count as
evidence, and it needs every stock's chart, so the run takes a few minutes longer.

Backup to revisit later: #26 with `--list-memory 21 --list-memory-min-run 63` (also
buy the setup in a stock that dropped out of the top 10 within the last month, if it
had been in the top 10 for 3 months in a row before, breaks of up to 5 trading days
allowed): $894k (20.5% a year, worst drawdown -24.0%, Sharpe 1.24, 344 trades, profit
factor 2.93) vs #26's $908k; it adds 18 trades (39% winners, +$3k in total, e.g. SMCI
Dec 2023 +58%) and keeps 2024 at +90% (#26: +78%) but 2026 at +34% (#26: +48%).

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
