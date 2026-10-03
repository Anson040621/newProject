import numpy as np
import pandas as pd
import pytest

from backtest.download_data import candidate_tickers
from backtest.listings import normalize_symbol, parse_listings, short_name
from backtest.marketcap import (
    drop_glitches,
    estimate_market_caps,
    large_cap_mask,
    large_cap_universe,
    shares_known_on,
    split_factors_after,
)
from backtest.sec import parse_concept


def listing_row(symbol, price, cap):
    return {
        "symbol": symbol, "name": f"{symbol.strip()} Inc.", "lastsale": price, "netchange": "0", "pctchange": "0%",
        "volume": "1", "marketCap": cap, "country": "United States", "ipoyear": "", "industry": "x", "sector": "y",
        "url": "",
    }


def test_parse_listings():
    payload = {"data": {"rows": [
        listing_row("BRK/B", "$500.50", "1,104,259,000,000.00"),
        listing_row("ABR^D", "$20.00", ""),            # preferred share: dropped
        listing_row("ECC  ", "$8.00", "210,499,000.00"),  # trailing spaces in symbol
        listing_row("ZERO", "$1.00", "0.00"),            # no market cap: dropped
    ]}}
    table = parse_listings(payload)
    assert list(table.index) == ["BRK.B", "ECC"]
    assert table.loc["BRK.B", "market_cap"] == pytest.approx(1.104259e12)
    assert table.loc["BRK.B", "last_price"] == pytest.approx(500.50)
    assert normalize_symbol(" BF/B ") == "BF.B"


@pytest.mark.parametrize("name,kept", [
    # Not common equity, but listed with the parent company's market cap: dropped
    ("AGNC Investment Corp. Depositary Shares each representing a 1/1000th interest in a share of 6.50% "
     "Series E Fixed-to-Floating Cumulative Redeemable Preferred Stock", False),
    ("Apollo Global Management Inc. 7.625% Fixed-Rate Resettable Junior Subordinated Notes due 2053", False),
    ("Alphabet Inc. Depositary Shares representing a 1/20th Interest in a Share of Series A Mandatory Convertible", False),
    ("SLM Corporation Floating Rate Non-Cumulative Preferred Stock Series B", False),
    ("BrightSpring Health Services Inc. Tangible Equity Unit", False),
    ("Southern Company (The) 2025 Series A Corporate Units", False),
    ("Comcast Holdings ZONES", False),
    # Real stocks: kept
    ("Microchip Technology Incorporated Common Stock", True),
    ("Itau Unibanco Banco Holding SA American Depositary Shares (Each repstg 500 Preferred shares)", True),
    ("Alibaba Group Holding Limited American Depositary Shares each representing eight Ordinary share", True),
    ("Energy Transfer LP Common Units", True),
    ("Bank Nova Scotia Halifax Pfd 3 Ordinary Shares", True),
])
def test_parse_listings_drops_non_common_securities(name, kept):
    row = listing_row("XYZ", "$25.00", "30,000,000,000.00")
    row["name"] = name
    assert ("XYZ" in parse_listings({"data": {"rows": [row]}}).index) == kept


def test_parse_concept_dedupes_and_sorts():
    payload = {"units": {"shares": [
        {"end": "2020-04-15", "val": 110, "filed": "2020-05-01", "form": "10-Q"},
        {"end": "2020-01-15", "val": 100, "filed": "2020-02-01", "form": "10-K"},
        {"end": "2020-01-15", "val": 100, "filed": "2020-02-01", "form": "10-K"},  # duplicate
        {"end": "2020-07-15", "val": 0, "filed": "2020-08-01", "form": "10-Q"},    # bad value
    ]}}
    frame = parse_concept(payload)
    assert list(frame["shares"]) == [100.0, 110.0]
    assert list(frame["filed"].dt.strftime("%Y-%m-%d")) == ["2020-02-01", "2020-05-01"]
    assert parse_concept(None).empty


def test_split_factors_after():
    splits = pd.Series([4.0, 10.0], index=pd.to_datetime(["2020-08-31", "2024-06-10"]))
    as_of = pd.Series(pd.to_datetime(["2019-01-01", "2021-01-01", "2024-06-10", "2025-01-01"]))
    assert list(split_factors_after(as_of, splits)) == [40.0, 10.0, 1.0, 1.0]
    assert list(split_factors_after(as_of, pd.Series(dtype=float))) == [1.0] * 4


def test_shares_known_on_is_point_in_time_and_split_adjusted():
    filings = pd.DataFrame({
        "end": pd.to_datetime(["2020-01-15", "2020-04-15", "2020-07-15"]),
        "filed": pd.to_datetime(["2020-02-01", "2020-05-01", "2020-08-01"]),
        "shares": [100.0, 205.0, 205_000.0],  # last one reported in thousands by mistake
    })
    splits = pd.Series([2.0], index=pd.to_datetime(["2020-03-02"]))  # 2-for-1 between the filings
    index = pd.bdate_range("2020-01-02", "2020-09-30")
    shares = shares_known_on(filings, splits, index, current=205.0)
    assert shares["2020-01-02"] == 200.0   # before the first filing: first value, split-adjusted
    assert shares["2020-04-30"] == 200.0   # second filing not public yet
    assert shares["2020-05-01"] == 205.0
    assert shares["2020-09-30"] == 205.0   # bad latest filing ignored: today's count agrees with the previous one


def test_drop_glitches_keeps_lasting_changes():
    # An isolated typo in the middle is dropped.
    assert list(drop_glitches(np.array([100.0, 100_000.0, 101.0, 102.0]))) == [True, False, True, True]
    # A merger that multiplies the share count (QXO, TeraWulf) is kept: every later filing agrees.
    merger = np.array([4.5, 4.6, 4.7, 400.0, 420.0, 430.0])
    assert drop_glitches(merger, current=430.0).all()
    # A jump in the latest filing is kept when today's share count confirms it...
    assert drop_glitches(np.array([4.5, 4.6, 450.0]), current=455.0).all()
    # ...and dropped when today's share count says it did not happen.
    assert list(drop_glitches(np.array([4.5, 4.6, 4600.0]), current=4.7)) == [True, True, False]


def test_estimate_market_caps_calibrates_to_nasdaq():
    index = pd.bdate_range("2024-01-01", periods=5)
    close = pd.DataFrame({"US": 10.0, "ADR": 20.0, "FLAT": 5.0, "NONE": 1.0}, index=index)
    splits = pd.DataFrame(0.0, index=index, columns=close.columns)
    shares = pd.DataFrame({
        "ticker": ["US", "US", "ADR"],
        "end": pd.to_datetime(["2023-06-30", "2024-01-02", "2023-12-31"]),
        "filed": pd.to_datetime(["2023-08-01", "2024-01-03", "2024-01-02"]),
        "shares": [100.0, 200.0, 500.0],
    })
    listings = pd.DataFrame(
        {"market_cap": [2000.0, 2000.0, 250.0], "last_price": [10.0, 20.0, 5.0]}, index=["US", "ADR", "FLAT"]
    )
    caps = estimate_market_caps(close, splits, shares, listings)
    # US: SEC history (100 -> 200 shares) already matches Nasdaq's 200 shares today.
    assert list(caps["US"]) == [1000.0, 1000.0, 2000.0, 2000.0, 2000.0]
    # ADR: SEC reports 500 ordinary shares, Nasdaq implies 100 ADRs -> scaled to 100.
    assert list(caps["ADR"]) == [2000.0] * 5
    # No SEC history: today's share count (50) for every date.
    assert list(caps["FLAT"]) == [250.0] * 5
    # Nothing known: NaN.
    assert caps["NONE"].isna().all()


def monthly(index, values_by_month):
    return pd.Series([values_by_month[d.month] for d in index], index=index, dtype=float)


def test_large_cap_mask_three_month_rule():
    index = pd.bdate_range("2021-01-01", "2021-07-30")
    nan = np.nan
    mcap = pd.DataFrame({
        # large at first sight, then below $10B for 3 month ends -> out from May
        "A": monthly(index, {1: 12, 2: 8, 3: 8, 4: 8, 5: 8, 6: 8, 7: 8}),
        # above the line, back below, then 3 months above -> in from July only
        "B": monthly(index, {1: 8, 2: 12, 3: 8, 4: 12, 5: 12, 6: 12, 7: 12}),
        # first listed in March at $15B -> in from April
        "C": monthly(index, {1: nan, 2: nan, 3: 15, 4: 15, 5: 15, 6: 15, 7: 15}),
        # delisted after March -> out from May
        "D": monthly(index, {1: 15, 2: 15, 3: 15, 4: nan, 5: nan, 6: nan, 7: nan}),
    })
    mask = large_cap_mask(mcap, threshold=10)
    first_days = ["2021-02-01", "2021-03-01", "2021-04-01", "2021-05-03", "2021-06-01", "2021-07-01"]
    by_month = {t: [bool(mask.at[pd.Timestamp(d), t]) for d in first_days] for t in mcap.columns}
    assert by_month["A"] == [True, True, True, False, False, False]
    assert by_month["B"] == [False, False, False, False, False, True]
    assert by_month["C"] == [False, False, True, True, True, True]
    assert by_month["D"] == [True, True, True, False, False, False]
    # Nothing is decided before the first month end, and a decision starts the next day.
    assert not mask.loc[:"2021-01-29"].any().any()
    assert not mask.at[pd.Timestamp("2021-01-29"), "A"] and mask.at[pd.Timestamp("2021-02-01"), "A"]


def test_large_cap_universe_falls_back_to_sp500_membership():
    index = pd.bdate_range("2021-01-01", "2021-03-31")
    close = pd.DataFrame({"OLD": 50.0, "BIG": 100.0}, index=index)
    splits = pd.DataFrame(0.0, index=index, columns=close.columns)
    no_shares = pd.DataFrame(columns=["ticker", "end", "filed", "shares", "form"])
    listings = pd.DataFrame({"market_cap": [20e9], "last_price": [100.0]}, index=["BIG"])
    membership = pd.Series([frozenset({"OLD", "BIG"}), frozenset({"BIG"})],
                           index=pd.to_datetime(["2020-01-01", "2021-03-01"]))
    mcap, mask = large_cap_universe(close, splits, no_shares, listings, membership)
    assert mcap["OLD"].isna().all()
    assert mask.at[pd.Timestamp("2021-02-15"), "OLD"] and not mask.at[pd.Timestamp("2021-03-15"), "OLD"]
    assert mask.at[pd.Timestamp("2021-02-15"), "BIG"]


def test_candidate_tickers():
    listings = pd.DataFrame({"market_cap": [5e9, 5e8]}, index=["MID", "TINY"])
    membership = pd.Series([frozenset({"OLD"})], index=pd.to_datetime(["2015-01-01"]))
    assert candidate_tickers(listings, membership, "2016-01-01", "2020-01-01", min_cap=1e9) == ["MID", "OLD"]


@pytest.mark.parametrize("name,short", [
    ("Moderna Inc. Common Stock", "Moderna Inc."),
    ("Everpure Inc. Class A common stock", "Everpure Inc."),
    ("United Microelectronics Corporation (NEW) Common Stock", "United Microelectronics Corporation"),
    ("Alibaba Group Holding Limited American Depositary Shares each representing eight Ordinary share",
     "Alibaba Group Holding Limited"),
    ("Alphabet Inc. Class C Capital Stock", "Alphabet Inc."),
    ("Energy Transfer LP Common Units", "Energy Transfer LP"),
    ("SAP  SE ADS", "SAP SE"),
    ("Taiwan Semiconductor Manufacturing Company Ltd.", "Taiwan Semiconductor Manufacturing Company Ltd."),
])
def test_short_name(name, short):
    assert short_name(name) == short
