"""Current US stock listings with market caps, from the Nasdaq stock screener.

Covers every stock on NYSE, Nasdaq and NYSE American, including foreign ADRs
(e.g. UMC), which StockCharts also counts in its US large-cap universe.
For companies with several share classes (GOOG/GOOGL, FOX/FOXA) Nasdaq reports
the whole company's market cap on each class.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import pandas as pd

from .universe import DATA_DIR

LISTINGS_FILE = DATA_DIR / "listings.csv"

# Securities that are not common equity but show the parent company's market
# cap in the screener: preferred shares, notes, warrants, rights and units.
# Partnership units (ET, MPLX) and ADRs that represent preferred shares (ITUB,
# CIB) are a company's main traded equity and are kept.
NON_COMMON = (
    r"\d+(?:\.\d+)?\s?%"                           # coupon rate: preferreds and notes
    r"|\bNotes?\b|Debenture|Subordinated"
    r"|\bWarrants?\b|\bRights\b"
    r"|Preferred Stock|Preferred Securities|Preferred Units|Mandatory Convertible"
    r"|Interest in a Share"                           # depositary shares of a preferred issue
    r"|Corporate Units|Equity Units?\b|Tangible Equity|\bZONES\b"
)
NASDAQ_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&download=true"


def normalize_symbol(symbol: str) -> str:
    """Nasdaq writes share classes as BRK/B; we use BRK.B (Yahoo: BRK-B, SEC: BRK-B)."""
    return symbol.strip().replace("/", ".")


SHARE_DESCRIPTION = (
    r"\s+(?:\(NEW\)\s*)?(?:Class [A-Z]\s+)?"
    r"(?:Common Stock|Ordinary Shares?|American Deposit[ao]ry Shares?|ADS\b|Sponsored ADR|Common Shares"
    r"|Common Units|Limited Partnership Units|N\.?Y\.? Registry Shares|New York Registry Shares|Capital Stock"
    r"|Subordinate voting shares|Shares of Beneficial Interest).*$"
)


def short_name(name: str) -> str:
    """Company name without the share description: 'Moderna, Inc. Common Stock' -> 'Moderna, Inc.'"""
    name = re.sub(SHARE_DESCRIPTION, "", str(name), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name.replace("(NEW)", "")).strip()


def _number(text: pd.Series) -> pd.Series:
    return pd.to_numeric(text.astype(str).str.replace(r"[$,%]", "", regex=True), errors="coerce")


def parse_listings(payload: dict) -> pd.DataFrame:
    """Turn the screener's JSON into a table indexed by ticker.

    Preferred shares, notes, warrants, units (see NON_COMMON) and rows without a
    market cap are dropped.
    """
    rows = pd.DataFrame(payload["data"]["rows"])
    table = pd.DataFrame(
        {
            "ticker": rows["symbol"].map(normalize_symbol),
            "name": rows["name"].str.strip(),
            "last_price": _number(rows["lastsale"]),
            "market_cap": _number(rows["marketCap"]),
            "country": rows["country"],
            "sector": rows["sector"],
            "industry": rows["industry"],
        }
    )
    common = ~table["ticker"].str.contains(r"\^") & ~table["name"].str.contains(NON_COMMON, case=False, regex=True)
    table = table[common & (table["market_cap"] > 0) & (table["last_price"] > 0)]
    return table.drop_duplicates("ticker").set_index("ticker").sort_index()


def download_listings(path: Path = LISTINGS_FILE, url: str = NASDAQ_SCREENER_URL) -> pd.DataFrame:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 sctr-backtest", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as resp:
        table = parse_listings(json.load(resp))
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path)
    return table


def load_listings(path: Path = LISTINGS_FILE) -> pd.DataFrame:
    if not path.exists():
        return download_listings(path)
    return pd.read_csv(path, index_col="ticker")
