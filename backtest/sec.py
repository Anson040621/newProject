"""Share-count history from SEC filings (free XBRL API at data.sec.gov).

Every 10-K/10-Q cover page reports the number of shares outstanding
(dei:EntityCommonStockSharesOutstanding). Companies with several share classes
often report it per class only, so we fall back to the balance-sheet figure
(us-gaap:CommonStockSharesOutstanding) when the cover-page value is missing or
stale. Foreign companies (20-F filers) report ordinary shares rather than ADRs;
marketcap.py rescales those to match Nasdaq's market cap.

The SEC asks automated tools to identify themselves with a User-Agent of the
form "app-name contact@email.com" and to stay under 10 requests per second.
Set it in the SEC_USER_AGENT environment variable.
"""

from __future__ import annotations

import gzip
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .universe import DATA_DIR

SHARES_FILE = DATA_DIR / "shares.csv"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/{concept}.json"
COVER_PAGE = "dei/EntityCommonStockSharesOutstanding"
BALANCE_SHEET = "us-gaap/CommonStockSharesOutstanding"
SHARE_COLUMNS = ["ticker", "end", "filed", "shares", "form"]


class _RateLimiter:
    def __init__(self, per_second: float):
        self.interval = 1.0 / per_second
        self.lock = threading.Lock()
        self.next_slot = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            delay = self.next_slot - now
            self.next_slot = max(now, self.next_slot) + self.interval
        if delay > 0:
            time.sleep(delay)


_limiter = _RateLimiter(per_second=8)


def fetch_json(url: str, user_agent: str, retries: int = 5):
    """GET a JSON document from the SEC. Returns None on 404."""
    request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip"})
    for attempt in range(retries):
        _limiter.wait()
        try:
            with urllib.request.urlopen(request, timeout=60) as resp:
                body = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                return json.loads(body)
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return None
            if err.code not in (403, 429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
        time.sleep(2**attempt)
    return None


def load_cik_map(user_agent: str) -> dict[str, int]:
    """Current ticker -> SEC CIK number, with tickers in BRK.B style."""
    data = fetch_json(TICKERS_URL, user_agent)
    return {row["ticker"].replace("-", "."): int(row["cik_str"]) for row in data.values()}


def parse_concept(payload: dict | None) -> pd.DataFrame:
    """Share counts from one companyconcept response: end (as-of date), filed, shares, form."""
    if not payload:
        return pd.DataFrame(columns=["end", "filed", "shares", "form"])
    rows = payload.get("units", {}).get("shares", [])
    frame = pd.DataFrame(
        {
            "end": pd.to_datetime([r["end"] for r in rows]),
            "filed": pd.to_datetime([r["filed"] for r in rows]),
            "shares": [float(r["val"]) for r in rows],
            "form": [r.get("form", "") for r in rows],
        }
    )
    frame = frame[frame["shares"] > 0]
    return frame.sort_values(["filed", "end"]).drop_duplicates(["end", "filed"], keep="last").reset_index(drop=True)


def company_shares(cik: int, user_agent: str, stale_after_days: int = 400) -> pd.DataFrame:
    """Share history for one company: cover-page values, or balance-sheet values if those are fresher."""
    cover = parse_concept(fetch_json(CONCEPT_URL.format(cik=cik, concept=COVER_PAGE), user_agent))
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=stale_after_days)
    if not cover.empty and cover["filed"].max() >= cutoff:
        return cover
    balance = parse_concept(fetch_json(CONCEPT_URL.format(cik=cik, concept=BALANCE_SHEET), user_agent))
    if balance.empty or (not cover.empty and cover["filed"].max() >= balance["filed"].max()):
        return cover
    return balance


def download_shares(tickers, user_agent: str, path: Path = SHARES_FILE, workers: int = 6) -> pd.DataFrame:
    """Share history for every ticker the SEC knows, saved as one long CSV."""
    tickers = list(tickers)
    cik_map = load_cik_map(user_agent)
    by_cik: dict[int, list[str]] = {}
    for ticker in tickers:
        if ticker in cik_map:
            by_cik.setdefault(cik_map[ticker], []).append(ticker)
    print(f"   {sum(len(v) for v in by_cik.values())} of {len(tickers)} tickers have an SEC CIK "
          f"({len(by_cik)} companies)", flush=True)

    def fetch(cik):
        try:
            return cik, company_shares(cik, user_agent)
        except Exception as err:  # keep going; a missing company just uses today's share count
            print(f"   CIK {cik}: {err}", flush=True)
            return cik, None

    frames = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (cik, shares) in enumerate(pool.map(fetch, by_cik), start=1):
            if shares is not None and not shares.empty:
                for ticker in by_cik[cik]:
                    frames.append(shares.assign(ticker=ticker))
            if done % 250 == 0:
                print(f"   {done}/{len(by_cik)} companies", flush=True)

    result = pd.concat(frames, ignore_index=True)[SHARE_COLUMNS] if frames else pd.DataFrame(columns=SHARE_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, date_format="%Y-%m-%d")
    return result


def load_shares(path: Path = SHARES_FILE) -> pd.DataFrame:
    """Saved share history (empty if it was never downloaded)."""
    if not path.exists():
        return pd.DataFrame(columns=SHARE_COLUMNS)
    return pd.read_csv(path, parse_dates=["end", "filed"])
