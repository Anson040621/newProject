"""Daily price data: download from Yahoo Finance or load from CSV files.

Prices are stored as one CSV per ticker in data/prices/<TICKER>.csv. Downloaded
files have these columns:

    Date, Open, High, Low, Close, Adj Close, Volume, Stock Splits

- Close is adjusted for splits only (used for market cap).
- Adj Close is adjusted for splits and dividends (used for SCTR indicators).
- Stock Splits is the split ratio on its ex-date (10.0 = 10-for-1), else 0.

You can drop your own CSV exports (TradingView, broker, etc.) into that folder
instead of downloading; they only need a Date column and a Close column.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .universe import DATA_DIR

PRICES_DIR = DATA_DIR / "prices"
COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Stock Splits"]


def yahoo_symbol(ticker: str) -> str:
    """We use BRK.B style; Yahoo uses BRK-B."""
    return ticker.replace(".", "-")


def download_prices(tickers, start, end=None, prices_dir: Path = PRICES_DIR, batch_size: int = 50) -> list[str]:
    """Download daily prices and split history from Yahoo. Returns tickers with no data."""
    prices_dir.mkdir(parents=True, exist_ok=True)
    tickers = list(tickers)
    missing = []
    for i in range(0, len(tickers), batch_size):
        missing += _download_batch(tickers[i : i + batch_size], start, end, prices_dir, threads=True)
        print(f"  downloaded {min(i + batch_size, len(tickers))}/{len(tickers)}", flush=True)
    if missing:
        # Parallel downloads sometimes fail on yfinance's cache ("database is
        # locked"); retry the failures one at a time.
        print(f"  retrying {len(missing)} failed tickers one at a time", flush=True)
        missing = _download_batch(missing, start, end, prices_dir, threads=False)
    if end is None:
        filled = fill_latest_session([t for t in tickers if t not in missing], prices_dir)
        if filled:
            print(f"  filled the latest close from Yahoo quotes for {filled} tickers", flush=True)
    return missing


def _session_quote(ticker: str) -> dict | None:
    """Official close of the most recent *finished* regular session, from Yahoo's chart metadata.

    Yahoo's daily bar for the latest session is sometimes empty for hours after
    the close, while the metadata already holds the official closing price.
    """
    import json
    import urllib.request

    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{yahoo_symbol(ticker)}?interval=1d&range=5d"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as resp:
                meta = json.load(resp)["chart"]["result"][0]["meta"]
            break
        except Exception:
            if attempt == 3:
                return None
            time.sleep(2**attempt)
    session = meta.get("currentTradingPeriod", {}).get("regular", {})
    price, when = meta.get("regularMarketPrice"), meta.get("regularMarketTime")
    if not price or not when or not session.get("end") or when < session["end"]:
        return None  # session still open, or no data
    day = pd.Timestamp(session["start"], unit="s", tz="UTC").tz_convert(meta.get("exchangeTimezoneName", "America/New_York"))
    return {
        "Date": day.tz_localize(None).normalize(),
        "Open": float("nan"),
        "High": meta.get("regularMarketDayHigh", float("nan")),
        "Low": meta.get("regularMarketDayLow", float("nan")),
        "Close": price,
        "Adj Close": price,  # the latest close is never dividend-adjusted
        "Volume": meta.get("regularMarketVolume", float("nan")),
        "Stock Splits": 0.0,
    }


def fill_latest_session(tickers, prices_dir: Path = PRICES_DIR, workers: int = 8) -> int:
    """Append the latest session's close to files that are missing it. Returns how many were filled.

    Only tickers that traded in the last 10 days are checked (delisted ones are skipped).
    """
    from concurrent.futures import ThreadPoolExecutor

    last_dates = {}
    for ticker in tickers:
        path = prices_dir / f"{ticker}.csv"
        if path.exists():
            last_dates[ticker] = pd.Timestamp(pd.read_csv(path, usecols=[0]).iloc[-1, 0])
    if not last_dates:
        return 0
    newest = max(last_dates.values())
    recent = [t for t, d in last_dates.items() if d >= newest - pd.Timedelta(days=10)]

    filled = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for ticker, quote in zip(recent, pool.map(_session_quote, recent)):
            if quote is None or quote["Date"] <= last_dates[ticker]:
                continue
            row = pd.DataFrame([quote]).set_index("Date")[COLUMNS]
            row.to_csv(prices_dir / f"{ticker}.csv", mode="a", header=False)
            filled += 1
    return filled


def _download_batch(batch, start, end, prices_dir: Path, threads: bool) -> list[str]:
    import yfinance as yf

    symbols = {yahoo_symbol(t): t for t in batch}
    data = yf.download(
        list(symbols),
        start=start,
        end=end,
        auto_adjust=False,
        actions=True,
        group_by="ticker",
        progress=False,
        threads=threads,
    )
    missing = []
    for symbol, ticker in symbols.items():
        try:
            frame = data[symbol] if isinstance(data.columns, pd.MultiIndex) else data
        except KeyError:
            frame = pd.DataFrame()
        frame = frame.dropna(subset=["Close"]) if "Close" in frame else pd.DataFrame()
        if frame.empty:
            missing.append(ticker)
            continue
        frame = frame.reindex(columns=COLUMNS)
        frame["Stock Splits"] = frame["Stock Splits"].fillna(0.0)
        frame.index.name = "Date"
        frame.to_csv(prices_dir / f"{ticker}.csv")
    return missing


def _read_price_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    date_col = next((c for c in frame.columns if c.lower() in ("date", "time", "datetime")), frame.columns[0])
    if "Close" not in frame.columns and "close" in frame.columns:
        frame = frame.rename(columns={"close": "Close"})
    if "Close" not in frame.columns and "Adj Close" not in frame.columns:
        raise ValueError(f"{path.name}: needs a 'Close' or 'Adj Close' column")
    # TradingView exports use Unix seconds in a "time" column.
    unit = "s" if pd.api.types.is_numeric_dtype(frame[date_col]) else None
    dates = pd.to_datetime(frame[date_col], utc=True, unit=unit).dt.tz_localize(None).dt.normalize()
    frame.index = pd.DatetimeIndex(dates, name="Date")
    return frame[~frame.index.duplicated(keep="last")]


def load_prices(tickers=None, start=None, end=None, prices_dir: Path = PRICES_DIR) -> dict[str, pd.DataFrame]:
    """Wide DataFrames (dates x tickers) from the CSV folder, keyed by field:

    adj_close  closes adjusted for splits and dividends (Adj Close, else Close)
    close      closes adjusted for splits only (Close, else Adj Close)
    splits     split ratio on ex-dates, 0 elsewhere
    """
    if tickers is None:
        files = sorted(prices_dir.glob("*.csv"))
    else:
        files = [prices_dir / f"{t}.csv" for t in tickers]
        files = [f for f in files if f.exists()]
    if not files:
        raise FileNotFoundError(f"No price CSV files found in {prices_dir}")

    adj_close, close, splits = {}, {}, {}
    for f in files:
        frame = _read_price_file(f)
        plain = frame["Close"] if "Close" in frame else frame["Adj Close"]
        adj_close[f.stem] = (frame["Adj Close"] if "Adj Close" in frame else plain).astype(float)
        close[f.stem] = plain.astype(float)
        if "Stock Splits" in frame:
            splits[f.stem] = frame["Stock Splits"].fillna(0.0).astype(float)

    out = {"adj_close": pd.DataFrame(adj_close), "close": pd.DataFrame(close)}
    out["splits"] = pd.DataFrame(splits).reindex(index=out["close"].index, columns=out["close"].columns).fillna(0.0)
    return {name: frame.sort_index().loc[start:end] for name, frame in out.items()}


def load_ohlc(tickers, prices_dir: Path = PRICES_DIR) -> dict[str, pd.DataFrame]:
    """Split-adjusted open/high/low/close per ticker (as on a TradingView chart).

    A missing open (e.g. a session filled from Yahoo's quote) is set to the close.
    """
    out = {}
    for ticker in tickers:
        path = prices_dir / f"{ticker}.csv"
        if not path.exists():
            continue
        frame = _read_price_file(path)
        ohlc = frame.reindex(columns=["Open", "High", "Low", "Close"]).astype(float)
        ohlc.columns = ["open", "high", "low", "close"]
        ohlc = ohlc.dropna(subset=["close"])
        ohlc["open"] = ohlc["open"].fillna(ohlc["close"])
        ohlc["high"] = ohlc["high"].fillna(ohlc[["open", "close"]].max(axis=1))
        ohlc["low"] = ohlc["low"].fillna(ohlc[["open", "close"]].min(axis=1))
        out[ticker] = ohlc
    return out


def load_closes(tickers=None, start=None, end=None, prices_dir: Path = PRICES_DIR) -> pd.DataFrame:
    """Closing prices for indicators (adjusted for splits and dividends when available)."""
    return load_prices(tickers, start, end, prices_dir)["adj_close"]
