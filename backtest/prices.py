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
    return missing


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


def load_closes(tickers=None, start=None, end=None, prices_dir: Path = PRICES_DIR) -> pd.DataFrame:
    """Closing prices for indicators (adjusted for splits and dividends when available)."""
    return load_prices(tickers, start, end, prices_dir)["adj_close"]
