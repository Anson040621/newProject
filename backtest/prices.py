"""Daily price data: download from Yahoo Finance or load from CSV files.

Prices are stored as one CSV per ticker in data/prices/<TICKER>.csv with at
least a Date column and a Close (or Adj Close) column. You can drop your own
CSV exports (TradingView, broker, etc.) into that folder instead of downloading.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .universe import DATA_DIR

PRICES_DIR = DATA_DIR / "prices"


def yahoo_symbol(ticker: str) -> str:
    """S&P list uses BRK.B style; Yahoo uses BRK-B."""
    return ticker.replace(".", "-")


def download_prices(tickers, start, end=None, prices_dir: Path = PRICES_DIR, batch_size: int = 50) -> list[str]:
    """Download split/dividend-adjusted daily OHLCV from Yahoo. Returns tickers with no data."""
    import yfinance as yf

    prices_dir.mkdir(parents=True, exist_ok=True)
    tickers = list(tickers)
    missing = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        symbols = {yahoo_symbol(t): t for t in batch}
        data = yf.download(
            list(symbols),
            start=start,
            end=end,
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
        for symbol, ticker in symbols.items():
            try:
                frame = data[symbol] if isinstance(data.columns, pd.MultiIndex) else data
            except KeyError:
                frame = pd.DataFrame()
            frame = frame.dropna(subset=["Close"]) if "Close" in frame else pd.DataFrame()
            if frame.empty:
                missing.append(ticker)
                continue
            frame.index.name = "Date"
            frame[["Open", "High", "Low", "Close", "Volume"]].to_csv(prices_dir / f"{ticker}.csv")
        print(f"  downloaded {min(i + batch_size, len(tickers))}/{len(tickers)}")
    return missing


def load_closes(tickers=None, start=None, end=None, prices_dir: Path = PRICES_DIR) -> pd.DataFrame:
    """Wide DataFrame of closing prices (dates x tickers) from the CSV folder.

    Uses "Adj Close" when a file has it, otherwise "Close".
    """
    if tickers is None:
        files = sorted(prices_dir.glob("*.csv"))
    else:
        files = [prices_dir / f"{t}.csv" for t in tickers]
        files = [f for f in files if f.exists()]
    if not files:
        raise FileNotFoundError(f"No price CSV files found in {prices_dir}")

    series = {}
    for f in files:
        frame = pd.read_csv(f)
        date_col = next((c for c in frame.columns if c.lower() in ("date", "time", "datetime")), frame.columns[0])
        price_col = "Adj Close" if "Adj Close" in frame.columns else "Close"
        if price_col not in frame.columns:
            raise ValueError(f"{f.name}: needs a 'Close' or 'Adj Close' column")
        # TradingView exports use Unix seconds in a "time" column.
        unit = "s" if pd.api.types.is_numeric_dtype(frame[date_col]) else None
        dates = pd.to_datetime(frame[date_col], utc=True, unit=unit).dt.tz_localize(None).dt.normalize()
        series[f.stem] = pd.Series(frame[price_col].to_numpy(dtype=float), index=dates).groupby(level=0).last()

    closes = pd.DataFrame(series).sort_index()
    return closes.loc[start:end]
