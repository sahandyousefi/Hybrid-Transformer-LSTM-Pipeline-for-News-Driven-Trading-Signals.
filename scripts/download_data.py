"""Download price history and a sample of financial headlines.

Prices come from Yahoo Finance via yfinance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yfinance as yf

from src.config import DATA_DIR


def download_prices(ticker: str, start: str, end: str, out_dir: Path) -> Path:
    frame = yf.download(ticker, start=start, end=end, auto_adjust=True)
    frame = frame.reset_index()
    # Flatten multi-index columns returned for single tickers.
    frame.columns = [c[0] if isinstance(c, tuple) else c for c in frame.columns]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker}_prices.csv"
    frame.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download price history.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2023-12-31")
    args = parser.parse_args()

    path = download_prices(args.ticker, args.start, args.end, DATA_DIR)
    print(f"Saved prices to {path}")


if __name__ == "__main__":
    main()
