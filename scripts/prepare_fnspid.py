"""Adapt the FNSPID news export to the schema the pipeline expects.

The raw FNSPID news file (nasdaq_exteral_data.csv) is ~23 GB, so it is read
in chunks and filtered to a single ticker. The output is a compact CSV with
the `Date` and `headline` columns consumed by run_pipeline.py.

The FNSPID news columns vary slightly across releases; this script detects
the date, headline, and symbol columns by common candidate names.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR

DATE_CANDIDATES = ["Date", "date", "publish_date", "Datetime"]
HEADLINE_CANDIDATES = ["Article_title", "title", "headline", "Headline"]
SYMBOL_CANDIDATES = ["Stock_symbol", "symbol", "Symbol", "ticker"]


def _pick(columns, candidates, role: str) -> str:
    for name in candidates:
        if name in columns:
            return name
    raise ValueError(
        f"Could not find a {role} column. Available columns: {list(columns)}"
    )


def adapt(raw_path: Path, ticker: str, out_path: Path, chunksize: int = 250_000) -> Path:
    collected = []
    columns = None
    for chunk in pd.read_csv(raw_path, chunksize=chunksize, low_memory=False):
        if columns is None:
            columns = chunk.columns
            date_col = _pick(columns, DATE_CANDIDATES, "date")
            head_col = _pick(columns, HEADLINE_CANDIDATES, "headline")
            sym_col = _pick(columns, SYMBOL_CANDIDATES, "symbol")

        subset = chunk[chunk[sym_col].astype(str).str.upper() == ticker.upper()]
        if not subset.empty:
            collected.append(subset[[date_col, head_col]].copy())

    if not collected:
        raise ValueError(f"No news rows found for ticker {ticker}.")

    news = pd.concat(collected, ignore_index=True)
    news.columns = ["Date", "headline"]
    news["Date"] = pd.to_datetime(news["Date"], errors="coerce").dt.normalize()
    news = news.dropna(subset=["Date", "headline"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    news.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter FNSPID news to one ticker.")
    parser.add_argument("--raw", type=Path, required=True, help="Path to nasdaq_exteral_data.csv")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or (DATA_DIR / f"{args.ticker}_news.csv")
    path = adapt(args.raw, args.ticker, out)
    print(f"Saved filtered news to {path}")


if __name__ == "__main__":
    main()
