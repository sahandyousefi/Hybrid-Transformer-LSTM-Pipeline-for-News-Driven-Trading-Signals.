"""Data acquisition and preprocessing for sentiment and price series."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

URL_PATTERN = re.compile(r"http\S+|www\.\S+")
TICKER_NOISE = re.compile(r"\$[A-Za-z]{1,5}")
MULTISPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalise a headline while preserving sentiment-bearing tokens."""
    text = str(text)
    text = URL_PATTERN.sub(" ", text)
    text = TICKER_NOISE.sub(" ", text)
    text = MULTISPACE.sub(" ", text).strip()
    return text


def load_phrasebank(path: Path) -> pd.DataFrame:
    """Load the Financial PhraseBank export into a tidy frame.

    Expected columns after loading: text, label (string).
    The raw file ships as `sentence@label` lines in latin-1 encoding.
    """
    rows = []
    with open(path, "r", encoding="latin-1") as handle:
        for line in handle:
            line = line.strip()
            if not line or "@" not in line:
                continue
            sentence, label = line.rsplit("@", 1)
            rows.append({"text": clean_text(sentence), "label": label.strip()})
    frame = pd.DataFrame(rows)
    frame = frame[frame["text"].str.len() > 0].reset_index(drop=True)
    return frame


def encode_labels(frame: pd.DataFrame, label_map: dict) -> pd.DataFrame:
    frame = frame.copy()
    frame["label_id"] = frame["label"].map(label_map)
    frame = frame.dropna(subset=["label_id"]).reset_index(drop=True)
    frame["label_id"] = frame["label_id"].astype(int)
    return frame


def load_prices(path: Path) -> pd.DataFrame:
    """Load OHLCV price data saved from yfinance.

    The index is parsed as the trading date; returns are computed on close.
    """
    prices = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    prices = prices.set_index("Date")
    prices["return"] = prices["Close"].pct_change()
    prices["log_return"] = np.log1p(prices["return"])
    return prices.dropna().reset_index()


def aggregate_daily_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-headline sentiment scores into a daily signal.

    `news` must contain Date, sentiment_score (signed, in [-1, 1]) columns.
    """
    daily = (
        news.groupby(news["Date"].dt.date)
        .agg(
            mean_sentiment=("sentiment_score", "mean"),
            sentiment_std=("sentiment_score", "std"),
            news_count=("sentiment_score", "size"),
        )
        .reset_index()
    )
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)
    return daily


def build_feature_table(prices: pd.DataFrame, daily_sentiment: pd.DataFrame) -> pd.DataFrame:
    """Join price returns with daily sentiment and engineer rolling features."""
    table = prices.merge(daily_sentiment, on="Date", how="left")
    sentiment_cols = ["mean_sentiment", "sentiment_std", "news_count"]
    table[sentiment_cols] = table[sentiment_cols].fillna(0.0)

    # Momentum and volatility context for the temporal model.
    table["roll_return_5"] = table["return"].rolling(5).mean()
    table["roll_vol_5"] = table["return"].rolling(5).std()
    table["sentiment_momentum"] = table["mean_sentiment"].rolling(3).mean()
    table = table.dropna().reset_index(drop=True)
    return table
