"""Unit tests covering preprocessing, sequence building, and the backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import generate_positions, performance_summary, run_backtest
from src.config import BacktestConfig, SignalConfig
from src.data import aggregate_daily_sentiment, build_feature_table, clean_text
from src.signal_model import FEATURE_COLUMNS, make_sequences


def test_clean_text_removes_urls_and_tickers():
    raw = "Apple beats on earnings http://x.co $AAPL surges"
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "$AAPL" not in cleaned
    assert "earnings" in cleaned


def test_aggregate_daily_sentiment_counts():
    news = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2022-01-03", "2022-01-03", "2022-01-04"]),
            "sentiment_score": [0.5, -0.1, 0.2],
        }
    )
    daily = aggregate_daily_sentiment(news)
    assert len(daily) == 2
    assert daily.loc[daily["Date"] == "2022-01-03", "news_count"].iloc[0] == 2


def _toy_table(n: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    prices = pd.DataFrame(
        {
            "Date": dates,
            "Close": np.linspace(100, 120, n),
            "return": np.random.normal(0, 0.01, n),
            "log_return": np.random.normal(0, 0.01, n),
        }
    )
    sentiment = pd.DataFrame(
        {
            "Date": dates,
            "mean_sentiment": np.random.uniform(-1, 1, n),
            "sentiment_std": np.random.uniform(0, 0.5, n),
            "news_count": np.random.randint(0, 5, n),
        }
    )
    return prices.merge(sentiment, on="Date")


def test_build_feature_table_has_expected_columns():
    prices = _toy_table()
    sentiment = prices[["Date", "mean_sentiment", "sentiment_std", "news_count"]]
    table = build_feature_table(
        prices.drop(columns=["mean_sentiment", "sentiment_std", "news_count"]),
        sentiment,
    )
    for col in ["roll_return_5", "roll_vol_5", "sentiment_momentum"]:
        assert col in table.columns


def test_make_sequences_shapes():
    table = _toy_table()
    table["roll_return_5"] = table["return"].rolling(5).mean()
    table["roll_vol_5"] = table["return"].rolling(5).std()
    table["sentiment_momentum"] = table["mean_sentiment"].rolling(3).mean()
    table = table.dropna().reset_index(drop=True)

    cfg = SignalConfig(sequence_length=10, horizon=1)
    x, y = make_sequences(table, FEATURE_COLUMNS, cfg.sequence_length, cfg.horizon)
    assert x.shape[1] == cfg.sequence_length
    assert x.shape[2] == len(FEATURE_COLUMNS)
    assert len(x) == len(y)


def test_backtest_charges_costs_on_position_change():
    returns = np.array([0.01, 0.02, -0.01, 0.0])
    positions = generate_positions(np.array([0.9, 0.9, 0.1, 0.9]))
    result = run_backtest(returns, positions, BacktestConfig(cost_bps=10.0))
    assert len(result) == len(returns)
    summary = performance_summary(result)
    assert "strategy_sharpe" in summary
