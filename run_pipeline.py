"""End-to-end pipeline: score news, train the LSTM, run the backtest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.backtest import generate_positions, performance_summary, run_backtest
from src.config import (
    ARTIFACT_DIR,
    DATA_DIR,
    BacktestConfig,
    SignalConfig,
)
from src.data import (
    aggregate_daily_sentiment,
    build_feature_table,
    clean_text,
    load_prices,
)
from src.sentiment import SentimentScorer
from src.signal_model import (
    FEATURE_COLUMNS,
    chronological_split,
    make_sequences,
    predict_proba,
    save_model,
    scale_sequences,
    train_signal_model,
)


def score_news(news_path: Path, model_dir: Path) -> pd.DataFrame:
    news = pd.read_csv(news_path, parse_dates=["Date"])
    news["text"] = news["headline"].map(clean_text)
    scorer = SentimentScorer(model_dir)
    news["sentiment_score"] = scorer.score(news["text"].tolist())
    return news


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FinSignal pipeline.")
    parser.add_argument("--prices", type=Path, default=DATA_DIR / "AAPL_prices.csv")
    parser.add_argument("--news", type=Path, default=DATA_DIR / "AAPL_news.csv")
    parser.add_argument(
        "--sentiment-model",
        type=Path,
        default=ARTIFACT_DIR / "sentiment" / "best",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    signal_cfg = SignalConfig()
    backtest_cfg = BacktestConfig()

    prices = load_prices(args.prices)
    news = score_news(args.news, args.sentiment_model)
    daily_sentiment = aggregate_daily_sentiment(news)
    table = build_feature_table(prices, daily_sentiment)

    x, y = make_sequences(
        table, FEATURE_COLUMNS, signal_cfg.sequence_length, signal_cfg.horizon
    )
    train_x, val_x, train_y, val_y = chronological_split(x, y)
    train_x, val_x, _ = scale_sequences(train_x, val_x)

    model, val_acc = train_signal_model(
        train_x, train_y, val_x, val_y, signal_cfg, device
    )
    print(f"Validation directional accuracy: {val_acc:.4f}")

    # Align validation forward returns with the predicted positions.
    val_offset = len(train_x) + signal_cfg.sequence_length
    val_returns = (
        table["return"].shift(-signal_cfg.horizon).to_numpy()[
            val_offset : val_offset + len(val_x)
        ]
    )
    probabilities = predict_proba(model, val_x, device)
    positions = generate_positions(probabilities)
    result = run_backtest(val_returns, positions, backtest_cfg)
    summary = performance_summary(result)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model, ARTIFACT_DIR / "signal_lstm.pt")
    result.to_csv(ARTIFACT_DIR / "backtest_curve.csv", index=False)
    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump({"validation_accuracy": val_acc, **summary}, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
