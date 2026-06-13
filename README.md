# FinSignal

A hybrid deep learning pipeline that converts financial news sentiment into a directional market signal and evaluates that signal through a transaction-cost-aware backtest.

The system has two stages. First, a transformer (FinBERT) is fine-tuned on labelled financial text to produce a signed sentiment score per headline. Second, those scores are aggregated daily, joined with price-derived features, and fed into an LSTM that predicts the direction of the next trading day. The resulting probabilities drive a simple long/flat strategy that is benchmarked against buy-and-hold.

## Motivation

Most published sentiment work stops at the classification metric. In a trading context, a high F1 score is necessary but not sufficient: a model can classify headlines well and still produce a strategy that loses money once costs and turnover are accounted for. This project deliberately carries the analysis through to a backtest so that model quality is judged on a metric an investment team would actually care about, namely risk-adjusted return relative to a passive benchmark.

## Architecture

```
Headlines ──▶ FinBERT (fine-tuned) ──▶ signed sentiment score
                                              │
Prices ──▶ returns, rolling vol, momentum ────┤
                                              ▼
                                   daily feature table
                                              │
                                  sliding windows (length 10)
                                              ▼
                                   LSTM ──▶ P(up next day)
                                              │
                                   threshold ▶ long / flat
                                              ▼
                              cost-aware backtest vs benchmark
```

The two models are trained independently. The sentiment model is a fine-tuned `ProsusAI/finbert` checkpoint; the signal model is a two-layer LSTM with a small feed-forward head. The split keeps each component interpretable and lets the sentiment encoder be reused across tickers without retraining.

## Project structure

```
finsignal/
├── src/
│   ├── config.py          Dataclass configuration for each stage
│   ├── data.py            Cleaning, daily aggregation, feature engineering
│   ├── sentiment.py       FinBERT fine-tuning and inference scorer
│   ├── signal_model.py    Sequence construction and the LSTM
│   └── backtest.py        Strategy simulation and performance metrics
├── scripts/
│   ├── download_data.py   Pull price history from Yahoo Finance
│   ├── train_sentiment.py Fine-tune FinBERT on the Financial PhraseBank
│   └── run_pipeline.py    Score news, train the LSTM, run the backtest
├── tests/                 Unit tests for preprocessing and the backtest
├── requirements.txt
└── README.md
```

## Datasets

The project relies on three public, freely available sources.

| Source | What it provides | Where to get it |
|--------|------------------|-----------------|
| Financial PhraseBank | Roughly 4,800 financial sentences labelled negative / neutral / positive, used to fine-tune the sentiment model | Available on the Hugging Face Hub as `financial_phrasebank`, and on Kaggle. The `Sentences_AllAgree.txt` subset (full annotator agreement) is the recommended starting point. |
| FiQA 2018 sentiment | Additional labelled financial sentences for optional augmentation | Hugging Face Hub: `pauri32/fiqa-2018` |
| Price history | Daily OHLCV bars used to compute returns and the prediction target | Yahoo Finance via the `yfinance` package; `scripts/download_data.py` handles the download |

For the news stream that is scored at inference time, any dated headline source works as long as it is saved as a CSV with `Date` and `headline` columns. For a fully reproducible large-scale run, the FNSPID dataset (`Zihan1004/FNSPID` on the Hugging Face Hub) provides millions of time-aligned news items and prices.

The Financial PhraseBank should be cited as Malo et al. (2014), "Good debt or bad debt: Detecting semantic orientations in economic texts."

## Setup

```bash
git clone https://github.com/sahandyousefi/finsignal.git
cd finsignal
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A GPU is recommended for the fine-tuning stage but not required; the LSTM trains comfortably on CPU.

## Usage

Download price history for a ticker:

```bash
python scripts/download_data.py --ticker AAPL --start 2018-01-01 --end 2023-12-31
```

Place the Financial PhraseBank export at `data/Sentences_AllAgree.txt`, then fine-tune the sentiment model:

```bash
python scripts/train_sentiment.py --phrasebank data/Sentences_AllAgree.txt
```

Place a news CSV at `data/AAPL_news.csv` (columns `Date`, `headline`), then run the full pipeline:

```bash
python scripts/run_pipeline.py --prices data/AAPL_prices.csv --news data/AAPL_news.csv
```

The pipeline writes the trained LSTM, the equity curve, and a metrics summary to `artifacts/`.

## Evaluation

The sentiment stage is evaluated with accuracy and macro F1 on a held-out split. The signal stage is evaluated on a strictly chronological hold-out to avoid look-ahead bias, and reported as directional accuracy plus the following strategy metrics:

- Total return, strategy versus benchmark
- Annualised Sharpe ratio, strategy versus benchmark
- Maximum drawdown
- Hit rate on days the strategy holds a position

The scaler is fit only on the training window, and the train/validation split preserves time order, so neither the features nor the target leak future information.

## Design notes and limitations

The strategy is intentionally simple (long or flat, single asset) so that the contribution of the sentiment signal is easy to isolate. It is a research artifact, not investment advice. Real deployment would require survivorship-bias-free data, slippage modelling, position sizing, and out-of-sample testing across multiple market regimes. The daily sentiment aggregation also assumes headlines are correctly timestamped to the session they affect; intraday timing and look-ahead at the news level are out of scope here.

## Testing

```bash
pytest -q
```

The tests cover text cleaning, daily aggregation, feature construction, sequence windowing, and the backtest accounting including transaction costs.

## License

Released under the MIT License. See `LICENSE` for details.
