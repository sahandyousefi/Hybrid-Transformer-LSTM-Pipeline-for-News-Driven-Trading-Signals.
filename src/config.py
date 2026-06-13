"""Central configuration objects for the FinSignal pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


@dataclass
class SentimentConfig:
    # Backbone is the domain-adapted FinBERT checkpoint.
    model_name: str = "ProsusAI/finbert"
    num_labels: int = 3
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epochs: int = 4
    seed: int = 42
    label_map: dict = field(
        default_factory=lambda: {"negative": 0, "neutral": 1, "positive": 2}
    )


@dataclass
class SignalConfig:
    # Window of trading days fed into the temporal model.
    sequence_length: int = 10
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 30
    seed: int = 42
    # Forward return horizon (days) used to build the binary target.
    horizon: int = 1


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    # Fraction of capital deployed per long signal.
    position_size: float = 1.0
    # Round-trip transaction cost in basis points.
    cost_bps: float = 5.0
