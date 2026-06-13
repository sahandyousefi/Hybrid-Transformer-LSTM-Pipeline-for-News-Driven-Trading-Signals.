"""Temporal signal model: an LSTM over sentiment-augmented price features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .config import SignalConfig

FEATURE_COLUMNS = [
    "log_return",
    "roll_return_5",
    "roll_vol_5",
    "mean_sentiment",
    "sentiment_std",
    "news_count",
    "sentiment_momentum",
]


def make_sequences(frame, feature_cols, seq_len: int, horizon: int):
    """Build sliding windows and a binary up/down target `horizon` days ahead."""
    features = frame[feature_cols].to_numpy(dtype=np.float32)
    forward_return = frame["return"].shift(-horizon).to_numpy(dtype=np.float32)

    sequences, targets = [], []
    for i in range(len(frame) - seq_len - horizon):
        sequences.append(features[i : i + seq_len])
        targets.append(1.0 if forward_return[i + seq_len - 1] > 0 else 0.0)
    return np.asarray(sequences), np.asarray(targets, dtype=np.float32)


def chronological_split(x, y, val_fraction: float = 0.2):
    """Split without shuffling to respect the temporal ordering."""
    cut = int(len(x) * (1 - val_fraction))
    return x[:cut], x[cut:], y[:cut], y[cut:]


def scale_sequences(train_x, val_x):
    """Fit a standard scaler on training windows only to avoid leakage."""
    n_features = train_x.shape[2]
    scaler = StandardScaler()
    scaler.fit(train_x.reshape(-1, n_features))

    def apply(arr):
        flat = arr.reshape(-1, n_features)
        return scaler.transform(flat).reshape(arr.shape).astype(np.float32)

    return apply(train_x), apply(val_x), scaler


class SentimentLSTM(nn.Module):
    """Two-layer LSTM with a sigmoid head for directional prediction."""

    def __init__(self, n_features: int, config: SignalConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1),
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]
        return self.head(last_step).squeeze(-1)


def train_signal_model(train_x, train_y, val_x, val_y, config: SignalConfig, device: str):
    torch.manual_seed(config.seed)
    model = SentimentLSTM(train_x.shape[2], config).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    val_x_t = torch.from_numpy(val_x).to(device)
    val_y_t = torch.from_numpy(val_y).to(device)

    best_state, best_acc = None, 0.0
    for epoch in range(config.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(val_x_t))
            val_acc = ((val_prob > 0.5).float() == val_y_t).float().mean().item()
        if val_acc >= best_acc:
            best_acc, best_state = val_acc, {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc


@torch.no_grad()
def predict_proba(model, x, device: str) -> np.ndarray:
    model.eval()
    logits = model(torch.from_numpy(x).to(device))
    return torch.sigmoid(logits).cpu().numpy()


def save_model(model, path: Path) -> None:
    torch.save(model.state_dict(), path)
