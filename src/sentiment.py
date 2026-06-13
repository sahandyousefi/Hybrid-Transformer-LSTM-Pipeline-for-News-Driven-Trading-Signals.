"""Fine-tuning and inference for the FinBERT sentiment classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .config import SentimentConfig


class HeadlineDataset(Dataset):
    """Tokenised dataset wrapping financial headlines and integer labels."""

    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def fine_tune(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    config: SentimentConfig,
    output_dir: Path,
) -> Path:
    """Fine-tune FinBERT and persist the best checkpoint by macro F1."""
    torch.manual_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=config.num_labels
    )

    train_ds = HeadlineDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_ds = HeadlineDataset(val_texts, val_labels, tokenizer, config.max_length)

    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        seed=config.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    return best_dir


class SentimentScorer:
    """Wraps a fine-tuned checkpoint to emit signed sentiment scores."""

    def __init__(self, model_dir: Path, max_length: int = 128, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(self.device).eval()
        self.max_length = max_length

    @torch.no_grad()
    def score(self, texts, batch_size: int = 32) -> np.ndarray:
        """Return a signed score per text: P(positive) - P(negative)."""
        scores = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            enc = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            probs = torch.softmax(self.model(**enc).logits, dim=-1)
            # Columns follow the label map: 0 negative, 1 neutral, 2 positive.
            signed = probs[:, 2] - probs[:, 0]
            scores.extend(signed.cpu().numpy().tolist())
        return np.asarray(scores)
