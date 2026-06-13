"""Train the FinBERT sentiment classifier on the Financial PhraseBank."""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.config import ARTIFACT_DIR, DATA_DIR, SentimentConfig
from src.data import encode_labels, load_phrasebank
from src.sentiment import fine_tune


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT.")
    parser.add_argument(
        "--phrasebank",
        type=Path,
        default=DATA_DIR / "Sentences_AllAgree.txt",
        help="Path to a Financial PhraseBank text export.",
    )
    args = parser.parse_args()

    config = SentimentConfig()
    frame = load_phrasebank(args.phrasebank)
    frame = encode_labels(frame, config.label_map)

    train_df, val_df = train_test_split(
        frame,
        test_size=0.2,
        stratify=frame["label_id"],
        random_state=config.seed,
    )

    output_dir = ARTIFACT_DIR / "sentiment"
    best_dir = fine_tune(
        train_df["text"].tolist(),
        train_df["label_id"].tolist(),
        val_df["text"].tolist(),
        val_df["label_id"].tolist(),
        config,
        output_dir,
    )
    print(f"Best sentiment checkpoint saved to {best_dir}")


if __name__ == "__main__":
    main()
