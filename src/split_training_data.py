"""
Split text generation training pairs into train, validation, and test sets.

Input:
- data/processed/training_pairs.csv

Outputs:
- data/processed/train.csv
- data/processed/val.csv
- data/processed/test.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = Path("data/processed/training_pairs.csv")
OUTPUT_DIR = Path("data/processed")

TRAIN_PATH = OUTPUT_DIR / "train.csv"
VAL_PATH = OUTPUT_DIR / "val.csv"
TEST_PATH = OUTPUT_DIR / "test.csv"

RANDOM_STATE = 42


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Training pairs not found at {INPUT_PATH}. "
            "Please run src/create_training_data.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("Dataset split completed.")
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    print(f"\nSaved train set to: {TRAIN_PATH}")
    print(f"Saved validation set to: {VAL_PATH}")
    print(f"Saved test set to: {TEST_PATH}")


if __name__ == "__main__":
    main()