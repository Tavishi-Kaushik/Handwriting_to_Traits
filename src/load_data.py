"""
Load and inspect the Handwriting & Personality Traits dataset.

This script is used for basic dataset checks:
- file exists
- shape
- column names
- missing values
- sample rows
"""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/handwriting_personality_large_dataset.csv")


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the handwriting personality dataset from a CSV file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Please place the Kaggle CSV file inside data/raw/."
        )

    return pd.read_csv(path)


def main() -> None:
    """Run basic dataset inspection."""
    df = load_dataset()

    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()