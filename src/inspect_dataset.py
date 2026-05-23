"""
Inspect the Handwriting & Personality Traits dataset.

This script checks:
- dataset shape
- column names
- missing values
- possible image-related columns
- personality trait columns
"""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/handwriting_personality_large_dataset.csv")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Please place the CSV file inside data/raw/."
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}")

    print("\nMissing values:")
    print(df.isnull().sum())

    possible_image_columns = [
        col for col in df.columns
        if any(keyword in col.lower() for keyword in ["image", "img", "path", "file", "sample"])
    ]

    print("\nPossible image/sample-related columns:")
    if possible_image_columns:
        for col in possible_image_columns:
            print(f"- {col}")
            print(df[col].head())
    else:
        print("No obvious image/path/sample columns found.")

    trait_keywords = [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    ]

    possible_trait_columns = [
        col for col in df.columns
        if any(keyword in col.lower() for keyword in trait_keywords)
    ]

    print("\nPossible Big Five trait columns:")
    if possible_trait_columns:
        for col in possible_trait_columns:
            print(f"- {col}")
    else:
        print("No obvious Big Five trait columns found.")

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()