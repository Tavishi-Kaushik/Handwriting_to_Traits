"""
Check dataset details for the handwriting personality dataset.

This script helps verify:
- column data types
- sample values
- whether Handwriting_Sample looks like an image path or just an ID/text field
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

    print("Dataset shape:")
    print(df.shape)

    print("\nColumn data types:")
    print(df.dtypes)

    print("\nFirst 10 rows:")
    print(df.head(10))

    print("\nSample values from each column:")
    for col in df.columns:
        print(f"\n--- {col} ---")
        print(df[col].dropna().head(10).tolist())

    if "Handwriting_Sample" in df.columns:
        print("\nUnique Handwriting_Sample examples:")
        print(df["Handwriting_Sample"].dropna().unique()[:20])

        image_like_values = df["Handwriting_Sample"].astype(str).str.contains(
            ".jpg|.jpeg|.png|.bmp|.tiff|.webp|/",
            case=False,
            regex=True,
        )

        print("\nDoes Handwriting_Sample look like image paths?")
        print(image_like_values.value_counts())


if __name__ == "__main__":
    main()