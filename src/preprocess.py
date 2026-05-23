"""
Preprocess the handwriting personality dataset into prompt-style examples.

This script creates structured text inputs that can later be used for
personality report generation.
"""

from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/handwriting_personality_large_dataset.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "prompt_examples.csv"


TRAIT_COLUMNS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]


def create_prompt(row: pd.Series) -> str:
    """Create a structured prompt from one dataset row."""
    handwriting_parts = []
    trait_parts = []

    for col in row.index:
        value = row[col]

        if col in TRAIT_COLUMNS:
            trait_parts.append(f"{col}: {value}")
        else:
            handwriting_parts.append(f"{col}: {value}")

    prompt = (
        "Generate a cautious personality-style report from the following handwriting information.\n\n"
        "Handwriting features:\n"
        + "\n".join(handwriting_parts)
        + "\n\nPersonality trait scores:\n"
        + "\n".join(trait_parts)
    )

    return prompt


def main() -> None:
    """Load raw data and create prompt examples."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DATA_PATH}. "
            "Please place the Kaggle CSV file inside data/raw/."
        )

    df = pd.read_csv(RAW_DATA_PATH)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    processed_df = pd.DataFrame()
    processed_df["prompt"] = df.apply(create_prompt, axis=1)

    processed_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Created prompt examples: {OUTPUT_PATH}")
    print(f"Number of examples: {len(processed_df)}")
    print("\nSample prompt:")
    print(processed_df["prompt"].iloc[0])


if __name__ == "__main__":
    main()