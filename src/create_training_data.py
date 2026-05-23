"""
Create training data for text generation.

This script creates input-output pairs:
- input_text: structured handwriting features and trait scores
- target_text: personality-style report

For now, the target report uses the template baseline. Later, this can be
replaced or expanded with more varied generated reports.
"""

from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/handwriting_personality_large_dataset.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_PATH = OUTPUT_DIR / "training_pairs.csv"


TRAIT_COLUMNS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]


TRAIT_DESCRIPTIONS = {
    "Openness": "curiosity, imagination, and openness to new experiences",
    "Conscientiousness": "organization, responsibility, and attention to detail",
    "Extraversion": "social energy, expressiveness, and outward engagement",
    "Agreeableness": "cooperation, empathy, and interpersonal warmth",
    "Neuroticism": "emotional sensitivity and tendency toward stress responses",
}


def describe_score(score) -> str:
    """Convert a numeric or text score into a simple level."""
    try:
        score = float(score)
        if score >= 0.67:
            return "high"
        if score >= 0.34:
            return "medium"
        return "low"
    except (ValueError, TypeError):
        return str(score).lower()


def create_input_text(row: pd.Series) -> str:
    """Create the model input text from one row."""
    handwriting_parts = []
    trait_parts = []

    for col in row.index:
        value = row[col]

        if col in TRAIT_COLUMNS:
            trait_parts.append(f"{col}: {value}")
        else:
            handwriting_parts.append(f"{col}: {value}")

    input_text = (
        "Generate a cautious personality-style report based on the following handwriting features "
        "and Big Five trait scores.\n\n"
        "Handwriting features:\n"
        + "\n".join(handwriting_parts)
        + "\n\nBig Five trait scores:\n"
        + "\n".join(trait_parts)
    )

    return input_text


def create_target_text(row: pd.Series) -> str:
    """Create the target personality-style report."""
    report_parts = []

    report_parts.append(
        "This handwriting sample is associated with a personality-style profile based on dataset patterns. "
        "This should not be treated as a psychological diagnosis."
    )

    for trait in TRAIT_COLUMNS:
        if trait in row:
            level = describe_score(row[trait])
            description = TRAIT_DESCRIPTIONS[trait]
            report_parts.append(
                f"The {trait.lower()} score appears {level}, suggesting possible links to {description}."
            )

    return " ".join(report_parts)


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DATA_PATH}. "
            "Please place the Kaggle CSV file inside data/raw/."
        )

    df = pd.read_csv(RAW_DATA_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_df = pd.DataFrame()
    training_df["input_text"] = df.apply(create_input_text, axis=1)
    training_df["target_text"] = df.apply(create_target_text, axis=1)

    training_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Training pairs saved to: {OUTPUT_PATH}")
    print(f"Number of examples: {len(training_df)}")

    print("\nSample input:")
    print(training_df["input_text"].iloc[0])

    print("\nSample target:")
    print(training_df["target_text"].iloc[0])


if __name__ == "__main__":
    main()