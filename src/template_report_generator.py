"""
Template-based baseline report generator.

This script creates simple deterministic personality-style reports from
Big Five trait scores. This will be used as a baseline to compare against
a generative model later.
"""

from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/handwriting_personality_large_dataset.csv")
OUTPUT_DIR = Path("reports")
OUTPUT_PATH = OUTPUT_DIR / "template_reports.csv"


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
    except ValueError:
        return str(score).lower()


def generate_template_report(row: pd.Series) -> str:
    """Generate a deterministic report from Big Five trait values."""
    report_parts = []

    report_parts.append(
        "This handwriting sample is associated with the following personality-style profile. "
        "The interpretation is based on dataset patterns and should not be treated as a psychological diagnosis."
    )

    for trait in TRAIT_COLUMNS:
        if trait in row:
            level = describe_score(row[trait])
            description = TRAIT_DESCRIPTIONS[trait]
            report_parts.append(
                f"The {trait.lower()} score appears to be {level}, which may relate to {description}."
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

    reports_df = df.copy()
    reports_df["template_report"] = df.apply(generate_template_report, axis=1)

    reports_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Template reports saved to: {OUTPUT_PATH}")
    print("\nSample report:")
    print(reports_df["template_report"].iloc[0])


if __name__ == "__main__":
    main()