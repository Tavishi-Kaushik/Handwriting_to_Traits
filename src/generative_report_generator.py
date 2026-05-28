"""
Generate varied personality-style reports from handwriting features and Big Five scores.

This is a lightweight GenAI-style report generator that creates less repetitive
reports than the deterministic template baseline. Later, this can be replaced
with or compared against a fine-tuned transformer model.
"""

from pathlib import Path
import random

import pandas as pd


RAW_DATA_PATH = Path("data/raw/handwriting_personality_large_dataset.csv")
OUTPUT_PATH = Path("reports/generative_reports.csv")

RANDOM_STATE = 42

TRAIT_COLUMNS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]


TRAIT_MEANINGS = {
    "Openness": {
        "high": "a stronger tendency toward curiosity, imagination, and openness to new ideas",
        "medium": "a balanced level of curiosity and openness to new experiences",
        "low": "a more practical or conventional style of approaching new experiences",
    },
    "Conscientiousness": {
        "high": "a stronger tendency toward organization, planning, and careful attention to tasks",
        "medium": "a moderate level of organization and task focus",
        "low": "a more flexible or less structured approach to tasks",
    },
    "Extraversion": {
        "high": "a more socially expressive and outward-facing profile",
        "medium": "a balanced level of social engagement and independence",
        "low": "a quieter or more reserved interpersonal style",
    },
    "Agreeableness": {
        "high": "a stronger tendency toward cooperation, empathy, and interpersonal warmth",
        "medium": "a balanced approach to cooperation and personal boundaries",
        "low": "a more independent or direct interpersonal style",
    },
    "Neuroticism": {
        "high": "greater emotional sensitivity or stronger reactions to stress",
        "medium": "a moderate level of emotional responsiveness",
        "low": "a relatively calm and emotionally steady profile",
    },
}


OPENINGS = [
    "Based on the handwriting-derived features and trait scores, this sample suggests a cautious personality-style profile.",
    "The available handwriting features point toward the following dataset-based personality interpretation.",
    "Using the structured handwriting indicators and Big Five scores, the sample can be summarized as follows.",
]

CLOSINGS = [
    "This report should be read as a dataset-based interpretation, not as a clinical or psychological diagnosis.",
    "These observations are generated from model patterns and should not be treated as a definitive assessment.",
    "The interpretation is intended to be descriptive and cautious rather than diagnostic.",
]


def score_to_level(score) -> str:
    """Convert a numeric score into low, medium, or high."""
    try:
        score = float(score)
        if score >= 0.67:
            return "high"
        if score >= 0.34:
            return "medium"
        return "low"
    except (ValueError, TypeError):
        return "medium"


def generate_report(row: pd.Series) -> str:
    """Generate a varied natural-language personality-style report."""
    opening = random.choice(OPENINGS)
    closing = random.choice(CLOSINGS)

    trait_sentences = []

    for trait in TRAIT_COLUMNS:
        if trait not in row:
            continue

        level = score_to_level(row[trait])
        meaning = TRAIT_MEANINGS[trait][level]

        sentence_options = [
            f"The {trait.lower()} score is {level}, which may indicate {meaning}.",
            f"For {trait.lower()}, the profile falls in the {level} range, suggesting {meaning}.",
            f"A {level} {trait.lower()} score can be associated with {meaning}.",
        ]

        trait_sentences.append(random.choice(sentence_options))

    random.shuffle(trait_sentences)

    report = " ".join([opening] + trait_sentences + [closing])
    return report


def main() -> None:
    random.seed(RANDOM_STATE)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DATA_PATH}. "
            "Please place the Kaggle CSV file inside data/raw/."
        )

    df = pd.read_csv(RAW_DATA_PATH)

    generated_df = df.copy()
    generated_df["generated_report"] = df.apply(generate_report, axis=1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    generated_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated reports saved to: {OUTPUT_PATH}")
    print("\nSample generated report:")
    print(generated_df["generated_report"].iloc[0])


if __name__ == "__main__":
    main()