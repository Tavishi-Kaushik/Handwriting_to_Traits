"""
Generate a personality-style report from predicted Big Five trait scores.

This script connects:
saved trait prediction models -> predicted Big Five scores -> natural-language report
"""

from pathlib import Path
import random
import joblib

import pandas as pd


DATA_PATH = Path("data/raw/handwriting_personality_large_dataset.csv")
MODEL_DIR = Path("models")

TRAIT_COLUMNS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]

RANDOM_STATE = 42


TRAIT_MEANINGS = {
    "Openness": {
        "high": "curiosity, imagination, and openness to new ideas",
        "medium": "a balanced level of curiosity and practical thinking",
        "low": "a more conventional or practical approach to new experiences",
    },
    "Conscientiousness": {
        "high": "organization, planning, and careful attention to tasks",
        "medium": "moderate organization and task focus",
        "low": "a more flexible or less structured approach to tasks",
    },
    "Extraversion": {
        "high": "social energy, expressiveness, and outward engagement",
        "medium": "a balanced level of social engagement and independence",
        "low": "a quieter or more reserved interpersonal style",
    },
    "Agreeableness": {
        "high": "cooperation, empathy, and interpersonal warmth",
        "medium": "a balanced approach to cooperation and personal boundaries",
        "low": "a more independent or direct interpersonal style",
    },
    "Neuroticism": {
        "high": "greater emotional sensitivity or stronger stress responses",
        "medium": "moderate emotional responsiveness",
        "low": "a relatively calm and emotionally steady profile",
    },
}


def score_to_level(score: float) -> str:
    """Convert a score between 0 and 1 into low, medium, or high."""
    if score >= 0.67:
        return "high"
    if score >= 0.34:
        return "medium"
    return "low"


def predict_traits(sample: pd.DataFrame) -> dict:
    """Load saved models and predict Big Five trait scores."""
    predictions = {}

    for trait in TRAIT_COLUMNS:
        model_path = MODEL_DIR / f"{trait.lower()}_model.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please run src/train_and_save_model.py first."
            )

        model = joblib.load(model_path)
        predicted_score = model.predict(sample)[0]
        predictions[trait] = round(float(predicted_score), 4)

    return predictions


def generate_report(predictions: dict) -> str:
    """Generate a cautious personality-style report from predicted scores."""
    openings = [
        "Based on the handwriting-derived features, the model predicts the following personality-style profile.",
        "Using the available handwriting features, the system generated this cautious Big Five interpretation.",
        "The predicted trait scores suggest the following dataset-based personality profile.",
    ]

    closings = [
        "This report is based on model patterns and should not be treated as a psychological diagnosis.",
        "These results are descriptive and should be interpreted cautiously.",
        "This is a dataset-based interpretation, not a clinical assessment.",
    ]

    report_parts = [random.choice(openings)]

    for trait, score in predictions.items():
        level = score_to_level(score)
        meaning = TRAIT_MEANINGS[trait][level]

        report_parts.append(
            f"{trait} is predicted to be {level} with a score of {score}, "
            f"which may relate to {meaning}."
        )

    report_parts.append(random.choice(closings))

    return " ".join(report_parts)


def main() -> None:
    random.seed(RANDOM_STATE)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Please place the Kaggle CSV file inside data/raw/."
        )

    df = pd.read_csv(DATA_PATH)

    feature_columns = [col for col in df.columns if col not in TRAIT_COLUMNS]
    sample = df[feature_columns].iloc[[0]].fillna("missing")

    predictions = predict_traits(sample)
    report = generate_report(predictions)

    print("Predicted trait scores:")
    for trait, score in predictions.items():
        print(f"{trait}: {score}")

    print("\nGenerated report:")
    print(report)


if __name__ == "__main__":
    main()