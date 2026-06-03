"""
Load saved trait prediction models and predict Big Five scores for one sample.

This script demonstrates model reuse after training.
It supports the ML Operations part of the project.
"""

from pathlib import Path
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


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Please place the Kaggle CSV file inside data/raw/."
        )

    df = pd.read_csv(DATA_PATH)

    available_traits = [col for col in TRAIT_COLUMNS if col in df.columns]
    feature_columns = [col for col in df.columns if col not in available_traits]

    sample = df[feature_columns].iloc[[0]].fillna("missing")

    predictions = {}

    for trait in available_traits:
        model_path = MODEL_DIR / f"{trait.lower()}_model.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please run src/train_and_save_model.py first."
            )

        model = joblib.load(model_path)
        predicted_score = model.predict(sample)[0]

        predictions[trait] = round(float(predicted_score), 4)

    print("Predicted Big Five trait scores for sample row 0:")
    for trait, score in predictions.items():
        print(f"{trait}: {score}")


if __name__ == "__main__":
    main()