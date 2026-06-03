"""
Train and save baseline trait prediction models.

This script trains one Random Forest model per Big Five personality trait
and saves the trained models as reusable artifacts.

This supports the ML Operations extra criterion.
"""

from pathlib import Path
import joblib

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing for numeric and categorical columns."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Please place the Kaggle CSV file inside data/raw/."
        )

    df = pd.read_csv(DATA_PATH)

    available_traits = [col for col in TRAIT_COLUMNS if col in df.columns]
    feature_columns = [col for col in df.columns if col not in available_traits]

    X = df[feature_columns].fillna("missing")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for trait in available_traits:
        y = df[trait]

        X_train, _, y_train, _ = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
        )

        model = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=200,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)

        model_path = MODEL_DIR / f"{trait.lower()}_model.joblib"
        joblib.dump(model, model_path)

        print(f"Saved {trait} model to {model_path}")


if __name__ == "__main__":
    main()