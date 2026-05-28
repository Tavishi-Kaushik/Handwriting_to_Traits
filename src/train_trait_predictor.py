"""
Train baseline models to predict Big Five personality traits from handwriting features.

This script trains a simple Random Forest model for each Big Five trait.
It is a baseline model, not the final GenAI report generator.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


DATA_PATH = Path("data/raw/handwriting_personality_large_dataset.csv")

TRAIT_COLUMNS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]

RANDOM_STATE = 42


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Please place the Kaggle CSV file inside data/raw/."
        )

    df = pd.read_csv(DATA_PATH)

    available_traits = [col for col in TRAIT_COLUMNS if col in df.columns]

    if not available_traits:
        raise ValueError(
            "No Big Five trait columns found. Please check the dataset column names."
        )

    feature_columns = [col for col in df.columns if col not in available_traits]

    X = df[feature_columns]
    X = X.fillna("missing")

    print("Feature columns:")
    for col in feature_columns:
        print(f"- {col}")

    print("\nTrait prediction results:")

    categorical_features = X.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    for trait in available_traits:
        y = df[trait]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
        )

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(
                    n_estimators=100,
                    random_state=RANDOM_STATE,
                )),
            ]
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"\n{trait}")
        print(f"MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")


if __name__ == "__main__":
    main()