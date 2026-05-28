"""
Compare baseline models for predicting Big Five personality traits.

This script compares:
- Mean baseline
- Linear regression
- Random forest
- Gradient boosting

The goal is to check whether handwriting features contain enough signal
to predict Big Five personality trait scores.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/raw/handwriting_personality_large_dataset.csv")

TRAIT_COLUMNS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]

RANDOM_STATE = 42


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing for numeric and categorical features."""
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

    models = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
        ),
    }

    results = []

    for trait in available_traits:
        y = df[trait]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
        )

        preprocessor = build_preprocessor(X_train)

        for model_name, model in models.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            results.append(
                {
                    "trait": trait,
                    "model": model_name,
                    "mae": mae,
                    "r2": r2,
                }
            )

    results_df = pd.DataFrame(results)

    print("\nModel comparison results:")
    print(results_df.sort_values(["trait", "mae"]))

    output_path = Path("reports/trait_model_comparison.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()