"""
Compute feature importances for Big Five trait prediction.

This script trains a Random Forest model for each personality trait and
prints the most important input features. This supports the Explainable AI
part of the project.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("data/raw/handwriting_personality_large_dataset.csv")
OUTPUT_PATH = Path("reports/feature_importance.csv")

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
    feature_columns = [col for col in df.columns if col not in available_traits]

    X = df[feature_columns].copy()
    y_df = df[available_traits]

    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    if categorical_columns:
        encoded_cats = encoder.fit_transform(X[categorical_columns])
        encoded_cat_names = encoder.get_feature_names_out(categorical_columns)

        X_encoded = pd.concat(
            [
                X[numeric_columns].reset_index(drop=True),
                pd.DataFrame(encoded_cats, columns=encoded_cat_names),
            ],
            axis=1,
        )
    else:
        X_encoded = X[numeric_columns]

    all_importances = []

    for trait in available_traits:
        y = y_df[trait]

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
        )

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
        )

        model.fit(X_train, y_train)

        trait_importance = pd.DataFrame(
            {
                "trait": trait,
                "feature": X_encoded.columns,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        all_importances.append(trait_importance)

        print(f"\nTop features for {trait}:")
        print(trait_importance.head(10))

    importance_df = pd.concat(all_importances, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved feature importances to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()