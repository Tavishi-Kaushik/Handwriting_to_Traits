"""
Generate simple human-readable explanations from feature importance results.

This script converts model feature importances into short explanation text
for each Big Five personality trait.
"""

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("reports/feature_importance.csv")
OUTPUT_PATH = Path("reports/trait_explanations.txt")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature importance file not found at {INPUT_PATH}. "
            "Please run src/feature_importance.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    explanations = []

    for trait in df["trait"].unique():
        trait_df = df[df["trait"] == trait].sort_values(
            "importance", ascending=False
        )

        top_features = trait_df.head(3)

        feature_text = ", ".join(top_features["feature"].tolist())

        explanation = (
            f"For {trait}, the model placed the highest importance on "
            f"{feature_text}. This suggests that these handwriting-related "
            f"features had the strongest influence on the model's prediction "
            f"for this trait. However, the importance values are close together, "
            f"so this should be interpreted cautiously rather than as a strong "
            f"psychological conclusion."
        )

        explanations.append(explanation)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        for explanation in explanations:
            file.write(explanation + "\n\n")

    print(f"Saved explanations to: {OUTPUT_PATH}")
    print("\nGenerated explanations:\n")
    for explanation in explanations:
        print(explanation)
        print()


if __name__ == "__main__":
    main()