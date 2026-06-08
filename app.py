"""
InkPersona Streamlit App

A simple UI for:
- selecting a handwriting sample
- predicting Big Five personality trait scores
- generating a cautious personality-style report
- showing basic explainability notes
"""

from pathlib import Path
import random

import joblib
import pandas as pd
import streamlit as st


DATA_PATH = Path("data/raw/handwriting_personality_large_dataset.csv")
MODEL_DIR = Path("models")
EXPLANATION_PATH = Path("reports/trait_explanations.txt")

TRAIT_COLUMNS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]

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
    """Convert score into low, medium, or high."""
    if score >= 0.67:
        return "high"
    if score >= 0.34:
        return "medium"
    return "low"


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load dataset."""
    if not DATA_PATH.exists():
        st.error(
            "Dataset not found. Please place handwriting_personality_large_dataset.csv "
            "inside data/raw/."
        )
        st.stop()

    return pd.read_csv(DATA_PATH)


def predict_traits(sample: pd.DataFrame) -> dict:
    """Load saved models and predict Big Five scores."""
    predictions = {}

    for trait in TRAIT_COLUMNS:
        model_path = MODEL_DIR / f"{trait.lower()}_model.joblib"

        if not model_path.exists():
            st.error(
                f"Model not found: {model_path}. "
                "Please run: python3 src/train_and_save_model.py"
            )
            st.stop()

        model = joblib.load(model_path)
        predictions[trait] = round(float(model.predict(sample)[0]), 4)

    return predictions


def generate_report(predictions: dict) -> str:
    """Generate a cautious natural-language report."""
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
            f"{trait} is predicted to be {level} with a score of {score}. "
            f"This may relate to {meaning}."
        )

    report_parts.append(random.choice(closings))

    return " ".join(report_parts)


def load_explanations() -> str:
    """Load explanation text if available."""
    if EXPLANATION_PATH.exists():
        return EXPLANATION_PATH.read_text(encoding="utf-8")

    return (
        "Explainability notes are not available yet. "
        "Please run: python3 src/feature_importance.py and "
        "python3 src/generate_explanations.py"
    )


def main() -> None:
    st.set_page_config(
        page_title="InkPersona",
        page_icon="🖋️",
        layout="wide",
    )

    st.title("🖋️ InkPersona")
    st.subheader("Handwriting-Based Personality Report Generation")

    st.markdown(
        """
        This app predicts Big Five personality-style scores from handwriting-derived
        tabular features and generates a cautious natural-language report.
        
        **Note:** This is a dataset-based interpretation and not a psychological diagnosis.
        """
    )

    df = load_data()

    available_traits = [col for col in TRAIT_COLUMNS if col in df.columns]
    feature_columns = [col for col in df.columns if col not in available_traits]

    st.sidebar.header("Input Selection")

    sample_index = st.sidebar.slider(
        "Choose sample row",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
    )

    selected_row = df.iloc[[sample_index]]
    sample_features = selected_row[feature_columns].fillna("missing")

    st.sidebar.write("Selected sample:")
    if "Handwriting_Sample" in selected_row.columns:
        st.sidebar.code(selected_row["Handwriting_Sample"].iloc[0])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Handwriting-Derived Features")
        feature_display = sample_features.T.reset_index()
        feature_display.columns = ["Feature", "Value"]
        feature_display["Value"] = feature_display["Value"].astype(str)
        st.dataframe(feature_display, use_container_width=True)

    with col2:
        st.markdown("### Actual Dataset Trait Scores")
        if available_traits:
            actual_scores = selected_row[available_traits].T
            actual_scores.columns = ["Actual Score"]
            st.dataframe(actual_scores)
        else:
            st.warning("No actual trait score columns found.")

    st.divider()

    if st.button("Generate Personality Report", type="primary"):
        predictions = predict_traits(sample_features)
        report = generate_report(predictions)

        st.markdown("### Predicted Big Five Scores")

        score_df = pd.DataFrame(
            {
                "Trait": list(predictions.keys()),
                "Predicted Score": list(predictions.values()),
                "Level": [score_to_level(score) for score in predictions.values()],
            }
        )

        st.dataframe(score_df, use_container_width=True)

        st.markdown("### Generated Personality-Style Report")
        st.info(report)

        st.markdown("### Explainable AI Notes")
        st.text(load_explanations())

    st.divider()

    st.markdown(
        """
        ### Project Components
        
        - **Text Generation:** Generates cautious personality-style reports.
        - **Baseline Comparison:** Uses deterministic template reports as a baseline.
        - **Explainable AI:** Uses feature importance to explain trait prediction.
        - **MLOps:** Includes scripts for preprocessing, model training, model saving, inference, and pipeline execution.
        """
    )


if __name__ == "__main__":
    main()