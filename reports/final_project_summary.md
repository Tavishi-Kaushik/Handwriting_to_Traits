# Final Project Summary: InkPersona

## Project Title

InkPersona: Handwriting-Based Personality Report Generation

## Problem Statement

The project explores whether handwriting-derived features and Big Five personality trait scores can be converted into cautious natural-language personality-style reports using a GenAI text generation pipeline.

The task is not intended to produce psychological diagnoses. Instead, it focuses on generating readable, cautious, dataset-based interpretations from structured input data.

## Dataset

The project uses the Handwriting & Personality Traits Dataset from Kaggle.

The downloaded dataset contains:

- 2000 rows
- 24 columns
- handwriting sample references
- writing speed
- gender
- age
- 15 numeric handwriting-derived features
- Big Five trait scores:
  - Openness
  - Conscientiousness
  - Extraversion
  - Agreeableness
  - Neuroticism

The `Handwriting_Sample` column contains values such as `sample_1.jpg`, but the downloaded dataset only includes the CSV file and not the actual image files. Therefore, the current version of the project is treated as a structured tabular-to-text generation task.

## Methodology

The project follows a multi-stage pipeline:

1. Load and inspect the dataset.
2. Check whether the dataset contains actual handwriting images.
3. Convert structured rows into input-output text generation pairs.
4. Create deterministic template-based reports as a baseline.
5. Train baseline models to predict Big Five trait scores.
6. Compare trait prediction models.
7. Save trained models for reuse.
8. Generate reports from predicted trait scores.
9. Add feature importance analysis for explainability.
10. Add experiment logging and a reproducible pipeline for MLOps.
11. Build a Streamlit UI for demo purposes.

## Text Generation Component

The project includes two report generation approaches:

### Template-Based Baseline

A deterministic template generator creates personality-style reports using fixed sentence structures. This is used only as a baseline.

### GenAI-Oriented Generation

The project includes a T5-small fine-tuning script for structured input-to-report generation. The goal is to compare more flexible generated reports against the deterministic template baseline.

## Explainable AI Component

The Explainable AI component uses feature importance from Random Forest models to identify which handwriting-derived features contribute most to each Big Five trait prediction.

The explanations are phrased cautiously because the feature importances are relatively close, meaning no single feature strongly dominates the model.

## ML Operations Component

The project includes several MLOps-style practices:

- organized project structure
- `.gitignore` for data, model files, and generated outputs
- reproducible train/validation/test split
- saved model artifacts
- inference script
- experiment logging
- pipeline runner
- requirements file
- configuration file
- Streamlit app for local demo

## Current Limitations

The current dataset does not include actual handwriting image files, so the project does not perform image-based handwriting analysis.

Baseline model performance is weak, with negative R² values in early experiments. This suggests that the handwriting-derived features may not strongly predict Big Five scores, or that the dataset may be synthetic or weakly correlated.

The generated reports should therefore be interpreted as cautious dataset-based summaries rather than reliable personality assessments.

## Future Work

Possible future improvements include:

- using an actual handwriting image dataset
- building a multimodal image-to-text generation pipeline
- improving trait prediction with better feature engineering
- adding stronger text generation evaluation metrics
- comparing template reports with T5-generated reports
- using SHAP for deeper explainability
- deploying the Streamlit app online

## Final Disclaimer

This project is for academic GenAI experimentation only. It should not be used for psychological diagnosis, employment screening, personality assessment, or any real-world decision-making.