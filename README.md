# InkPersona: Handwriting-Based Personality Report Generation

InkPersona is a GenAI text generation project that creates cautious personality-style reports from handwriting-derived tabular features and Big Five personality trait scores.

The project uses the **Handwriting & Personality Traits Dataset** from Kaggle. The dataset contains 2000 rows and 24 columns, including handwriting sample references, writing speed, demographic information, 15 handwriting-derived numeric features, and Big Five trait scores.

Although the dataset contains a `Handwriting_Sample` column with values such as `sample_1.jpg`, the downloaded Kaggle file only includes the CSV and does not include the actual handwriting image files. Therefore, this project treats the dataset as structured/tabular data.

## Project Goal

The goal of this project is to build a text generation pipeline that converts handwriting-related features and predicted personality trait scores into readable, cautious personality-style reports.

The generated reports are dataset-based interpretations and should not be treated as psychological diagnoses.

## Extra Criteria

This project focuses on two extra criteria:

### 1. Explainable AI

The project includes feature importance analysis to identify which handwriting-derived features contribute most to each Big Five trait prediction.

The explainability component is implemented through:

- `src/feature_importance.py`
- `src/generate_explanations.py`

### 2. ML Operations

The project includes MLOps-style structure through:

- reproducible preprocessing scripts
- train/validation/test split
- model comparison
- experiment logging
- saved model artifacts
- inference scripts
- a single pipeline runner
- Streamlit demo app

The MLOps components are implemented through:

- `src/run_experiment.py`
- `src/train_and_save_model.py`
- `src/predict_traits.py`
- `src/run_pipeline.py`

## Project Pipeline

The main workflow is:

```text
Raw dataset
→ data inspection
→ preprocessing
→ training pair creation
→ train/validation/test split
→ trait prediction model training
→ model saving
→ trait prediction inference
→ report generation
→ explainability output
→ Streamlit UI demo

## Repository Structure

Handwriting_to_Traits/
├── app.py
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── experiments/
├── models/
├── notebooks/
│   └── 01_data_exploration.ipynb
├── reports/
│   └── dataset_notes.md
├── src/
│   ├── check_dataset_details.py
│   ├── compare_trait_models.py
│   ├── create_training_data.py
│   ├── evaluate_reports.py
│   ├── feature_importance.py
│   ├── generate_explanations.py
│   ├── generate_report_from_predictions.py
│   ├── generative_report_generator.py
│   ├── inspect_dataset.py
│   ├── load_data.py
│   ├── predict_traits.py
│   ├── preprocess.py
│   ├── run_experiment.py
│   ├── run_pipeline.py
│   ├── split_training_data.py
│   ├── template_report_generator.py
│   ├── train_and_save_model.py
│   ├── train_text_generator.py
│   └── train_trait_predictor.py
├── .gitignore
├── README.md
└── requirements.txt

## Dataset

data/raw/handwriting_personality_large_dataset.csv

## How to Run
## 1. Install dependencies

python3 -m pip install -r requirements.txt

##2. Run the full pipeline
python3 src/run_pipeline.py

##3. Run Explainable AI scripts
python3 src/feature_importance.py
python3 src/generate_explanations.py

##4. Run the Streamlit app
python3 -m streamlit run app.py
 
 Then open:
 http://localhost:8501

# Main Scripts
## Data Inspection

python3 src/check_dataset_details.py
python3 src/inspect_dataset.py

##Preprocessing
python3 src/create_training_data.py
python3 src/split_training_data.py

## Trait Prediction

python3 src/train_trait_predictor.py
python3 src/compare_trait_models.py
python3 src/train_and_save_model.py
python3 src/predict_traits.py

##Report Generation
python3 src/template_report_generator.py
python3 src/generative_report_generator.py
python3 src/generate_report_from_predictions.py

##Text Generation Model
python3 src/train_text_generator.py

##Notes and Limitations
The current dataset is tabular and does not include actual handwriting image files. Because of this, the project does not claim to be a full multimodal system.

The model results should be interpreted cautiously. Early baseline prediction results show weak predictive performance, which suggests that the dataset may not contain strong signal between handwriting-derived features and personality traits.

The template-based report generator is used only as a baseline. The project also includes a T5-small text generation training script to support the GenAI text generation component.

## Disclaimer

InkPersona is an academic GenAI project. The generated reports are based on dataset patterns and should not be used as psychological, clinical, hiring, or diagnostic assessments.


