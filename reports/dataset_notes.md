# Dataset Notes

Dataset: Handwriting & Personality Traits Dataset from Kaggle

The downloaded dataset contains one CSV file:

- `handwriting_personality_large_dataset.csv`

The dataset has 2000 rows and 24 columns.

Important columns include:

- `Handwriting_Sample`
- `Writing_Speed_wpm`
- `Openness`
- `Conscientiousness`
- `Extraversion`
- `Agreeableness`
- `Neuroticism`
- `Gender`
- `Age`
- `Feature_1` to `Feature_15`

The `Handwriting_Sample` column contains values such as `sample_1.jpg`, `sample_2.jpg`, etc. These look like image filenames. However, the current Kaggle download only contains the CSV file and does not include the actual `.jpg` or `.png` handwriting images.

Therefore, the current project will treat the dataset as a structured/tabular dataset. A multimodal extension may be added later if actual handwriting image files are found or if a separate handwriting image dataset is used.

This is important because the deterministic template-based report generator will only be used as a baseline. The main GenAI task will focus on generating more flexible natural-language reports from structured handwriting features and predicted Big Five trait scores.