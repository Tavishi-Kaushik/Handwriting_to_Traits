"""
Generate a personality-style report using the fine-tuned T5 model.

This script loads the trained T5 model from models/t5_inkpersona.
If the final saved model folder is not available, it falls back to the latest checkpoint.
"""

from pathlib import Path

import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DATA_PATH = Path("data/processed/test.csv")
MODEL_DIR = Path("models/t5_inkpersona")


def get_model_path() -> Path:
    """Return final model path or latest checkpoint path."""
    if (MODEL_DIR / "config.json").exists():
        return MODEL_DIR

    checkpoints = sorted(
        [path for path in MODEL_DIR.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: int(path.name.split("-")[-1]),
    )

    if not checkpoints:
        raise FileNotFoundError(
            "No trained T5 model or checkpoint found. "
            "Please run src/train_text_generator.py first."
        )

    return checkpoints[-1]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test data not found at {DATA_PATH}. "
            "Please run src/create_training_data.py and src/split_training_data.py first."
        )

    model_path = get_model_path()

    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    test_df = pd.read_csv(DATA_PATH)

    sample_input = test_df["input_text"].iloc[0]
    target_text = test_df["target_text"].iloc[0]

    inputs = tokenizer(
        sample_input,
        return_tensors="pt",
        max_length=256,
        truncation=True,
    )

    output_ids = model.generate(
        **inputs,
        max_length=180,
        num_beams=4,
        early_stopping=True,
    )

    generated_text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    print("\nInput text:")
    print(sample_input)

    print("\nExpected target report:")
    print(target_text)

    print("\nT5 generated report:")
    print(generated_text)


if __name__ == "__main__":
    main()