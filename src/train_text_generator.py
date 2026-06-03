"""
Fine-tune T5-small for personality-style report generation.

Input:
- data/processed/train.csv
- data/processed/val.csv

Each file should contain:
- input_text
- target_text

This script supports the Text Generation part of the project.
"""

from pathlib import Path

from datasets import Dataset
import pandas as pd
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/val.csv")
OUTPUT_DIR = Path("models/t5_inkpersona")

MODEL_NAME = "t5-small"

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 180


def load_data() -> tuple[Dataset, Dataset]:
    """Load train and validation data."""
    if not TRAIN_PATH.exists() or not VAL_PATH.exists():
        raise FileNotFoundError(
            "Train/validation files not found. "
            "Please run src/create_training_data.py and src/split_training_data.py first."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    required_columns = {"input_text", "target_text"}

    if not required_columns.issubset(train_df.columns):
        raise ValueError("Train file must contain input_text and target_text columns.")

    if not required_columns.issubset(val_df.columns):
        raise ValueError("Validation file must contain input_text and target_text columns.")

    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)


def preprocess_examples(examples, tokenizer):
    """Tokenize input and target text."""
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )

    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main() -> None:
    train_dataset, val_dataset = load_data()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    tokenized_train = train_dataset.map(
        lambda examples: preprocess_examples(examples, tokenizer),
        batched=True,
    )

    tokenized_val = val_dataset.map(
        lambda examples: preprocess_examples(examples, tokenizer),
        batched=True,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=2,
        predict_with_generate=True,
        # logging_dir="experiments/t5_logs",
        logging_steps=20,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"T5 model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()