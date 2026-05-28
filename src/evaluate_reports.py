"""
Evaluate generated personality reports.

This script computes simple text statistics for the template-generated reports:
- number of reports
- average word count
- minimum and maximum word count
- average character count

This provides a basic evaluation baseline before comparing with a GenAI model.
"""

from pathlib import Path

import pandas as pd


REPORT_PATH = Path("reports/template_reports.csv")
OUTPUT_PATH = Path("reports/report_evaluation_summary.txt")


def count_words(text: str) -> int:
    """Count words in a text string."""
    return len(str(text).split())


def main() -> None:
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Template report file not found at {REPORT_PATH}. "
            "Please run src/template_report_generator.py first."
        )

    df = pd.read_csv(REPORT_PATH)

    if "template_report" not in df.columns:
        raise ValueError("Expected column 'template_report' not found.")

    df["word_count"] = df["template_report"].apply(count_words)
    df["char_count"] = df["template_report"].astype(str).apply(len)

    summary = {
        "number_of_reports": len(df),
        "average_word_count": round(df["word_count"].mean(), 2),
        "min_word_count": int(df["word_count"].min()),
        "max_word_count": int(df["word_count"].max()),
        "average_character_count": round(df["char_count"].mean(), 2),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        file.write("Report Evaluation Summary\n")
        file.write("=========================\n\n")
        for key, value in summary.items():
            file.write(f"{key}: {value}\n")

    print("Report evaluation summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nSaved summary to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()