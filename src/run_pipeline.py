"""
Run the main InkPersona pipeline.

This script runs the core project steps in sequence:
1. Create text generation training pairs
2. Split training data
3. Train and save trait prediction models
4. Run trait prediction inference
5. Generate a report from predicted traits

This supports the ML Operations extra criterion by providing a reproducible
pipeline entry point.
"""

import subprocess
import sys


PIPELINE_STEPS = [
    ("Create training data", "src/create_training_data.py"),
    ("Split training data", "src/split_training_data.py"),
    ("Train and save models", "src/train_and_save_model.py"),
    ("Predict traits", "src/predict_traits.py"),
    ("Generate report from predictions", "src/generate_report_from_predictions.py"),
]


def run_step(step_name: str, script_path: str) -> None:
    """Run one pipeline step."""
    print(f"\nRunning step: {step_name}")
    print("-" * 60)

    result = subprocess.run(
        [sys.executable, script_path],
        check=False,
        text=True,
        capture_output=True,
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Pipeline step failed: {step_name}")


def main() -> None:
    """Run all pipeline steps."""
    print("Starting InkPersona pipeline")

    for step_name, script_path in PIPELINE_STEPS:
        run_step(step_name, script_path)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()