"""Run the notebook and research scripts in project order."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "notebooks" / "data_cleaning_&_EDA.ipynb"
OUTPUT_NOTEBOOK_DIR = ROOT / "outputs" / "notebooks"
OUTPUT_DIRS = [
    OUTPUT_NOTEBOOK_DIR,
    ROOT / "outputs" / "results",
    ROOT / "outputs" / "figures",
]
RAW_DATA_FILES = [
    ROOT / "data" / "raw" / "vehicles.csv",
    ROOT / "data" / "raw" / "car_price_dataset.csv",
]


def run_step(label: str, command: list[str]) -> None:
    """Run one project step and show its progress in the terminal."""
    print(f"\n--- {label} ---", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    missing_files = [path for path in RAW_DATA_FILES if not path.is_file()]
    if missing_files:
        print("Copy these source files into data/raw/ before running:")
        for path in missing_files:
            print(f"- {path.relative_to(ROOT)}")
        raise SystemExit(1)

    # Output folders are ignored by Git, so create them on a fresh clone.
    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)

    run_step(
        "Clean data and run EDA notebook",
        [
            sys.executable,
            "-m",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=-1",
            "--output",
            "data_cleaning_&_EDA_executed.ipynb",
            "--output-dir",
            str(OUTPUT_NOTEBOOK_DIR),
            str(NOTEBOOK_PATH),
        ],
    )

    steps = [
        ("Check cleaned datasets", "src/cleaned_datasets_compatibility.py"),
        ("Prepare market context", "src/prepare_market_context.py"),
        ("Prepare vehicle modelling data", "src/prepare_model_data.py"),
        ("Join market context", "src/integrate_market_context.py"),
        ("Check preprocessing and data splits", "src/model_preprocessing.py"),
        (
            "Evaluate vehicle-only models (Linear Regression, Random Forest, XGBoost)",
            "src/price_prediction.py",
        ),
        (
            "Compare vehicle-only and market-aware models",
            "src/compare_vehicle_and_market_aware_models.py",
        ),
        (
            "Compare model methods and validation approaches",
            "src/compare_model_method_experiments.py",
        ),
        ("Tune selected models", "src/tune_selected_models.py"),
        (
            "Analyze feature importance and collinearity",
            "src/analyze_model_features.py",
        ),
        ("Detect price anomalies", "src/analyze_price_anomalies.py"),
        ("Prepare SHAP explanations", "src/explain_model_predictions.py"),
        ("Create result plots", "src/plot_model_results.py"),
    ]

    for label, script in steps:
        run_step(label, [sys.executable, script])

    print("\nProject run completed.", flush=True)
    print("Executed notebook: outputs/notebooks/data_cleaning_&_EDA_executed.ipynb")
    print("Results and figures: outputs/results/ and outputs/figures/")
    print("The interactive sample check and Streamlit demo are run separately.")


if __name__ == "__main__":
    main()
