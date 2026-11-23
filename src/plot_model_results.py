"""Save static model-comparison and prediction-diagnostic charts."""

from pathlib import Path
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "vehicle_market_matplotlib"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "outputs" / "results" / "vehicle_market_model_results.csv"
PREDICTIONS_PATH = ROOT / "outputs" / "results" / "vehicle_market_price_predictions.csv"
FIGURE_DIR = ROOT / "outputs" / "figures"
EVALUATIONS = [
    "Dataset 1 chronological validation",
    "Dataset 2 later-period test",
]
CONFIGURATIONS = ["A: vehicle-only", "B: vehicle + market context"]
MODELS = ["Linear Regression", "Random Forest", "XGBoost"]
COLORS = ["#3b6ea8", "#e58b32"]


def plot_performance_comparison(results: pd.DataFrame) -> None:
    """Compare MAE, RMSE, and R² for both configurations and periods."""
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    metrics = [
        ("mae_lkr", "MAE", 1_000_000, "LKR millions"),
        ("rmse_lkr", "RMSE", 1_000_000, "LKR millions"),
        ("r2", "R²", 1, "Score"),
    ]
    x_positions = np.arange(len(MODELS))
    bar_width = 0.36

    for row, evaluation in enumerate(EVALUATIONS):
        period_results = results.loc[results["evaluation"] == evaluation]
        for column, (metric, title, divisor, axis_label) in enumerate(metrics):
            axis = axes[row, column]

            for config_index, configuration in enumerate(CONFIGURATIONS):
                selected = period_results.loc[
                    period_results["configuration"] == configuration
                ]
                values = [
                    selected.loc[selected["model"] == name, metric].iloc[0] / divisor
                    for name in MODELS
                ]
                offset = (config_index - 0.5) * bar_width
                bars = axis.bar(
                    x_positions + offset,
                    values,
                    bar_width,
                    label=configuration,
                    color=COLORS[config_index],
                )
                format_string = "%.3f" if metric == "r2" else "%.2f"
                axis.bar_label(
                    bars,
                    fmt=format_string,
                    label_type="center",
                    color="white",
                    fontsize=8,
                )

            axis.set_title(f"{evaluation}\n{title}")
            axis.set_ylabel(axis_label)
            axis.set_xticks(x_positions, MODELS, rotation=12)
            axis.grid(axis="y", alpha=0.25)
            if metric == "r2":
                axis.set_ylim(0, 1)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Price model performance by feature configuration", fontsize=14, y=0.99)
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
    figure.tight_layout(rect=[0, 0, 1, 0.89])
    figure.savefig(FIGURE_DIR / "model_performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_prediction_comparison(predictions: pd.DataFrame, evaluation: str) -> None:
    """Plot actual versus predicted prices for each model and configuration."""
    period_predictions = predictions.loc[predictions["evaluation"] == evaluation]
    figure, axes = plt.subplots(3, 2, figsize=(12, 14))
    million_formatter = FuncFormatter(lambda value, _: f"{value:g}")

    for row, model_name in enumerate(MODELS):
        for column, configuration in enumerate(CONFIGURATIONS):
            axis = axes[row, column]
            selected = period_predictions.loc[
                (period_predictions["model"] == model_name)
                & (period_predictions["configuration"] == configuration)
            ]
            actual = selected["actual_lkr"] / 1_000_000
            predicted = selected["predicted_lkr"] / 1_000_000
            lower = min(0, actual.min(), predicted.min())
            upper = max(actual.max(), predicted.max())

            # Symmetric log axes retain negative linear-regression predictions and high prices.
            axis.scatter(actual, predicted, s=7, alpha=0.25, color="#3569a8", rasterized=True)
            axis.plot(
                [lower, upper],
                [lower, upper],
                color="#e07a2d",
                linestyle="--",
                linewidth=1.4,
                label="Perfect prediction",
            )
            axis.set_xscale("symlog", linthresh=0.1)
            axis.set_yscale("symlog", linthresh=0.1)
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            axis.xaxis.set_major_formatter(million_formatter)
            axis.yaxis.set_major_formatter(million_formatter)
            axis.set_title(f"{model_name}\n{configuration}")
            axis.grid(True, which="both", alpha=0.2)

    figure.supxlabel("Actual asking price (LKR millions; symlog scale)")
    figure.supylabel("Predicted asking price (LKR millions; symlog scale)")
    figure.suptitle(f"Actual versus predicted prices — {evaluation}", fontsize=14)
    axes[0, 0].legend(frameon=False, loc="lower right")
    figure.tight_layout(rect=[0.03, 0.03, 1, 0.96])

    filename = (
        "dataset1_validation_actual_vs_predicted.png"
        if evaluation == EVALUATIONS[0]
        else "dataset2_test_actual_vs_predicted.png"
    )
    figure.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    # Read saved scores and predictions so charting does not retrain models.
    results = pd.read_csv(RESULTS_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_performance_comparison(results)
    for evaluation in EVALUATIONS:
        plot_prediction_comparison(predictions, evaluation)

    print(f"Saved three static PNG charts to {FIGURE_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
