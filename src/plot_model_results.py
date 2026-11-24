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
RESULTS_PATH = ROOT / "outputs" / "results" / "model_method_experiments.csv"
PREDICTIONS_PATH = ROOT / "outputs" / "results" / "vehicle_market_price_predictions.csv"
EXPERIMENT_PREDICTIONS_PATH = ROOT / "outputs" / "results" / "model_method_experiment_predictions.csv"
IMPORTANCE_PATH = ROOT / "outputs" / "results" / "feature_permutation_importance.csv"
COLLINEARITY_PATH = ROOT / "outputs" / "results" / "feature_collinearity.csv"
VIF_PATH = ROOT / "outputs" / "results" / "numeric_feature_vif.csv"
TUNING_PATH = ROOT / "outputs" / "results" / "selected_model_tuning.csv"
ANOMALY_SUMMARY_PATH = ROOT / "outputs" / "results" / "anomaly_detection_summary.csv"
SHAP_GLOBAL_PATH = ROOT / "outputs" / "results" / "shap_global_importance.csv"
SHAP_LOCAL_PATH = ROOT / "outputs" / "results" / "shap_local_explanations.csv"
FIGURE_DIR = ROOT / "outputs" / "figures"
EVALUATIONS = [
    "Dataset 1 chronological validation",
    "Dataset 2 later-period test",
    "Combined random 80/20 split",
]
CONFIGURATIONS = ["A: vehicle-only", "B: vehicle + market context"]
MODELS = [
    "Linear Regression",
    "Ridge Regression",
    "Log-target Linear Regression",
    "Log-target Ridge Regression",
    "Random Forest",
    "XGBoost",
]
DISPLAY_NAMES = [
    "Linear\nreg.",
    "Ridge",
    "Log-linear",
    "Log-Ridge",
    "Random\nforest",
    "XGBoost",
]
COLORS = ["#3b6ea8", "#e58b32"]


def plot_performance_comparison(results: pd.DataFrame) -> None:
    """Compare all model methods and feature sets on both held-out periods."""
    held_out = results.loc[results["evaluation"].isin(EVALUATIONS)]
    figure, axes = plt.subplots(3, 3, figsize=(18, 14))
    metrics = [
        ("mae_lkr", "MAE", 1_000_000, "LKR millions"),
        ("rmse_lkr", "RMSE", 1_000_000, "LKR millions"),
        ("r2", "R²", 1, "Score"),
    ]
    x_positions = np.arange(len(MODELS))
    bar_width = 0.36

    row_titles = {
        EVALUATIONS[0]: "Dataset 1 validation",
        EVALUATIONS[1]: "Dataset 2 later test",
        EVALUATIONS[2]: "Mixed-period random split",
    }
    for row, evaluation in enumerate(EVALUATIONS):
        period_results = held_out.loc[held_out["evaluation"] == evaluation]
        for column, (metric, title, divisor, axis_label) in enumerate(metrics):
            axis = axes[row, column]
            for config_index, configuration in enumerate(CONFIGURATIONS):
                selected = period_results.loc[
                    period_results["configuration"] == configuration
                ].set_index("model")
                values = [selected.loc[name, metric] / divisor for name in MODELS]
                offset = (config_index - 0.5) * bar_width
                axis.bar(
                    x_positions + offset,
                    values,
                    bar_width,
                    label=configuration,
                    color=COLORS[config_index],
                )

            axis.set_title(f"{row_titles[evaluation]}\n{title}")
            axis.set_ylabel(axis_label)
            axis.set_xticks(
                x_positions,
                DISPLAY_NAMES,
                rotation=15,
                ha="right",
                fontsize=9,
            )
            axis.grid(axis="y", alpha=0.25)
            if metric == "r2":
                axis.set_ylim(min(-0.1, period_results[metric].min() - 0.05), 1)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Model comparison", fontsize=14, y=0.99)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.88])
    figure.savefig(
        FIGURE_DIR / "model_performance_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_cross_validation_comparison(results: pd.DataFrame) -> None:
    """Compare chronological and shuffled CV means and fold variation."""
    cv_results = results.loc[
        results["evaluation"].str.contains("CV fold", na=False)
    ].copy()
    cv_results["cv_strategy"] = np.where(
        cv_results["evaluation"].str.startswith("Combined shuffled"),
        "Shuffled CV (both datasets)",
        "Forward CV (Dataset 1)",
    )
    summary = cv_results.groupby(["cv_strategy", "configuration", "model"])[
        ["mae_lkr", "rmse_lkr", "r2"]
    ].agg(["mean", "std"])
    strategies = ["Forward CV (Dataset 1)", "Shuffled CV (both datasets)"]
    figure, axes = plt.subplots(2, 3, figsize=(18, 11))
    metrics = [
        ("mae_lkr", "MAE", 1_000_000, "LKR millions"),
        ("rmse_lkr", "RMSE", 1_000_000, "LKR millions"),
        ("r2", "R²", 1, "Score"),
    ]
    x_positions = np.arange(len(MODELS))
    bar_width = 0.36

    for row, strategy in enumerate(strategies):
        for column, (metric, title, divisor, axis_label) in enumerate(metrics):
            axis = axes[row, column]
            for config_index, configuration in enumerate(CONFIGURATIONS):
                means = [
                    summary.loc[(strategy, configuration, model), (metric, "mean")]
                    / divisor
                    for model in MODELS
                ]
                deviations = [
                    summary.loc[(strategy, configuration, model), (metric, "std")]
                    / divisor
                    for model in MODELS
                ]
                offset = (config_index - 0.5) * bar_width
                axis.bar(
                    x_positions + offset,
                    means,
                    bar_width,
                    yerr=deviations,
                    capsize=3,
                    label=configuration,
                    color=COLORS[config_index],
                )

            axis.set_title(f"{strategy}\n{title}")
            axis.set_ylabel(axis_label)
            axis.set_xticks(
                x_positions,
                DISPLAY_NAMES,
                rotation=15,
                ha="right",
                fontsize=9,
            )
            axis.grid(axis="y", alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Cross-validation (mean ± SD)", fontsize=14, y=0.99)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.93])
    figure.savefig(
        FIGURE_DIR / "model_cross_validation_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_feature_diagnostics(
    importance: pd.DataFrame,
    correlations: pd.DataFrame,
    vif: pd.DataFrame,
) -> None:
    """Plot XGBoost validation importance beside numeric feature correlations."""
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 8),
        gridspec_kw={"width_ratios": [1, 1, 1.1]},
    )
    configurations = CONFIGURATIONS
    feature_labels = {
        "vehicle_age_years": "Vehicle age (years)",
        "engine_cc": "Engine capacity (cc)",
        "make": "Make",
        "model": "Model",
        "transmission": "Transmission",
        "mileage_km": "Mileage (km)",
        "fuel_type": "Fuel type",
        "condition": "Condition",
        "usd_lkr_monthly_average": "USD/LKR monthly average",
        "import_policy_environment": "Import policy environment",
    }
    for axis, configuration in zip(axes[:2], configurations):
        selected = importance.loc[
            (importance["model"] == "XGBoost")
            & (importance["configuration"] == configuration)
        ].sort_values("mae_increase_lkr")
        axis.barh(
            selected["feature"].map(feature_labels),
            selected["mae_increase_lkr"] / 1_000_000,
            xerr=selected["mae_increase_std_lkr"] / 1_000_000,
            color=COLORS[configurations.index(configuration)],
            capsize=2,
        )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(configuration)
        axis.set_xlabel("MAE increase (LKR millions)")
        axis.grid(axis="x", alpha=0.25)

    matrix = correlations.pivot(
        index="feature",
        columns="other_feature",
        values="correlation",
    )
    feature_order = [
        "vehicle_age_years",
        "engine_cc",
        "mileage_km",
        "usd_lkr_monthly_average",
    ]
    matrix = matrix.reindex(index=feature_order, columns=feature_order)
    axis = axes[2]
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    labels = ["Vehicle age", "Engine cc", "Mileage", "USD/LKR"]
    axis.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(
                column,
                row,
                f"{matrix.iloc[row, column]:.2f}",
                ha="center",
                va="center",
                color="black",
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_title(f"Correlations (max VIF {vif['vif'].max():.2f})")
    figure.suptitle("Feature checks", fontsize=14)
    figure.tight_layout(rect=[0, 0, 1, 0.94])
    figure.savefig(
        FIGURE_DIR / "feature_diagnostics.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_tuning_comparison(tuning_results: pd.DataFrame) -> None:
    """Compare the small parameter search using five Dataset 1 time folds."""
    summary = tuning_results.groupby(["model", "setting"])[
        ["mae_lkr", "rmse_lkr", "r2"]
    ].agg(["mean", "std"])
    models = ["Random Forest", "XGBoost"]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    metrics = [
        ("mae_lkr", "Mean MAE", 1_000_000, "LKR millions"),
        ("rmse_lkr", "Mean RMSE", 1_000_000, "LKR millions"),
        ("r2", "Mean R²", 1, "Score"),
    ]

    for row, model in enumerate(models):
        model_summary = summary.loc[model]
        settings = list(model_summary.index)
        labels = [setting.split(" (")[0] for setting in settings]
        positions = np.arange(len(settings))
        for column, (metric, title, divisor, axis_label) in enumerate(metrics):
            axis = axes[row, column]
            means = model_summary[(metric, "mean")].to_numpy() / divisor
            deviations = model_summary[(metric, "std")].to_numpy() / divisor
            axis.bar(
                positions,
                means,
                yerr=deviations,
                capsize=3,
                color=COLORS[row],
            )
            axis.set_title(f"{model}: {title}")
            axis.set_ylabel(axis_label)
            axis.set_xticks(positions, labels)
            axis.grid(axis="y", alpha=0.25)

    figure.suptitle("Tuning check (Dataset 1, 5 folds)", fontsize=14)
    figure.text(0.5, 0.02, "Error bars show fold standard deviation.", ha="center")
    figure.tight_layout(rect=[0, 0.04, 1, 0.93])
    figure.savefig(
        FIGURE_DIR / "selected_model_tuning_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_prediction_comparison(predictions: pd.DataFrame, evaluation: str) -> None:
    """Compare actual and predicted prices for every model and feature set."""
    filename = (
        "dataset1_validation_actual_vs_predicted.png"
        if evaluation == EVALUATIONS[0]
        else "dataset2_test_actual_vs_predicted.png"
    )
    figure_path = FIGURE_DIR / filename
    newest_predictions = max(
        PREDICTIONS_PATH.stat().st_mtime,
        EXPERIMENT_PREDICTIONS_PATH.stat().st_mtime,
    )
    if figure_path.exists() and figure_path.stat().st_mtime >= newest_predictions:
        return

    period_predictions = predictions.loc[predictions["evaluation"] == evaluation]
    figure, axes = plt.subplots(len(MODELS), 2, figsize=(14, 22))
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

            axis.scatter(
                actual,
                predicted,
                s=5,
                alpha=0.22,
                color="#3569a8",
                rasterized=True,
            )
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

    figure.supxlabel("Actual price (LKR millions)")
    figure.supylabel("Predicted price (LKR millions)")
    figure.suptitle(f"Actual vs predicted: {evaluation}", fontsize=14)
    figure.text(0.5, 0.005, "Sym-log axes include negative predictions.", ha="center")
    axes[0, 0].legend(frameon=False, loc="lower right")
    figure.tight_layout(rect=[0.03, 0.03, 1, 0.96])
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_anomaly_review(summary: pd.DataFrame) -> None:
    """Compare price and vehicle-profile flags."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    evaluations = [
        "Dataset 1 chronological validation",
        "Dataset 2 later-period test",
    ]
    labels = ["D1 validation", "D2 later test"]
    models = ["Random Forest", "XGBoost depth 8"]
    colors = ["#3b6ea8", "#e58b32"]

    for model_name, color in zip(models, colors):
        rates = []
        for evaluation in evaluations:
            row = summary.loc[
                (summary["evaluation"] == evaluation)
                & (summary["prediction_model"] == model_name)
            ].iloc[0]
            rates.append(100 * row["high_price_flags"] / row["listings"])
        axes[0].plot(labels, rates, marker="o", linewidth=2, color=color, label=model_name)

    axes[0].set_title("High price residuals")
    axes[0].set_ylabel("Share of listings (%)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    selected = summary.loc[summary["evaluation"].isin(evaluations)]
    positions = np.arange(len(evaluations))
    width = 0.36
    for model_index, (model_name, color) in enumerate(zip(models, colors)):
        rows = selected.loc[selected["prediction_model"] == model_name].set_index("evaluation")
        lof_rates = [100 * rows.loc[evaluation, "unusual_vehicle_profiles"] / rows.loc[evaluation, "listings"] for evaluation in evaluations]
        axes[1].bar(
            positions + (model_index - 0.5) * width,
            lof_rates,
            width,
            color=color,
            label=model_name,
        )

    axes[1].set_title("Unusual vehicle profiles (LOF)")
    axes[1].set_ylabel("Share of listings (%)")
    axes[1].set_xticks(positions, labels)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    figure.suptitle("Anomaly flags")
    figure.tight_layout(rect=[0, 0, 1, 0.93])
    figure.savefig(FIGURE_DIR / "anomaly_detection_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_shap_explanations(global_values: pd.DataFrame, local_values: pd.DataFrame) -> None:
    """Plot sampled SHAP values and one flagged listing."""
    models = ["Random Forest", "XGBoost depth 8"]
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17, 7),
        gridspec_kw={"width_ratios": [1, 1, 1.2]},
    )

    for axis, model_name, color in zip(axes[:2], models, COLORS):
        selected = global_values.loc[
            global_values["prediction_model"] == model_name
        ].nlargest(8, "mean_absolute_shap_lkr").sort_values(
            "mean_absolute_shap_lkr"
        )
        axis.barh(
            selected["feature"].str.replace("_", " ", regex=False).str.title(),
            selected["mean_absolute_shap_lkr"] / 1_000_000,
            color=color,
        )
        sample_size = int(selected["explained_rows"].iloc[0])
        axis.set_title(f"{model_name} (n = {sample_size})")
        axis.set_xlabel("Mean |SHAP| (LKR millions)")
        axis.grid(axis="x", alpha=0.25)

    local = local_values.loc[
        (local_values["prediction_model"] == "Random Forest")
        & (local_values["price_flag"] == "high")
    ]
    case_key = local.sort_values("relative_deviation", ascending=False).iloc[0]
    case_rows = local.loc[
        (local["listing_date"] == case_key["listing_date"])
        & (local["make"] == case_key["make"])
        & (local["vehicle_model"] == case_key["vehicle_model"])
        & (local["year"] == case_key["year"])
    ].assign(
        absolute_contribution=lambda rows: rows["shap_contribution_lkr"].abs()
    ).nlargest(6, "absolute_contribution").sort_values("shap_contribution_lkr")
    axes[2].barh(
        case_rows["feature"].str.replace("_", " ", regex=False).str.title(),
        case_rows["shap_contribution_lkr"] / 1_000_000,
        color=[
            "#4b78a6" if value < 0 else "#a64b4b"
            for value in case_rows["shap_contribution_lkr"]
        ],
    )
    axes[2].axvline(0, color="black", linewidth=0.8)
    axes[2].set_title(
        f"Random Forest: {case_key['make']} {case_key['vehicle_model']} ({case_key['year']})"
    )
    axes[2].set_xlabel("Feature contribution (LKR millions)")
    axes[2].grid(axis="x", alpha=0.25)
    figure.suptitle("What influenced the model predictions? (Dataset 2)")
    figure.tight_layout(rect=[0, 0, 1, 0.93])
    figure.savefig(
        FIGURE_DIR / "shap_model_explanations.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    # Load saved scores and predictions; charting does not train models.
    results = pd.read_csv(RESULTS_PATH)
    baseline_predictions = pd.read_csv(PREDICTIONS_PATH)
    experiment_predictions = pd.read_csv(EXPERIMENT_PREDICTIONS_PATH)
    feature_importance = pd.read_csv(IMPORTANCE_PATH)
    feature_correlations = pd.read_csv(COLLINEARITY_PATH)
    numeric_vif = pd.read_csv(VIF_PATH)
    tuning_results = pd.read_csv(TUNING_PATH)
    anomaly_summary = pd.read_csv(ANOMALY_SUMMARY_PATH)
    shap_global = pd.read_csv(SHAP_GLOBAL_PATH)
    shap_local = pd.read_csv(SHAP_LOCAL_PATH)
    predictions = pd.concat(
        [baseline_predictions, experiment_predictions],
        ignore_index=True,
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_performance_comparison(results)
    plot_cross_validation_comparison(results)
    plot_feature_diagnostics(feature_importance, feature_correlations, numeric_vif)
    plot_tuning_comparison(tuning_results)
    for evaluation in EVALUATIONS:
        plot_prediction_comparison(predictions, evaluation)
    plot_anomaly_review(anomaly_summary)
    plot_shap_explanations(shap_global, shap_local)

    print(f"Saved static PNG charts to {FIGURE_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
