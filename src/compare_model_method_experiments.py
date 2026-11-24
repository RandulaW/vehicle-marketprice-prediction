"""Compare regularized and transformed-target models on time-based splits."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split
from xgboost import XGBRegressor

from model_preprocessing import (
    FEATURES,
    MARKET_FEATURES,
    TARGET,
    split_by_period,
)
from price_prediction import make_model, score_model


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "vehicle_market_model_data.csv"
RESULTS_PATH = ROOT / "outputs" / "results" / "model_method_experiments.csv"
SELECTION_PATH = ROOT / "outputs" / "results" / "model_selection_summary.csv"
PREDICTIONS_PATH = ROOT / "outputs" / "results" / "model_method_experiment_predictions.csv"
CV_SPLITS = 5
CV_RANDOM_STATE = 42
RANDOM_SPLIT_EVAL = "Combined random 80/20 split"
# Use one fixed alpha for this initial comparison; tune it only if Ridge is promising.
RIDGE_ALPHA = 10.0
SCATTER_MODELS = {
    "Ridge Regression",
    "Log-target Linear Regression",
    "Log-target Ridge Regression",
}


def get_experiment_models() -> list[tuple[str, object, bool, bool]]:
    """Keep existing tree models as references and add linear alternatives."""
    return [
        ("Linear Regression", LinearRegression(), True, False),
        ("Ridge Regression", Ridge(alpha=RIDGE_ALPHA), True, False),
        ("Log-target Linear Regression", LinearRegression(), True, True),
        (
            "Log-target Ridge Regression",
            Ridge(alpha=RIDGE_ALPHA),
            True,
            True,
        ),
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
            False,
            False,
        ),
        (
            "XGBoost",
            XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            ),
            False,
            False,
        ),
    ]


def make_experiment_model(
    estimator: object,
    scale_numeric: bool,
    include_market_context: bool,
    log_target: bool,
):
    """Build a leakage-safe pipeline and optionally transform the target."""
    if log_target:
        estimator = TransformedTargetRegressor(
            regressor=estimator,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    return make_model(estimator, scale_numeric, include_market_context)


def add_evaluation(
    result_rows: list[dict],
    model_name: str,
    estimator: object,
    scale_numeric: bool,
    log_target: bool,
    configuration: str,
    feature_columns: list[str],
    include_market_context: bool,
    evaluation: str,
    training_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    fold: int | None = None,
    prediction_rows: list[pd.DataFrame] | None = None,
) -> None:
    """Fit once for an evaluation split and retain its scores."""
    print(f"Fitting {configuration} / {model_name}: {evaluation}", flush=True)
    model = make_experiment_model(
        clone(estimator),
        scale_numeric,
        include_market_context,
        log_target,
    )
    model.fit(training_data[feature_columns], training_data[TARGET])
    predictions = model.predict(evaluation_data[feature_columns])
    scores = score_model(
        model,
        evaluation_data[feature_columns],
        evaluation_data[TARGET],
        predictions,
    )
    result_rows.append({
        "configuration": configuration,
        "model": model_name,
        "evaluation": evaluation,
        "fold": fold,
        "training_rows": len(training_data),
        "evaluation_rows": len(evaluation_data),
        "evaluation_start": evaluation_data["listing_date"].min().date(),
        "evaluation_end": evaluation_data["listing_date"].max().date(),
        **scores,
    })
    if prediction_rows is not None and model_name in SCATTER_MODELS:
        prediction_rows.append(pd.DataFrame({
            "configuration": configuration,
            "model": model_name,
            "evaluation": evaluation,
            "listing_date": evaluation_data["listing_date"].dt.strftime("%Y-%m-%d"),
            "dataset": evaluation_data["dataset"].to_numpy(),
            "actual_lkr": evaluation_data[TARGET].to_numpy(),
            "predicted_lkr": predictions,
        }))


def add_median_baseline(
    result_rows: list[dict],
    evaluation: str,
    training_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    fold: int | None = None,
) -> None:
    """Score a feature-free median predictor as a simple reference point."""
    model = DummyRegressor(strategy="median")
    model.fit(np.zeros((len(training_data), 1)), training_data[TARGET])
    predictions = model.predict(np.zeros((len(evaluation_data), 1)))
    scores = score_model(
        model,
        np.zeros((len(evaluation_data), 1)),
        evaluation_data[TARGET],
        predictions,
    )
    result_rows.append({
        "configuration": "Feature-free median",
        "model": "Median baseline",
        "evaluation": evaluation,
        "fold": fold,
        "training_rows": len(training_data),
        "evaluation_rows": len(evaluation_data),
        "evaluation_start": evaluation_data["listing_date"].min().date(),
        "evaluation_end": evaluation_data["listing_date"].max().date(),
        **scores,
    })


def add_forward_chaining_cv(
    data: pd.DataFrame,
    configurations: list[tuple[str, list[str], bool]],
    result_rows: list[dict],
) -> None:
    """Validate on later date blocks while keeping each listing date together."""
    date_values = data["listing_date"].dt.normalize()
    unique_dates = np.sort(date_values.unique())
    splitter = TimeSeriesSplit(n_splits=CV_SPLITS)

    for fold, (train_indices, test_indices) in enumerate(splitter.split(unique_dates), 1):
        train_dates = unique_dates[train_indices]
        test_dates = unique_dates[test_indices]
        training_data = data.loc[date_values.isin(train_dates)]
        evaluation_data = data.loc[date_values.isin(test_dates)]
        evaluation = f"Dataset 1 forward-chaining CV fold {fold}"

        add_median_baseline(
            result_rows,
            evaluation,
            training_data,
            evaluation_data,
            fold,
        )
        for configuration, feature_columns, include_market_context in configurations:
            for model_name, estimator, scale_numeric, log_target in get_experiment_models():
                add_evaluation(
                    result_rows,
                    model_name,
                    estimator,
                    scale_numeric,
                    log_target,
                    configuration,
                    feature_columns,
                    include_market_context,
                    evaluation,
                    training_data,
                    evaluation_data,
                    fold,
                )


def add_shuffled_cv(
    data: pd.DataFrame,
    configurations: list[tuple[str, list[str], bool]],
    result_rows: list[dict],
) -> None:
    """Run shuffled folds while preserving the mix of the two source datasets."""
    splitter = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )

    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(data, data["dataset"]),
        1,
    ):
        training_data = data.iloc[train_indices].copy()
        evaluation_data = data.iloc[test_indices].copy()
        evaluation = f"Combined shuffled CV fold {fold}"

        add_median_baseline(
            result_rows,
            evaluation,
            training_data,
            evaluation_data,
            fold,
        )
        for configuration, feature_columns, include_market_context in configurations:
            for model_name, estimator, scale_numeric, log_target in get_experiment_models():
                add_evaluation(
                    result_rows,
                    model_name,
                    estimator,
                    scale_numeric,
                    log_target,
                    configuration,
                    feature_columns,
                    include_market_context,
                    evaluation,
                    training_data,
                    evaluation_data,
                    fold,
                )


def add_random_split_evaluation(
    data: pd.DataFrame,
    configurations: list[tuple[str, list[str], bool]],
    result_rows: list[dict],
) -> None:
    """Compare models on a reproducible random holdout, stratified by dataset."""
    train_indices, test_indices = train_test_split(
        np.arange(len(data)),
        test_size=0.2,
        random_state=42,
        stratify=data["dataset"],
    )
    training_data = data.iloc[train_indices].copy()
    evaluation_data = data.iloc[test_indices].copy()

    add_median_baseline(
        result_rows,
        RANDOM_SPLIT_EVAL,
        training_data,
        evaluation_data,
    )
    for configuration, feature_columns, include_market_context in configurations:
        for model_name, estimator, scale_numeric, log_target in get_experiment_models():
            add_evaluation(
                result_rows,
                model_name,
                estimator,
                scale_numeric,
                log_target,
                configuration,
                feature_columns,
                include_market_context,
                RANDOM_SPLIT_EVAL,
                training_data,
                evaluation_data,
            )


def build_model_selection_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates by Dataset 1 forward-CV MAE, with other scores for context."""
    model_results = results.loc[results["configuration"].isin([
        "A: vehicle-only",
        "B: vehicle + market context",
    ])]
    keys = ["configuration", "model"]

    validation = model_results.loc[
        model_results["evaluation"] == "Dataset 1 chronological validation",
        keys + ["mae_lkr", "rmse_lkr", "r2"],
    ].rename(columns={
        "mae_lkr": "dataset1_validation_mae_lkr",
        "rmse_lkr": "dataset1_validation_rmse_lkr",
        "r2": "dataset1_validation_r2",
    })
    forward = model_results.loc[
        model_results["evaluation"].str.startswith("Dataset 1 forward-chaining CV fold")
    ].groupby(keys)[["mae_lkr", "rmse_lkr", "r2"]].agg(["mean", "std"])
    forward.columns = [f"forward_cv_{metric}_{stat}" for metric, stat in forward.columns]
    forward = forward.reset_index()
    shuffled = model_results.loc[
        model_results["evaluation"].str.startswith("Combined shuffled CV fold")
    ].groupby(keys)["mae_lkr"].mean().rename("shuffled_cv_mean_mae_lkr").reset_index()
    later_test = model_results.loc[
        model_results["evaluation"] == "Dataset 2 later-period test",
        keys + ["mae_lkr", "rmse_lkr", "r2"],
    ].rename(columns={
        "mae_lkr": "dataset2_test_mae_lkr",
        "rmse_lkr": "dataset2_test_rmse_lkr",
        "r2": "dataset2_test_r2",
    })

    summary = validation.merge(forward, on=keys).merge(shuffled, on=keys).merge(later_test, on=keys)
    summary["dataset1_forward_cv_mae_rank"] = summary[
        "forward_cv_mae_lkr_mean"
    ].rank(method="min")
    summary["dataset2_test_mae_rank"] = summary["dataset2_test_mae_lkr"].rank(method="min")
    return summary.sort_values("forward_cv_mae_lkr_mean")


def main() -> None:
    # Keep the existing model outputs unchanged and save this comparison separately.
    data = pd.read_csv(DATA_PATH)
    data["listing_date"] = pd.to_datetime(data["listing_date"], utc=True)
    train, validation, later_test = split_by_period(data)
    earlier_data = pd.concat([train, validation], ignore_index=True)
    configurations = [
        ("A: vehicle-only", FEATURES, False),
        ("B: vehicle + market context", FEATURES + MARKET_FEATURES, True),
    ]
    result_rows = []
    prediction_rows = []

    # Compare each method on the same historical validation and later test rows.
    add_median_baseline(
        result_rows,
        "Dataset 1 chronological validation",
        train,
        validation,
    )
    add_median_baseline(
        result_rows,
        "Dataset 2 later-period test",
        earlier_data,
        later_test,
    )
    for configuration, feature_columns, include_market_context in configurations:
        for model_name, estimator, scale_numeric, log_target in get_experiment_models():
            add_evaluation(
                result_rows,
                model_name,
                estimator,
                scale_numeric,
                log_target,
                configuration,
                feature_columns,
                include_market_context,
                "Dataset 1 chronological validation",
                train,
                validation,
                prediction_rows=prediction_rows,
            )
            add_evaluation(
                result_rows,
                model_name,
                estimator,
                scale_numeric,
                log_target,
                configuration,
                feature_columns,
                include_market_context,
                "Dataset 2 later-period test",
                earlier_data,
                later_test,
                prediction_rows=prediction_rows,
            )

    # Treat these folds as a diagnostic alongside the separate Dataset 2 test.
    add_forward_chaining_cv(earlier_data, configurations, result_rows)
    add_shuffled_cv(data, configurations, result_rows)
    # This mixed-period random split is secondary to the future-period tests.
    add_random_split_evaluation(data, configurations, result_rows)
    results = pd.DataFrame(result_rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    selection = build_model_selection_summary(results)
    selection.to_csv(SELECTION_PATH, index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(PREDICTIONS_PATH, index=False)

    print("\nResults by evaluation and model:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print(f"\nSaved metrics to {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Saved model selection summary to {SELECTION_PATH.relative_to(ROOT)}")
    print(f"Saved alternative model predictions to {PREDICTIONS_PATH.relative_to(ROOT)}")
    print("Random 80/20 split is stratified by dataset and is a secondary comparison.")
    print("Dataset 2 LKR scores use the adopted x100,000 price conversion.")


if __name__ == "__main__":
    main()
