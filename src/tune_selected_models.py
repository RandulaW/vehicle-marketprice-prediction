"""Tune the selected market-aware models on Dataset 1 time folds."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from model_preprocessing import FEATURES, MARKET_FEATURES, TARGET
from price_prediction import make_model, score_model


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "vehicle_market_model_data.csv"
RESULTS_PATH = ROOT / "outputs" / "results" / "selected_model_tuning.csv"
LATER_TEST_PATH = ROOT / "outputs" / "results" / "selected_model_later_test.csv"
CV_SPLITS = 5


def get_candidates() -> list[tuple[str, str, object]]:
    """Vary one complexity setting while keeping each model's other settings fixed."""
    return [
        (
            "Random Forest",
            "leaf 2 (current)",
            RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "Random Forest",
            "leaf 5",
            RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "Random Forest",
            "leaf 10",
            RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "XGBoost",
            "depth 3",
            XGBRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            ),
        ),
        (
            "XGBoost",
            "depth 6 (current)",
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
        ),
        (
            "XGBoost",
            "depth 8",
            XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            ),
        ),
    ]


def main() -> None:
    # Keep Dataset 2 separate and split Dataset 1 by whole listing dates.
    all_data = pd.read_csv(DATA_PATH, parse_dates=["listing_date"])
    data = all_data.loc[all_data["market_period"] == "earlier"].copy()
    later_test = all_data.loc[all_data["market_period"] == "later"].copy()
    date_values = data["listing_date"].dt.normalize()
    unique_dates = np.sort(date_values.dropna().unique())
    splitter = TimeSeriesSplit(n_splits=CV_SPLITS)
    feature_columns = FEATURES + MARKET_FEATURES
    result_rows = []

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(unique_dates),
        1,
    ):
        training_dates = unique_dates[train_indices]
        validation_dates = unique_dates[validation_indices]
        training_data = data.loc[date_values.isin(training_dates)]
        validation_data = data.loc[date_values.isin(validation_dates)]

        for model_name, setting, estimator in get_candidates():
            print(f"Fold {fold}/{CV_SPLITS}: {model_name}, {setting}", flush=True)
            model = make_model(
                estimator,
                scale_numeric=False,
                include_market_context=True,
            )
            model.fit(training_data[feature_columns], training_data[TARGET])
            predictions = model.predict(validation_data[feature_columns])
            scores = score_model(
                model,
                validation_data[feature_columns],
                validation_data[TARGET],
                predictions,
            )
            result_rows.append({
                "model": model_name,
                "setting": setting,
                "fold": fold,
                "training_rows": len(training_data),
                "validation_rows": len(validation_data),
                "validation_start": validation_data["listing_date"].min().date(),
                "validation_end": validation_data["listing_date"].max().date(),
                **scores,
            })

    results = pd.DataFrame(result_rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    summary = results.groupby(["model", "setting"])[
        ["mae_lkr", "rmse_lkr", "r2"]
    ].agg(["mean", "std"])
    best_model_name, best_setting = summary.sort_values(
        ("mae_lkr", "mean")
    ).index[0]
    best_estimator = next(
        estimator
        for model_name, setting, estimator in get_candidates()
        if model_name == best_model_name and setting == best_setting
    )
    selected_model = make_model(
        best_estimator,
        scale_numeric=False,
        include_market_context=True,
    )
    selected_model.fit(data[feature_columns], data[TARGET])
    later_predictions = selected_model.predict(later_test[feature_columns])
    later_scores = score_model(
        selected_model,
        later_test[feature_columns],
        later_test[TARGET],
        later_predictions,
    )
    later_summary = pd.DataFrame([{
        "model": best_model_name,
        "setting": best_setting,
        "training_rows": len(data),
        "test_rows": len(later_test),
        "test_start": later_test["listing_date"].min().date(),
        "test_end": later_test["listing_date"].max().date(),
        **later_scores,
    }])

    print("\nFive-fold Dataset 1 summary (ranked by mean MAE):")
    print(
        summary.sort_values(("mae_lkr", "mean"))
        .to_string(float_format=lambda value: f"{value:,.4f}")
    )
    print(f"\nSaved fold results to {RESULTS_PATH.relative_to(ROOT)}")
    print(f"\nSelected Dataset 1 CV candidate tested on Dataset 2:")
    print(later_summary.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    later_summary.to_csv(LATER_TEST_PATH, index=False)
    print(f"Saved later-period check to {LATER_TEST_PATH.relative_to(ROOT)}")
    print("Dataset 2 was not used to tune or choose the hyperparameter setting.")


if __name__ == "__main__":
    main()
