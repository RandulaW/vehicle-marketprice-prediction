"""Compare vehicle-only models on the period-based evaluation splits."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from model_preprocessing import DATA_PATH, FEATURES, TARGET, build_preprocessor, split_by_period

def make_model(
    estimator,
    scale_numeric: bool,
    include_market_context: bool = False,
) -> Pipeline:
    # Fit preprocessing and the estimator together to prevent data leakage.
    return Pipeline([
        (
            "preprocessing",
            build_preprocessor(
                scale_numeric=scale_numeric,
                include_market_context=include_market_context,
            ),
        ),
        ("regression", estimator),
    ])


def score_model(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    predictions: np.ndarray | None = None,
) -> dict:
    # Report errors in LKR and R-squared on the original target scale.
    if predictions is None:
        predictions = model.predict(features)
    return {
        "mae_lkr": mean_absolute_error(target, predictions),
        "rmse_lkr": np.sqrt(mean_squared_error(target, predictions)),
        "r2": r2_score(target, predictions),
    }


def get_models() -> list[tuple[str, object, bool]]:
    """Return the fixed estimators used in both feature configurations."""
    return [
        ("Linear Regression", LinearRegression(), True),
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
        ),
    ]


def main() -> None:
    # Split the earlier period for development and hold the later period for testing.
    data = pd.read_csv(DATA_PATH, parse_dates=["listing_date"])
    train, validation, later_test = split_by_period(data)
    x_train, y_train = train[FEATURES], train[TARGET]
    earlier_data = pd.concat([train, validation], ignore_index=True)

    # Keep model settings fixed for a fair feature comparison.
    models = get_models()

    # Compare each model on the same validation and later-period test sets.
    result_rows = []
    for name, estimator, scale_numeric in models:
        validation_model = make_model(clone(estimator), scale_numeric)
        validation_model.fit(x_train, y_train)
        validation_scores = score_model(validation_model, validation[FEATURES], validation[TARGET])
        result_rows.append({
            "model": name,
            "evaluation": "Dataset 1 chronological validation",
            **validation_scores,
        })

        later_model = make_model(clone(estimator), scale_numeric)
        later_model.fit(earlier_data[FEATURES], earlier_data[TARGET])
        later_scores = score_model(later_model, later_test[FEATURES], later_test[TARGET])
        result_rows.append({
            "model": name,
            "evaluation": "Dataset 2 later-period test",
            **later_scores,
        })

    results = pd.DataFrame(result_rows)
    print(results.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print("\nRun compare_vehicle_and_market_aware_models.py to save the A/B comparison.")

if __name__ == "__main__":
    main()
