"""Compare vehicle-only and market-aware price models on matching splits."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from model_preprocessing import (
    FEATURES,
    MARKET_FEATURES,
    TARGET,
    split_by_period,
)
from price_prediction import get_models, make_model, score_model


ROOT = Path(__file__).resolve().parents[1]
MARKET_DATA_PATH = ROOT / "data" / "processed" / "vehicle_market_model_data.csv"
PREDICTIONS_PATH = ROOT / "outputs" / "results" / "vehicle_market_price_predictions.csv"


def make_prediction_rows(
    configuration: str,
    model_name: str,
    evaluation: str,
    data: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Keep row-level predictions for diagnostics and anomaly analysis."""
    return pd.DataFrame({
        "configuration": configuration,
        "model": model_name,
        "evaluation": evaluation,
        "listing_date": data["listing_date"].dt.strftime("%Y-%m-%d"),
        "dataset": data["dataset"].to_numpy(),
        "actual_lkr": data[TARGET].to_numpy(),
        "predicted_lkr": predictions,
    })


def main() -> None:
    # Use the joined file for both configurations so their rows and splits match.
    data = pd.read_csv(MARKET_DATA_PATH, parse_dates=["listing_date"])
    train, validation, later_test = split_by_period(data)
    training_period = pd.concat([train, validation], ignore_index=True)

    configurations = [
        ("A: vehicle-only", FEATURES, False),
        ("B: vehicle + market context", FEATURES + MARKET_FEATURES, True),
    ]
    result_rows = []
    prediction_frames = []

    # Reuse each fixed estimator and split for a direct feature comparison.
    for configuration, feature_columns, include_market_context in configurations:
        for model_name, estimator, scale_numeric in get_models():
            print(
                f"Fitting {configuration} / {model_name}: Dataset 1 validation",
                flush=True,
            )
            validation_model = make_model(
                clone(estimator),
                scale_numeric,
                include_market_context,
            )
            validation_model.fit(train[feature_columns], train[TARGET])
            validation_predictions = validation_model.predict(validation[feature_columns])
            validation_scores = score_model(
                validation_model,
                validation[feature_columns],
                validation[TARGET],
                validation_predictions,
            )
            prediction_frames.append(make_prediction_rows(
                configuration,
                model_name,
                "Dataset 1 chronological validation",
                validation,
                validation_predictions,
            ))
            result_rows.append({
                "configuration": configuration,
                "model": model_name,
                "evaluation": "Dataset 1 chronological validation",
                **validation_scores,
            })

            print(
                f"Fitting {configuration} / {model_name}: Dataset 2 later-period test",
                flush=True,
            )
            later_model = make_model(
                clone(estimator),
                scale_numeric,
                include_market_context,
            )
            later_model.fit(training_period[feature_columns], training_period[TARGET])
            later_predictions = later_model.predict(later_test[feature_columns])
            later_scores = score_model(
                later_model,
                later_test[feature_columns],
                later_test[TARGET],
                later_predictions,
            )
            prediction_frames.append(make_prediction_rows(
                configuration,
                model_name,
                "Dataset 2 later-period test",
                later_test,
                later_predictions,
            ))
            result_rows.append({
                "configuration": configuration,
                "model": model_name,
                "evaluation": "Dataset 2 later-period test",
                **later_scores,
            })

    # Keep row-level predictions for plots; aggregate scores are in the full experiment table.
    results = pd.DataFrame(result_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    print(results.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print(f"Saved row-level predictions to {PREDICTIONS_PATH.relative_to(ROOT)}")
    unseen_policy = sorted(
        set(later_test["import_policy_environment"].dropna())
        - set(training_period["import_policy_environment"].dropna())
    )
    if unseen_policy:
        print("Policy categories unseen during training:", ", ".join(unseen_policy))
    print("Dataset 2 LKR metrics depend on the adopted x100,000 price conversion.")
    print("DMT transfers are excluded because December 2024 is missing.")


if __name__ == "__main__":
    main()
