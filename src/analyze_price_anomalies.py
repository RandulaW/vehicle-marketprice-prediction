"""Flag unusual asking prices and vehicle profiles for review."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from model_preprocessing import FEATURES, MARKET_FEATURES, NUMERIC_FEATURES, TARGET
from price_prediction import make_model


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "vehicle_market_model_data.csv"
RESULT_DIR = ROOT / "outputs" / "results"
CANDIDATE_PATH = RESULT_DIR / "price_anomaly_candidates.csv"
SUMMARY_PATH = RESULT_DIR / "anomaly_detection_summary.csv"
THRESHOLD_PATH = RESULT_DIR / "residual_thresholds.csv"
CV_SPLITS = 5
LOF_NEIGHBORS = 35
LOF_CONTAMINATION = 0.05
TAIL_PERCENTILE = 2.5


def get_models() -> list[tuple[str, object]]:
    """Return the strongest current model and the tuned challenger."""
    return [
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "XGBoost depth 8",
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


def chronological_thresholds(
    data: pd.DataFrame,
    feature_columns: list[str],
    validation_cutoff,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    """Calibrate validation cutoffs without using validation-period residuals."""
    dates = data["listing_date"].dt.normalize()
    unique_dates = np.sort(dates.dropna().unique())
    splitter = TimeSeriesSplit(n_splits=CV_SPLITS)
    training_errors = {name: [] for name, _ in get_models()}
    all_errors = {name: [] for name, _ in get_models()}

    for fold, (train_indices, heldout_indices) in enumerate(splitter.split(unique_dates), 1):
        train_dates = unique_dates[train_indices]
        heldout_dates = unique_dates[heldout_indices]
        training = data.loc[dates.isin(train_dates)]
        heldout = data.loc[dates.isin(heldout_dates)]

        for name, estimator in get_models():
            print(f"Residual calibration fold {fold}/{CV_SPLITS}: {name}", flush=True)
            model = make_model(estimator, scale_numeric=False, include_market_context=True)
            model.fit(training[feature_columns], training[TARGET])
            prediction = np.maximum(model.predict(heldout[feature_columns]), 1.0)
            # Relative errors make the cutoffs comparable across price ranges.
            relative = heldout[TARGET].to_numpy() / prediction - 1
            all_errors[name].extend(relative)
            before_validation = heldout["listing_date"].dt.normalize() < validation_cutoff
            training_errors[name].extend(relative[before_validation.to_numpy()])

    def cutoffs(errors_by_model: dict[str, list[float]]) -> dict[str, tuple[float, float]]:
        return {
            name: (
                float(np.percentile(errors, TAIL_PERCENTILE)),
                float(np.percentile(errors, 100 - TAIL_PERCENTILE)),
            )
            for name, errors in errors_by_model.items()
        }

    return cutoffs(training_errors), cutoffs(all_errors)


def score_lof(training: pd.DataFrame, evaluation: pd.DataFrame) -> np.ndarray:
    """Flag unusual vehicle age, engine size, or mileage without using price."""
    lof = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LocalOutlierFactor(
            n_neighbors=LOF_NEIGHBORS,
            contamination=LOF_CONTAMINATION,
            novelty=True,
        ),
    )
    lof.fit(training[NUMERIC_FEATURES])
    return lof.predict(evaluation[NUMERIC_FEATURES]) == -1


def build_predictions(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    evaluation_name: str,
    feature_columns: list[str],
    thresholds: dict[str, tuple[float, float]],
) -> list[pd.DataFrame]:
    """Score a held-out period and add residual and LOF review flags."""
    lof_flags = score_lof(training, evaluation)
    rows = []
    for name, estimator in get_models():
        model = make_model(estimator, scale_numeric=False, include_market_context=True)
        model.fit(training[feature_columns], training[TARGET])
        predicted = np.maximum(model.predict(evaluation[feature_columns]), 1.0)
        actual = evaluation[TARGET].to_numpy()
        relative = actual / predicted - 1
        low_cutoff, high_cutoff = thresholds[name]

        result = evaluation[[
            "listing_date", "dataset", "make", "model", "year", *NUMERIC_FEATURES, TARGET,
        ]].rename(columns={"model": "vehicle_model"}).copy()
        result["row_id"] = evaluation.index
        result["prediction_model"] = name
        result["evaluation"] = evaluation_name
        result["predicted_lkr"] = predicted
        result["residual_lkr"] = actual - predicted
        result["relative_deviation"] = relative
        result["low_residual_cutoff"] = low_cutoff
        result["high_residual_cutoff"] = high_cutoff
        result["low_price_flag"] = relative < low_cutoff
        result["high_price_flag"] = relative > high_cutoff
        result["lof_vehicle_flag"] = lof_flags
        result["any_price_flag"] = result["low_price_flag"] | result["high_price_flag"]
        result["price_and_vehicle_overlap"] = result["any_price_flag"] & result["lof_vehicle_flag"]
        rows.append(result)
    return rows


def main() -> None:
    # Calibrate with earlier listings, then check a held-out slice and the later dataset.
    data = pd.read_csv(DATA_PATH, parse_dates=["listing_date"])
    data["listing_date"] = pd.to_datetime(data["listing_date"], utc=True)
    earlier = data.loc[data["market_period"] == "earlier"].copy()
    later = data.loc[data["market_period"] == "later"].copy()
    dates = np.sort(earlier["listing_date"].dt.normalize().dropna().unique())
    cutoff = dates[max(1, min(len(dates) - 1, int(len(dates) * 0.8)))]
    training = earlier.loc[earlier["listing_date"].dt.normalize() < cutoff]
    validation = earlier.loc[earlier["listing_date"].dt.normalize() >= cutoff]
    feature_columns = FEATURES + MARKET_FEATURES

    validation_thresholds, later_thresholds = chronological_thresholds(
        earlier,
        feature_columns,
        cutoff,
    )
    threshold_rows = []
    for evaluation, thresholds, calibration in [
        (
            "Dataset 1 chronological validation",
            validation_thresholds,
            "Dataset 1 training-period out-of-fold residuals",
        ),
        (
            "Dataset 2 later-period test",
            later_thresholds,
            "Full Dataset 1 out-of-fold residuals",
        ),
    ]:
        threshold_rows.extend({
            "evaluation": evaluation,
            "model": name,
            "low_relative_deviation": low,
            "high_relative_deviation": high,
            "calibration": calibration,
        } for name, (low, high) in thresholds.items())

    candidate_rows = []
    candidate_rows.extend(build_predictions(
        training, validation, "Dataset 1 chronological validation",
        feature_columns, validation_thresholds,
    ))
    candidate_rows.extend(build_predictions(
        earlier, later, "Dataset 2 later-period test",
        feature_columns, later_thresholds,
    ))
    candidates = pd.concat(candidate_rows, ignore_index=True)
    candidates["listing_date"] = candidates["listing_date"].dt.strftime("%Y-%m-%d")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(CANDIDATE_PATH, index=False)
    pd.DataFrame(threshold_rows).to_csv(THRESHOLD_PATH, index=False)

    summary = candidates.groupby(["evaluation", "prediction_model"]).agg(
        listings=(TARGET, "size"),
        low_price_flags=("low_price_flag", "sum"),
        high_price_flags=("high_price_flag", "sum"),
        unusual_vehicle_profiles=("lof_vehicle_flag", "sum"),
        price_and_vehicle_overlap=("price_and_vehicle_overlap", "sum"),
    ).reset_index()
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nResidual cutoffs (relative to predicted price):")
    print(pd.DataFrame(threshold_rows).to_string(index=False))
    print("\nFlags by evaluation:")
    print(summary.to_string(index=False))
    print(f"\nSaved candidate rows to {CANDIDATE_PATH.relative_to(ROOT)}")
    print("LOF flags unusual vehicle characteristics; price flags identify unusually high or low residuals.")


if __name__ == "__main__":
    main()
