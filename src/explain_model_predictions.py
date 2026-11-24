"""Summarize how the selected models form Dataset 2 price estimates."""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from model_preprocessing import FEATURES, MARKET_FEATURES, TARGET
from price_prediction import make_model


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "vehicle_market_model_data.csv"
CANDIDATE_PATH = ROOT / "outputs" / "results" / "price_anomaly_candidates.csv"
GLOBAL_PATH = ROOT / "outputs" / "results" / "shap_global_importance.csv"
LOCAL_PATH = ROOT / "outputs" / "results" / "shap_local_explanations.csv"
GLOBAL_SAMPLE_SIZE = 50
LOCAL_CASES_PER_TAIL = 1
RANDOM_STATE = 42
BACKGROUND_SIZE = 20


def get_models() -> list[tuple[str, object]]:
    """Use the current later-test leader and the tuned challenger."""
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


def source_feature(encoded_name: str) -> str:
    """Map one-hot encoded columns back to their original input field."""
    name = encoded_name.split("__", 1)[-1]
    for feature in FEATURES + MARKET_FEATURES:
        if name == feature or name.startswith(f"{feature}_"):
            return feature
    return name


def main() -> None:
    # Summarize Dataset 2 and explain a few flagged listings.
    data = pd.read_csv(DATA_PATH, parse_dates=["listing_date"])
    earlier = data.loc[data["market_period"] == "earlier"].copy()
    later = data.loc[data["market_period"] == "later"].copy()
    candidates = pd.read_csv(CANDIDATE_PATH)
    features = FEATURES + MARKET_FEATURES
    global_rows = []
    local_rows = []

    for model_name, estimator in get_models():
        print(f"Fitting {model_name} on Dataset 1 for SHAP", flush=True)
        model = make_model(estimator, scale_numeric=False, include_market_context=True)
        model.fit(earlier[features], earlier[TARGET])
        model_candidates = candidates.loc[
            (candidates["evaluation"] == "Dataset 2 later-period test")
            & (candidates["prediction_model"] == model_name)
        ].reset_index(drop=True)

        # Explain the largest positive and negative residuals.
        high_positions = model_candidates.nlargest(
            LOCAL_CASES_PER_TAIL, "relative_deviation"
        ).index
        low_positions = model_candidates.nsmallest(
            LOCAL_CASES_PER_TAIL, "relative_deviation"
        ).index
        local_positions = list(dict.fromkeys([*high_positions, *low_positions]))
        local_row_ids = model_candidates.iloc[local_positions]["row_id"].astype(int)
        local_data = data.loc[local_row_ids]
        sample_data = later.sample(
            min(GLOBAL_SAMPLE_SIZE, len(later)),
            random_state=RANDOM_STATE,
        )
        explanation_data = pd.concat([sample_data, local_data])

        transformed = model.named_steps["preprocessing"].transform(
            explanation_data[features]
        )
        background = model.named_steps["preprocessing"].transform(
            earlier[features].sample(
                min(BACKGROUND_SIZE, len(earlier)),
                random_state=RANDOM_STATE,
            )
        )
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        if hasattr(background, "toarray"):
            background = background.toarray()
        feature_names = model.named_steps["preprocessing"].get_feature_names_out()
        print(f"Calculating SHAP values for {len(explanation_data)} listings", flush=True)
        explainer = shap.TreeExplainer(
            model.named_steps["regression"],
            data=background,
            feature_perturbation="interventional",
        )
        explanation = explainer.shap_values(transformed)
        if isinstance(explanation, list):
            explanation = explanation[0]
        explanation = np.asarray(explanation)
        predictions = model.named_steps["regression"].predict(transformed)
        additivity_error = np.max(
            np.abs(explainer.expected_value + explanation.sum(axis=1) - predictions)
        )
        print(f"Largest SHAP additivity difference: {additivity_error:,.2f} LKR", flush=True)
        grouped_names = [source_feature(name) for name in feature_names]
        explanation_rows = pd.DataFrame(explanation, columns=grouped_names)
        grouped = explanation_rows.T.groupby(level=0).sum().T

        # Use a random sample for the feature summary.
        global_values = grouped.iloc[:len(sample_data)].abs().mean().sort_values(ascending=False)
        global_rows.extend({
            "prediction_model": model_name,
            "feature": feature,
            "mean_absolute_shap_lkr": value,
            "explained_rows": len(sample_data),
        } for feature, value in global_values.items())

        # Save each selected listing's signed feature contributions.
        local_start = len(sample_data)
        for local_index, (_, candidate) in enumerate(
            model_candidates.iloc[local_positions].iterrows()
        ):
            row_values = grouped.iloc[local_start + local_index]
            for feature, value in row_values.items():
                local_rows.append({
                    "prediction_model": model_name,
                    "price_flag": "high" if candidate["relative_deviation"] > 0 else "low",
                    "listing_date": candidate["listing_date"],
                    "make": candidate["make"],
                    "vehicle_model": candidate["vehicle_model"],
                    "year": candidate["year"],
                    "actual_lkr": candidate[TARGET],
                    "predicted_lkr": candidate["predicted_lkr"],
                    "relative_deviation": candidate["relative_deviation"],
                    "feature": feature,
                    "shap_contribution_lkr": value,
                })

    pd.DataFrame(global_rows).to_csv(GLOBAL_PATH, index=False)
    pd.DataFrame(local_rows).to_csv(LOCAL_PATH, index=False)
    print(f"Saved global SHAP summaries to {GLOBAL_PATH.relative_to(ROOT)}")
    print(f"Saved local explanations to {LOCAL_PATH.relative_to(ROOT)}")
    print("SHAP values describe model contributions, not causes of a listing price.")


if __name__ == "__main__":
    main()
