"""Check model feature importance and numeric collinearity."""

from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression

from model_preprocessing import (
    FEATURES,
    MARKET_FEATURES,
    NUMERIC_FEATURES,
    MARKET_NUMERIC_FEATURES,
    TARGET,
    split_by_period,
)
from price_prediction import get_models, make_model


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "vehicle_market_model_data.csv"
IMPORTANCE_PATH = ROOT / "outputs" / "results" / "feature_permutation_importance.csv"
COLLINEARITY_PATH = ROOT / "outputs" / "results" / "feature_collinearity.csv"
IMPORTANCE_MODELS = {"XGBoost"}
CONFIGURATIONS = [
    ("A: vehicle-only", FEATURES, False),
    ("B: vehicle + market context", FEATURES + MARKET_FEATURES, True),
]


def calculate_numeric_diagnostics(training_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize numeric feature correlations and variance inflation."""
    numeric_features = NUMERIC_FEATURES + MARKET_NUMERIC_FEATURES
    numeric_data = training_data[numeric_features]
    correlations = numeric_data.corr().rename_axis("feature").reset_index()
    correlations = correlations.melt(
        id_vars="feature",
        var_name="other_feature",
        value_name="correlation",
    )

    complete_data = numeric_data.dropna()
    vif_rows = []
    for feature in numeric_features:
        other_features = [name for name in numeric_features if name != feature]
        model = LinearRegression().fit(complete_data[other_features], complete_data[feature])
        r_squared = model.score(complete_data[other_features], complete_data[feature])
        vif_rows.append({
            "feature": feature,
            "vif": 1 / max(1 - r_squared, 1e-12),
            "complete_training_rows": len(complete_data),
        })

    return correlations, pd.DataFrame(vif_rows)


def main() -> None:
    # Use Dataset 1 training rows for diagnostics and its later dates for importance.
    data = pd.read_csv(DATA_PATH, parse_dates=["listing_date"])
    training_data, validation_data, _ = split_by_period(data)
    importance_rows = []

    for configuration, feature_columns, include_market_context in CONFIGURATIONS:
        for model_name, estimator, scale_numeric in get_models():
            if model_name not in IMPORTANCE_MODELS:
                continue

            print(f"Measuring {model_name}: {configuration}", flush=True)
            model = make_model(
                estimator,
                scale_numeric,
                include_market_context,
            )
            model.fit(training_data[feature_columns], training_data[TARGET])
            result = permutation_importance(
                model,
                validation_data[feature_columns],
                validation_data[TARGET],
                scoring="neg_mean_absolute_error",
                n_repeats=5,
                random_state=42,
                n_jobs=1,
            )

            for index, feature in enumerate(feature_columns):
                importance_rows.append({
                    "model": model_name,
                    "configuration": configuration,
                    "evaluation": "Dataset 1 chronological validation",
                    "feature": feature,
                    "mae_increase_lkr": result.importances_mean[index],
                    "mae_increase_std_lkr": result.importances_std[index],
                })

    correlations, vif = calculate_numeric_diagnostics(training_data)
    correlations.to_csv(COLLINEARITY_PATH, index=False)
    vif.to_csv(COLLINEARITY_PATH.with_name("numeric_feature_vif.csv"), index=False)
    pd.DataFrame(importance_rows).to_csv(IMPORTANCE_PATH, index=False)

    print("\nPermutation importance: mean increase in validation MAE when shuffled")
    print(
        pd.DataFrame(importance_rows)
        .sort_values(["model", "configuration", "mae_increase_lkr"], ascending=[True, True, False])
        .to_string(index=False, float_format=lambda value: f"{value:,.0f}")
    )
    print("\nNumeric feature VIF:")
    print(vif.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print(f"\nSaved feature results to {IMPORTANCE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
