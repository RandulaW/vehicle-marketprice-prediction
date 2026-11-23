"""Create time-based splits and training-only preprocessing pipelines."""

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "common_vehicle_model_data.csv"
# Age, engine size, and mileage describe vehicle age, specification, and use.
NUMERIC_FEATURES = ["vehicle_age_years", "engine_cc", "mileage_km"]
# These common vehicle categories capture model and listing differences.
CATEGORICAL_FEATURES = ["make", "model", "transmission", "fuel_type", "condition"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MARKET_NUMERIC_FEATURES = ["usd_lkr_monthly_average"]
MARKET_CATEGORICAL_FEATURES = ["import_policy_environment"]
MARKET_FEATURES = MARKET_NUMERIC_FEATURES + MARKET_CATEGORICAL_FEATURES
TARGET = "price_lkr"


def split_by_period(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use the last fifth of Dataset 1 for validation and Dataset 2 for later testing."""
    earlier = data.loc[data["market_period"] == "earlier"].copy()
    later = data.loc[data["market_period"] == "later"].copy()
    # Split on whole dates so listings from one day stay together.
    dates = sorted(earlier["listing_date"].dt.normalize().dropna().unique())

    if len(dates) < 2:
        raise ValueError("Dataset 1 needs at least two listing dates for a time split.")

    cutoff = dates[max(1, min(len(dates) - 1, int(len(dates) * 0.8)))]
    train = earlier.loc[earlier["listing_date"].dt.normalize() < cutoff].copy()
    validation = earlier.loc[earlier["listing_date"].dt.normalize() >= cutoff].copy()
    return train, validation, later


def build_preprocessor(
    scale_numeric: bool = False,
    include_market_context: bool = False,
) -> ColumnTransformer:
    """Build unfitted imputers and encoders for a model pipeline."""
    # Fill numeric gaps and optionally standardize for distance/linear methods.
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    # Fill missing labels and ignore categories absent from the training data.
    category_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = [
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", category_pipeline, CATEGORICAL_FEATURES),
    ]

    # Add the context variables as separate groups for Configuration B.
    if include_market_context:
        market_numeric_pipeline = Pipeline(numeric_steps.copy())
        transformers.extend([
            ("market_numeric", market_numeric_pipeline, MARKET_NUMERIC_FEATURES),
            ("market_categorical", category_pipeline, MARKET_CATEGORICAL_FEATURES),
        ])

    return ColumnTransformer(transformers)


def main() -> None:
    # Keep all Dataset 2 rows as the later-period test set.
    data = pd.read_csv(DATA_PATH, parse_dates=["listing_date"])
    train, validation, later_test = split_by_period(data)
    cutoff = validation["listing_date"].min().date()

    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,} (from {cutoff})")
    print(f"Later-period test rows: {len(later_test):,}")
    print("Features:", ", ".join(FEATURES))
    print("Imputers and encoders are defined; they fit only on training data.")


if __name__ == "__main__":
    main()
