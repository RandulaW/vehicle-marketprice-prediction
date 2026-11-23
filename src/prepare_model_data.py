"""Build a shared modelling table from the cleaned vehicle datasets."""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "cleaned"
OUTPUT_PATH = ROOT / "data" / "processed" / "common_vehicle_model_data.csv"

CATEGORY_COLUMNS = ["make", "model", "transmission", "fuel_type", "condition"]
MODEL_COLUMNS = [
    "listing_date",
    "year",
    "engine_cc",
    "mileage_km",
    *CATEGORY_COLUMNS,
    "price_lkr",
]


def prepare_dataset(path: Path, dataset: str, period: str) -> pd.DataFrame:
    """Select common fields and apply the same basic preparation."""
    data = pd.read_csv(path)
    print(f"{dataset} exact duplicate source rows: {data.duplicated().sum()}")

    # Keep the adopted Dataset 2 price conversion visible.
    if dataset == "dataset_2":
        conversion = data["price_original"] * 100_000
        matches = (data["price_lkr"] - conversion).abs() < 1e-6
        print(f"Dataset 2 price conversion consistent: {matches.all()}")

    # Match the mileage rule used when cleaning Dataset 1.
    high_mileage = data["mileage_km"] > 600_000
    if high_mileage.any():
        print(f"{dataset} mileage values set missing above 600,000 km: {high_mileage.sum()}")
        data.loc[high_mileage, "mileage_km"] = pd.NA

    # Standardize dates and select fields shared by both sources.
    data["listing_date"] = pd.to_datetime(data["listing_date"], errors="coerce", utc=True)
    result = data[MODEL_COLUMNS].copy()

    # Derive vehicle age and normalize category text for consistent encoding.
    year = pd.to_numeric(result["year"], errors="coerce")
    result["vehicle_age_years"] = result["listing_date"].dt.year - year
    for column in CATEGORY_COLUMNS:
        result[column] = result[column].astype("string").str.strip().str.casefold()

    result["dataset"] = dataset
    result["market_period"] = period
    return result


def report_quality(data: pd.DataFrame) -> None:
    """Print checks for review without dropping listings."""
    print("\nQA summary:")
    invalid_prices = data["price_lkr"].isna() | (data["price_lkr"] <= 0)
    print("Missing or non-positive target prices:", invalid_prices.sum())

    feature_columns = [
        "year",
        "vehicle_age_years",
        "engine_cc",
        "mileage_km",
        *CATEGORY_COLUMNS,
    ]
    print("Missing model features by source:")
    for source, group in data.groupby("dataset"):
        missing = group[feature_columns].isna().sum().to_dict()
        print(f"{source}: {missing}")

    print("Future manufacturing years:", (data["vehicle_age_years"] < 0).sum())
    print("Negative mileage values:", (data["mileage_km"] < 0).sum())
    high_mileage = (data["dataset"] == "dataset_2") & (data["mileage_km"] > 600_000)
    print("Dataset 2 mileage above 600,000 km remaining:", high_mileage.sum())


def main() -> None:
    # Prepare each dataset, then append rows for period-based evaluation.
    earlier = prepare_dataset(CLEAN_DIR / "df1_clean.csv", "dataset_1", "earlier")
    later = prepare_dataset(CLEAN_DIR / "df2_clean.csv", "dataset_2", "later")
    model_data = pd.concat([earlier, later], ignore_index=True)

    report_quality(model_data)

    # Save a derived table and leave cleaned source files unchanged.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_data.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(model_data):,} rows to {OUTPUT_PATH.relative_to(ROOT)}")
    print(model_data.groupby(["dataset", "market_period"]).size().to_string())
    print("Dataset 2 price_lkr uses the adopted x100,000 assumption.")


if __name__ == "__main__":
    main()
