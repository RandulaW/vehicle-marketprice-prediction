"""Review cleaned datasets before integration."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cleaned"

def inspect_dataset(path: Path, target: str) -> pd.DataFrame:
    # Read a cleaned file and show its period and target range.
    data = pd.read_csv(path)
    print(f"\n{path.name}: {len(data):,} rows, {len(data.columns)} columns")
    if "listing_date" in data.columns:
        dates = pd.to_datetime(data["listing_date"], errors="coerce")
        print(f"Listing dates: {dates.min()} to {dates.max()}")
    if target in data.columns:
        print(f"{target} summary (source scale):")
        print(data[target].describe().to_string())
    return data

def main() -> None:
    # Review existing files without changing them.
    d1 = inspect_dataset(DATA_DIR / "df1_clean.csv", "price_lkr")
    d2 = inspect_dataset(DATA_DIR / "df2_clean.csv", "price_original")
    price_fields = {"price_lkr", "price_original"}
    shared = sorted((set(d1.columns) & set(d2.columns)) - price_fields)
    print("\nShared vehicle/listing columns:", ", ".join(shared))

    # Compare ranges and common labels for likely shared model features.
    print("\nNumeric feature ranges (min / median / max):")
    for column in ["year", "engine_cc", "mileage_km"]:
        print(f"{column}: D1 {d1[column].min():g} / {d1[column].median():g} / {d1[column].max():g}; "
              f"D2 {d2[column].min():g} / {d2[column].median():g} / {d2[column].max():g}")

    # Review Dataset 2 mileage above Dataset 1's 600,000 km cutoff.
    high_mileage = d2.loc[d2["mileage_km"] > 600_000]
    print(f"\nDataset 2 records above 600,000 km: {len(high_mileage)}")
    print(high_mileage[["make", "model", "year", "mileage_km", "price_original"]]
          .sort_values("mileage_km", ascending=False).head(10).to_string(index=False))

    # Check how many category labels can be matched after simple normalization.
    print("\nCategorical label overlap (case and whitespace normalized):")
    for column in ["make", "transmission", "fuel_type", "condition", "location"]:
        labels1 = set(d1[column].dropna().astype(str).str.strip().str.casefold())
        labels2 = set(d2[column].dropna().astype(str).str.strip().str.casefold())
        print(f"{column}: {len(labels1 & labels2)} shared of {len(labels1)} D1 and {len(labels2)} D2 labels")

    # Confirm the body-type text and show the adopted price assumption.
    print("Dataset 1 body-type label intact:", "Coupé/Sports" in set(d1["body_type"].dropna()))
    # Keep the adopted Dataset 2 price assumption visible in the review.
    print("Dataset 2 price_lkr uses the adopted x100,000 assumption; the source unit is unverified.")

if __name__ == "__main__":
    main()
