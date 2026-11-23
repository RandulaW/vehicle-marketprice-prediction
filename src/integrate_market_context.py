"""Join monthly market context to the prepared vehicle listings."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VEHICLE_PATH = ROOT / "data" / "processed" / "common_vehicle_model_data.csv"
MARKET_PATH = ROOT / "data" / "processed" / "market_context_monthly.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "vehicle_market_model_data.csv"


def main() -> None:
    # Keep the original vehicle-only modelling table unchanged.
    vehicles = pd.read_csv(VEHICLE_PATH, parse_dates=["listing_date"])
    market = pd.read_csv(MARKET_PATH)
    vehicles["listing_month"] = vehicles["listing_date"].dt.strftime("%Y-%m")

    if market["listing_month"].duplicated().any():
        raise ValueError("Market context must contain one row per month.")

    # Join on listing month; retain the DMT field even where its value is unavailable.
    market_features = market[[
        "listing_month",
        "usd_lkr_monthly_average",
        "dmt_car_transfers",
        "import_policy_environment",
    ]]
    combined = vehicles.merge(market_features, on="listing_month", how="left", validate="many_to_one")

    missing_months = combined.loc[
        combined["usd_lkr_monthly_average"].isna(), "listing_month"
    ].drop_duplicates()
    if not missing_months.empty:
        raise ValueError(f"No market context found for listing months: {missing_months.tolist()}")

    # Preserve the market-aware copy separately from the vehicle-only table.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(combined):,} rows to {OUTPUT_PATH.relative_to(ROOT)}")
    print("Rows by source:")
    print(combined.groupby("dataset").size().to_string())
    print("Missing DMT transfer counts:", combined["dmt_car_transfers"].isna().sum())
    print("Market-aware model features: USD/LKR average and import policy environment.")
    print("DMT transfers are retained for reference but excluded from model features until coverage is complete.")


if __name__ == "__main__":
    main()
