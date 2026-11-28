"""Prepare sourced monthly market context for the vehicle listing periods."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "processed" / "market_context_monthly.csv"

# Monthly USD/LKR spot averages are shown to one decimal in the CBSL series.
MARKET_CONTEXT = [
    ("2022-09", 362.4, 6966, "suspended"),
    ("2022-10", 363.1, 6658, "suspended"),
    ("2022-11", 363.4, 6925, "suspended"),
    ("2022-12", 363.2, 7098, "suspended"),
    ("2023-01", 362.4, 7701, "suspended"),
    ("2023-02", 362.0, 8045, "suspended"),
    ("2023-03", 329.5, 10042, "suspended"),
    ("2023-04", 321.4, 7271, "suspended"),
    ("2024-12", 291.7, None, "suspension_then_reopening"),
    ("2025-01", 296.2, 12649, "reopening_transition"),
    ("2025-02", 296.8, 12716, "reopened_under_controls"),
    ("2025-03", 295.9, 13752, "reopened_under_controls"),
    ("2025-04", 298.5, 14657, "reopened_under_controls"),
    ("2025-05", 299.4, 14851, "reopened_under_controls"),
    ("2025-06", 299.9, 17393, "reopened_under_controls"),
    ("2025-07", 301.1, 18903, "reopened_under_controls"),
    ("2025-08", 301.5, 17804, "reopened_under_controls"),
    ("2025-09", 302.2, 16773, "reopened_under_controls"),
    ("2025-10", 303.1, 16560, "reopened_under_controls"),
    ("2025-11", 306.3, 14468, "reopened_under_controls"),
    ("2025-12", 309.2, 16496, "reopened_under_controls"),
    ("2026-01", 309.7, None, "reopened_under_controls"),
    ("2026-02", 309.4, None, "reopened_under_controls"),
    ("2026-03", 311.9, None, "reopened_under_controls"),
    ("2026-04", 316.5, None, "reopened_with_revised_duties"),
    ("2026-05", 324.9, None, "reopened_with_vehicle_duty_surcharge"),
    ("2026-06", 334.4, None, "reopened_with_vehicle_duty_surcharge"),
    ("2026-07", 336.0, None, "reopened_with_vehicle_duty_surcharge"),
    ("2026-08", 332.9, None, "reopened_with_surcharge_extended"),
    ("2026-09", 329.5, None, "reopened_with_surcharge_extended"),
]

CBSL_SOURCE = (
    "https://www.cbsl.gov.lk/en/rates-and-indicators/exchange-rates; "
    "https://www.cbsl.gov.lk/sites/default/files/cbslweb_documents/statistics/sheets/IF_Monthly_Average_Exchange_Rates_September%202026.xlsx; "
    "https://macrolab.lk/data/series/LK.EXT.FX_MONTHLY.USD.AVG.M/"
)
DMT_SOURCES = {
    "2022": "https://dmt.gov.lk/images/2023/transfer/Monthly_Transfer_of_Vehicles_2022.pdf",
    "2023": "https://dmt.gov.lk/images/2023/transfer/Monthly_Transfer_of_Vehicles_2023.pdf",
    "2025": "https://dmt.gov.lk/images/2026/Monthly_Transfer_of_Vehicles_2025.pdf",
}
POLICY_SOURCE = (
    "https://documents.gov.lk/view/extra-gazettes/2024/12/2415-35_E.pdf; "
    "https://documents.gov.lk/view/extra-gazettes/2025/1/2421-04_E.pdf; "
    "https://documents.gov.lk/view/extra-gazettes/2025/1/2421-44_E.pdf; "
    "https://documents.gov.lk/view/egz/2026/5/2488-56_E.pdf; "
    "https://treasury.gov.lk/si/api/file/04b45c52-3ded-4621-86c7-b8c67455c9d8; "
    "https://www.treasury.gov.lk/acts-gazettes-circulars"
)


def source_for_transfers(month: str) -> str:
    """Return the DMT report used for a month's car-transfer count."""
    year = month[:4]
    return DMT_SOURCES.get(year, "")


def main() -> None:
    """Write one row per month covered by either vehicle dataset."""
    columns = [
        "listing_month",
        "usd_lkr_monthly_average",
        "dmt_car_transfers",
        "import_policy_environment",
        "exchange_rate_source",
        "transfer_count_source",
        "policy_source",
    ]
    rows = [
        {
            "listing_month": month,
            "usd_lkr_monthly_average": exchange_rate,
            "dmt_car_transfers": transfer_count,
            "import_policy_environment": policy,
            "exchange_rate_source": CBSL_SOURCE,
            "transfer_count_source": (
                source_for_transfers(month) if transfer_count is not None else ""
            ),
            "policy_source": POLICY_SOURCE,
        }
        for month, exchange_rate, transfer_count, policy in MARKET_CONTEXT
    ]

    market_context = pd.DataFrame(rows, columns=columns)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    market_context.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(market_context)} monthly records to {OUTPUT_PATH}")
    print(f"Missing DMT car-transfer counts: {market_context['dmt_car_transfers'].isna().sum()}")


if __name__ == "__main__":
    main()
