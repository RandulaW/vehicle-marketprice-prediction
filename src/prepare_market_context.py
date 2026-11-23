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
    ("2025-01", 296.2, 12716, "reopening_transition"),
    ("2025-02", 296.8, 12649, "reopened_under_controls"),
]

CBSL_SOURCE = (
    "https://www.cbsl.gov.lk/en/node/146; "
    "https://research.dulajmendis.com/data/exchange-rates/us-dollar-usd/"
)
DMT_SOURCES = {
    "2022": "https://dmt.gov.lk/images/2023/transfer/Monthly_Transfer_of_Vehicles_2022.pdf",
    "2023": "https://dmt.gov.lk/images/2023/transfer/Monthly_Transfer_of_Vehicles_2023.pdf",
    "2025": "https://dmt.gov.lk/images/2026/Monthly_Transfer_of_Vehicles_2025.pdf",
}
POLICY_SOURCE = (
    "https://documents.gov.lk/view/extra-gazettes/2024/12/2415-35_E.pdf; "
    "https://documents.gov.lk/view/extra-gazettes/2025/1/2421-04_E.pdf; "
    "https://documents.gov.lk/view/extra-gazettes/2025/1/2421-44_E.pdf"
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
            "transfer_count_source": source_for_transfers(month),
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
