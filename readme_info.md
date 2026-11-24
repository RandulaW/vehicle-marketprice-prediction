# MSc Research Project Handover

**Last updated:** 24 November 2026\
**Research topic:** Explainable Machine Learning for Detecting Abnormal
Used-Vehicle Prices in the Sri Lankan Market\
**Purpose:** Working handover for Codex / VS Code and future ChatGPT
sessions.

## 1. Research objective

Estimate expected asking prices for used vehicles in Sri Lanka and
identify listings with unusually high or low prices relative to those
expectations. An abnormal price is a deviation from a modelled
expectation, **not evidence of fraud**.

## 2. Proposed methodology

### Stage 1 --- Expected asking-price modelling

Evaluate: - Linear Regression baseline - Random Forest - XGBoost or
LightGBM

Compare: - **Configuration A:** Vehicle/listing characteristics only. -
**Configuration B:** Vehicle characteristics plus selected
market-context variables.

Potential vehicle features include make, model, manufacturing year,
engine capacity, mileage, transmission, fuel type, condition, body type
and location, subject to availability and quality.

Potential market-context variables: - Monthly average USD/LKR exchange
rate from the Central Bank of Sri Lanka. - Monthly vehicle-transfer
activity from the Department of Motor Traffic, if compatible data are
available. - Categorical indicator for the prevailing
vehicle-import-policy environment.

Evaluate price models using MAE, RMSE and R².

### Stage 2 --- Abnormal asking-price detection

-   Compare actual asking prices with model-estimated expected prices.
-   Calculate residuals and/or relative deviations.
-   Calibrate abnormality thresholds using the model-error distribution.
-   Consider unusually high and unusually low deviations.
-   Apply an unsupervised method such as Local Outlier Factor (LOF) to
    identify unusual listings based on their characteristics.
-   Compare residual-based flags with LOF results. Listings flagged by
    both can be prioritised for review, but are not confirmed fraud.

Precision, recall and F1 require suitable labelled evaluation data.
Manual review alone cannot establish recall/F1 without labels for
flagged and unflagged records.

### Explainability

Use SHAP to explain model predictions and selected flagged listings.
SHAP explains predictions; it is not an anomaly-detection method.

### Cross-period considerations

The two datasets cover separated periods, not a continuous time series.
Market variables should characterise the relevant listing periods. Avoid
unsupported continuous-trend or causal claims.

## 3. Datasets

### Dataset 1 --- Sri Lanka Vehicle Ads Dataset

Source:
https://www.kaggle.com/datasets/ivantha/sri-lanka-vehicle-ads-dataset\
Original file: `vehicles.csv`\
Initial size: 258,888 rows × 26 columns.\
Approximate listing-date range: 2022-09-06 to 2023-04-02.\
Cars category: 48,293 records.

The original `properties` column contains dictionary-like vehicle
attributes, including Condition, Brand, Model, Mileage, Engine capacity,
Year of Manufacture, Transmission, Fuel type, Body type and Model year.

### Dataset 2 --- Sri Lankan Second Vehicle/Car Price Dataset

Source:
https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset\
Original file: `car_price_dataset.csv`\
Initial size: 9,788 rows × 17 columns.\
Approximate listing-date range: 2024-12-03 to 2025-02-05.

Original fields include Brand, Model, YOM, Engine (cc), Gear, Fuel Type,
Millage(KM), Town, Date, Leasing, Condition, equipment fields and Price.

**Dataset 2 price scale:** The project uses `price_lkr = price_original x
100,000` as its adopted working conversion, based on the observed price
values and Sri Lankan listing ranges reviewed during the project.
`price_original` is preserved. The source's unit label has not been
independently documented, so Dataset 2 results depend on this conversion.

## 4. Work completed so far

The latest notebook contains initial cleaning and descriptive analysis
for both datasets. Treat the latest notebook and its saved outputs as
the primary record of exact cell-level progress. Do not repeat completed
work without checking the notebook first.

### Dataset 1 --- cleaning decisions and results

-   Selected the Cars category.
-   Extracted vehicle attributes from nested `properties` using
    corrected index-aligned extraction.
-   Removed 203 records with incomplete make/model/year details, mostly
    wanted/request advertisements with empty properties.
-   Excluded `Other brand` records as a scope decision. This may exclude
    genuine passenger vehicles and should be acknowledged as a
    limitation / possible selection bias.
-   Set engine capacity below 400 cc or above 6,000 cc to missing rather
    than deleting listings (357 affected values).
-   Set mileage above 600,000 km to missing rather than deleting
    listings (96 affected values).
-   Removed 14 records with missing or non-usable prices because price
    is the modelling target.
-   Removed 7 clearly implausible prices: LKR 1, 2, 84, 1,234, 2,150, 1
    billion and 1.98 billion.
-   Retained other unusual but potentially genuine price observations.
    Do not automatically remove statistical price outliers; they are
    central to this research.
-   Set four manufacturing-year values later than their listing dates to
    missing.
-   Corrected the body-type text encoding issue for `Coupé/Sports`.
-   Kept rare categories and missing body types unless a justified
    decision was made.
-   Checked exact duplicate records.

Documented cleaned Dataset 1 size: **47,958 rows**. Previously reported
validation: - `price_lkr`: no missing values; positive values only. -
`engine_cc`: 357 missing; remaining range 400--6,000. - `mileage_km`: 96
missing; remaining range 0--600,000. - `year`: four future-year values
set to missing; no future manufacturing years remain. - Price range
after exclusions: LKR 38,500--249,000,000. - Median year: 2012; median
engine capacity: 1,500 cc; median mileage: 98,000 km; median price: LKR
5,100,000.

### Dataset 2 --- initial cleaning and analysis

Original columns were renamed to consistent names such as `make`,
`model`, `year`, `engine_cc`, `transmission`, `fuel_type`, `mileage_km`,
`location`, `listing_date`, `condition` and `price_original`.

Numeric fields and dates were converted to appropriate types, and
categorical strings were stripped.

Previously reported checks: - 9,788 records. - No missing values in the
checked fields / final cleaned columns. - No non-positive original
prices. - No future manufacturing years. - No negative mileage. - No
exact duplicate rows. - Some records share selected vehicle details but
may represent separate advertisements; do not remove them solely for
similarity. - `Unnamed: 0` is an original index column and should not be
used as a model feature. - Rare categories such as Electric and NEW were
retained.

`price_lkr` uses the adopted x100,000 working conversion. `price_original`
remains unchanged in the cleaned dataset. Model results for Dataset 2 use
`price_lkr` and should be interpreted under this conversion.

### Exploratory data analysis completed

The notebook includes descriptive analysis for both datasets: -
Numerical distributions / histograms. - Categorical frequency plots. -
Asking price against manufacturing year and mileage. - Spearman
correlation heatmaps. - Linear regression trend lines and LOWESS
smoothers.

Dataset 2 price analysis uses the adopted `price_lkr` conversion; retain
the source scale in `price_original` for reference.

The LOWESS warning encountered for Dataset 2 was addressed by grouping
by each feature and taking the median price per distinct feature value
before smoothing.

## 5. Current status and next work

**Completed:** Cleaning and EDA; compatibility review; vehicle-only and
market-aware datasets; leakage-safe preprocessing; model comparison and
time-based validation; a small tuning comparison; feature importance and
collinearity checks; residual flags, LOF comparison and SHAP summaries.

**Current model results:** Random Forest with market context has the
lowest MAE on the Dataset 2 later-period test (about LKR 1.61 million,
R-squared 0.54). Log-target Linear Regression has the lowest mean MAE
across Dataset 1 forward-chaining folds (about LKR 1.23 million).
Performance depends on the evaluation period; no one model leads every
comparison.

**Next:** Review flagged listings as candidates for investigation, then
write up the methods, results and limitations. Dataset 2 scores use the
adopted x100,000 price conversion. DMT transfer counts were not included
because the required December 2024 value was unavailable.

### Sample prediction demo

Run `python src/sample_vehicle_check.py` to enter a listing month, vehicle
details and asking price. The script looks up the exchange rate and policy
label from `market_context_monthly.csv`, then reports price and LOF flags
and SHAP contributions. It accepts only months in that file. Dataset 1
contains one policy label and a higher exchange-rate range than later
months, so later-period estimates have those limitations. Category labels
are matched without regard to capitalization; unseen labels may be ignored.

For the form-based demo, install requirements and run
`streamlit run src/sample_vehicle_app.py`. The fitted model is cached while
the app is open so it is not retrained for every form submission.

## 6. Important research cautions

-   Do not delete high/low prices solely because they are statistical
    outliers; abnormal prices are the research focus.
-   Prefer setting questionable feature values to missing rather than
    deleting otherwise useful listings when appropriate.
-   Document all exclusions and cleaning rules.
-   `Other brand` exclusion is a scope choice and possible source of
    selection bias.
-   Engine and mileage thresholds are pragmatic cleaning rules, not
    universal truths.
-   Dataset periods are separated, not a continuous time series.
-   Dataset 2 price unit remains unverified.
-   Abnormal price does not establish fraud.
-   SHAP explains predictions; it does not detect anomalies.
-   Precision/recall/F1 require suitable labels.
-   Do not claim causal effects from descriptive market-period
    comparisons.

## 7. Recommended project organisation

``` text
vehicle-price-research/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── data_cleaning_&_EDA.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── data_integration.py
│   ├── market_factors.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── price_prediction.py
│   ├── anomaly_detection.py
│   └── explainability.py
├── outputs/
│   ├── figures/
│   └── results/
├── README.md
├── AGENTS.md
└── requirements.txt
```

This is a proposed structure, not a requirement to create every file
immediately. Build incrementally.

## 8. Collaboration preferences

-   Keep explanations short and practical.
-   Provide **one code cell or implementation step at a time** and wait
    for the user to share the output before proceeding.
-   Include a short natural comment at the beginning of each code cell.
-   Add Markdown headings when starting a new main section or meaningful
    subsection, but do not add headings unnecessarily.
-   Use graphs when they materially help inspect distributions or
    explain results.
-   Preserve existing notebook work and section numbering.
-   Do not repeat completed work.
-   Do not make major changes to the notebook or data without explaining
    the proposed change first.
-   Do not silently invent data, assumptions, research results or source
    facts.

## 9. Instructions for Codex

Before making changes: 1. Inspect the repository, notebook, datasets and
this handover. 2. Read `AGENTS.md`. 3. Summarise the current project
state and identify inconsistencies between this handover and the
notebook. 4. Do not overwrite or restructure the existing notebook
without approval. 5. Do not infer or silently convert Dataset 2 prices.
Keep original values and ask for confirmation / evidence before using a
conversion. 6. Work incrementally, with reproducible scripts and clear
input/output paths. 7. Keep raw data unchanged; write cleaned outputs to
`data/processed/`. 8. Avoid committing large datasets or sensitive
material to a public GitHub repository. Use an appropriate private
repository or Git LFS / external storage where needed. 9. Do not scrape
marketplace data unless the user confirms permission and the site's
terms allow the intended collection and research use. 10. Explain each
proposed step briefly and wait for approval before major implementation
changes.

## 10. Initial prompt for Codex

> Read `AGENTS.md`, this handover document, and inspect the existing
> notebook and datasets. Do not modify anything yet. First report the
> current repository structure, notebook progress, available data files,
> discrepancies between the notebook and this handover, and the safest
> next step. Keep the report concise. Do not repeat completed cleaning
> or EDA. Do not assume Dataset 2's price unit or convert it without
> verification.
