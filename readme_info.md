# Project notes and reproduction guide

**Last reviewed:** 28 November 2026
**Research topic:** Explainable Machine Learning for Detecting Abnormal
Used-Vehicle Prices in the Sri Lankan Market

This file records the project decisions and results needed to continue
the work on another computer. `AGENTS.md` contains the working rules.

## Research aim

Estimate expected asking prices for used vehicles in Sri Lanka and flag
listings that differ substantially from model estimates. A flag is a
candidate for review, not evidence of fraud. SHAP explains model
predictions; it is not an anomaly detector.

## Data sources and price decision

| Dataset | Source file | Original coverage |
|---|---|---|
| Dataset 1: Sri Lanka Vehicle Ads | `vehicles.csv` | 258,888 rows; Cars category has 48,293 listings; listings from 2022-09-06 to 2023-04-02 |
| Dataset 2: Sri Lankan Second Vehicle/Car Price | `car_price_dataset.csv` | 9,788 listings from 2024-12-03 to 2025-02-05 |

Sources: [Dataset 1](https://www.kaggle.com/datasets/ivantha/sri-lanka-vehicle-ads-dataset), [Dataset 2](https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset).

The project uses the user-approved working conversion
`price_lkr = price_original * 100,000` for Dataset 2. This was adopted
after reviewing the row values against plausible Sri Lankan listing
prices. `price_original` remains unchanged. The dataset publisher does
not independently document the unit, so Dataset 2 metrics and findings
are conditional on the adopted conversion.

## Cleaning and EDA completed

The notebook `notebooks/data_cleaning_&_EDA.ipynb` has 77 cells and
covers data loading, inspection, cleaning and exploratory analysis for
both datasets. Its saved outputs are retained.

### Dataset 1

- Selected the Cars category and extracted attributes from `properties`
  with index alignment.
- Removed 203 listings with incomplete make, model or year details;
  most had empty properties and appeared to be wanted/request listings.
  Excluded `Other brand` as a scope decision; this may remove genuine
  passenger vehicles and create selection bias.
- Set engine capacities below 400 cc or above 6,000 cc to missing
  (357 values); set mileage above 600,000 km to missing (96 values).
- Removed 14 listings without usable prices and seven clearly
  implausible prices: LKR 1, 2, 84, 1,234, 2,150, 1 billion and
  1.98 billion.
  Other unusual prices were retained for the anomaly research.
- Set four manufacturing years later than listing dates to missing and
  corrected a body-type text-encoding issue.
- Checked exact duplicates; retained rare categories and missing body
  types unless a specific data-quality reason applied. Body type is
  missing for 4,146 listings.
- Final cleaned data: **47,958 rows, 39 columns**. Price is positive and
  complete; price range is LKR 38,500 to 249,000,000; median is LKR
  5,100,000. Engine capacity has 357 missing values (remaining range
  400-6,000 cc); mileage has 96 missing values (remaining range
  0-600,000 km); four future years were set missing.

### Dataset 2

- Standardized field names and converted dates and numeric columns;
  retained rare categories and did not remove exact or near-duplicate
  listings without justification.
- Checked duplicate vehicle details manually as possible separate
  advertisements; `Unnamed: 0` is an original index, not a model feature.
- Kept `price_original` and created `price_lkr` using the adopted
  conversion above.
- Final cleaned data: **9,788 rows, 18 columns**. No missing prices,
  non-positive prices, future manufacturing years, negative mileage or
  exact duplicate rows were found in the final notebook checks.
- Under the adopted conversion, `price_lkr` ranges from LKR 265,000 to
  LKR 79,000,000, with a median of LKR 4,300,000.
- During model-table preparation, mileage above 600,000 km is set to
  missing (33 Dataset 2 records); those listings are retained.
- Dataset 2 median values are year 2009, engine capacity 1,300 cc,
  mileage 176,000 km and `price_original` 43 (LKR 4,300,000 after the
  adopted conversion).

EDA includes numerical and categorical distributions, price against
year and mileage, Spearman correlations, linear trend lines and LOWESS
smoothers. Dataset 2 smoothing groups by feature value and uses median
price to avoid the earlier LOWESS warning.

## Modelling and evaluation

`src/prepare_model_data.py` builds a common vehicle table by appending
the two periods. It derives vehicle age from listing year and
manufacturing year, standardizes category text, and preserves source
and period labels. Common model features are:

- Numeric: vehicle age, engine capacity and mileage.
- Categorical: make, model, transmission, fuel type and condition.
- Market-aware configuration: listing-month USD/LKR average and import
  policy environment.

Body type is Dataset 1 only; location and equipment fields were not used
in the common model feature set. DMT monthly transfer counts are kept in
the market context file but excluded from models because December 2024
coverage is unavailable.

Imputation and one-hot encoding are part of each model pipeline and fit
on training rows only. Numeric scaling is used for linear and
regularized models; tree models do not need it. Evaluation includes a
chronological Dataset 1 validation, Dataset 2 as a separate later-period
test, forward-chaining cross-validation on Dataset 1, and separate
shuffled cross-validation and random 80/20 split comparisons on the
combined table. Random splits can mix the separated periods and do not
replace the later-period test. Features were chosen from shared fields
and research relevance; permutation importance and collinearity were
reviewed after fitting, with no separate automated feature-selection
step. The two source periods are not continuous, so results do not
establish causal effects.

The numeric feature collinearity checks found a maximum VIF of 1.62 in
the Dataset 1 training rows. This did not indicate a strong numeric
collinearity concern for the features checked.

Models include Linear Regression, log-target Linear Regression, Ridge,
Random Forest and XGBoost. Both vehicle-only and vehicle-plus-market
configurations were compared where applicable.

### Results currently saved

Values below come from `outputs/results/` and are rounded. MAE is in
LKR; lower MAE is better. R-squared is not an accuracy percentage.

| Evaluation | Model/configuration | MAE | R-squared |
|---|---|---:|---:|
| Dataset 1 forward-chaining CV, mean | Log-target Linear Regression, vehicle-only | 1.235 million | 0.865 |
| Dataset 2 later-period test | Random Forest, vehicle + market context | 1.614 million | 0.541 |
| Dataset 2 later-period test | Random Forest, vehicle-only | 1.616 million | 0.533 |
| Dataset 2 later-period test | Tuned XGBoost, depth 8 | 1.799 million | 0.495 |
| Dataset 2 later-period test | Log-target Linear Regression, vehicle-only | 1.840 million | -0.019 |

No model leads every evaluation. Market context gives only a small
improvement for Random Forest on the Dataset 2 test. Dataset 2 results
remain conditional on the price conversion and the shift in time and
vehicle-label coverage.

## Anomaly analysis and explanations

`src/analyze_price_anomalies.py` uses relative prediction residuals
with thresholds calibrated from out-of-fold Dataset 1 residuals and
compares them with LOF on vehicle age, engine capacity and mileage. The
current summary reports:

| Evaluation/model | Low-price flags | High-price flags | LOF profile flags |
|---|---:|---:|---:|
| Dataset 1 validation, Random Forest | 100 | 107 | 477 |
| Dataset 1 validation, XGBoost depth 8 | 165 | 167 | 477 |
| Dataset 2 test, Random Forest | 105 | 1,505 | 275 |
| Dataset 2 test, XGBoost depth 8 | 133 | 3,123 | 275 |

The high Dataset 2 flag counts may reflect cross-period underprediction
and domain shift as well as unusual listings. Flags are review
candidates, not labels. There are no suitable verified labels for
precision, recall or F1.

`src/explain_model_predictions.py` writes global and selected-listing
SHAP summaries. `src/plot_model_results.py` creates static PNG figures
under `outputs/figures/`, including model performance,
cross-validation, actual-versus-predicted, feature diagnostics, anomaly
and SHAP plots. Tables are under `outputs/results/`, including
`model_method_experiments.csv`, `selected_model_tuning.csv`,
`price_anomaly_candidates.csv` and SHAP summaries.

## Market context and limitations

- Market factors are joined by **listing month**, not manufacturing
  year.
- Dataset 1 covers only the `suspended` policy label and USD/LKR rates
  from 321.4 to 363.4. Later periods have unseen policy categories
  and/or rates outside that training range; their effects are not
  reliably learned.
- December 2024 DMT transfer count is missing, so transfer counts are
  not model features.
- The datasets cover separated periods. Temporal change and different
  label coverage limit generalization claims.
- LOF is unsupervised and identifies unusual feature profiles; it does
  not establish unusual prices.
- The Streamlit demo was added, but a successful rendered form has not
  yet been confirmed after an initial blank browser view.
- The proposal PDF is a local file and is not included in Git. Copy it
  separately to another device if needed for report writing.

## Remaining work

- Run `python run_project.py` end to end after the path updates and
  confirm regenerated model, anomaly and SHAP outputs.
- Resolve any Streamlit startup issue and confirm the form works.
- Review flagged listings as research candidates, then write the final
  report with the conversion, period-shift and data-coverage limitations.

## Running on another device

The raw datasets and generated outputs are excluded from Git. After
cloning, copy these source files into `data/raw/`:

```text
vehicles.csv
car_price_dataset.csv
```

The runner regenerates cleaned and processed tables, model results and
figures; the previous device's generated files are not needed.

Then, from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_project.py
```

The runner executes the notebook into
`outputs/notebooks/data_cleaning_&_EDA_executed.ipynb`, then runs the
data preparation, modelling, diagnostics, SHAP and plotting scripts in
order. It stops if a step fails. Model comparison and tuning can take
time. To run the notebook interactively instead, use:

```powershell
.\.venv\Scripts\jupyter.exe notebook "notebooks/data_cleaning_&_EDA.ipynb"
```

The command-line sample check and Streamlit form demo are optional and
run separately:

```powershell
.\.venv\Scripts\python.exe src/sample_vehicle_check.py
.\.venv\Scripts\python.exe -m streamlit run src/sample_vehicle_app.py
```
