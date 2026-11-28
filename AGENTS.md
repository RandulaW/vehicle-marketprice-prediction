# Project instructions

## Research purpose

Estimate asking prices for used vehicles in Sri Lanka and identify
listings whose prices differ substantially from model estimates. A flag
is a review signal, not proof of fraud.

Read `readme_info.md` before changing the project. It records the data
decisions, completed work, current results, reproduction steps and open
limitations. The notebook and scripts are the source of truth for the
implementation.

## Decisions to preserve

- Dataset 2 uses the project-approved working conversion
  `price_lkr = price_original * 100,000`. Keep `price_original`. The
  dataset source does not independently document its unit, so describe
  results involving Dataset 2 as conditional on this conversion. Do not
  change the conversion without reliable source evidence or a direct
  user instruction.
- Dataset 1 and Dataset 2 cover separate periods. Do not present them as
  a continuous time series or make causal claims about market factors.
- Vehicle listings are evaluated using vehicle-only features and, in a
  separate configuration, vehicle features plus listing-month USD/LKR
  and import-policy context. DMT vehicle-transfer counts are retained
  in context data but excluded from model features because December 2024
  coverage is missing.
- Price outliers are relevant to the research. Do not remove them solely
  because they are statistically unusual. Apply only justified data
  quality rules and document them.
- Residual-based price flags and LOF profile flags are different
  signals. SHAP explains predictions; it does not detect anomalies.
- There are no verified anomaly labels. Do not report precision,
  recall or F1 as measured model performance.
- The feature list is based on shared fields and research relevance.
  Permutation importance and collinearity were reviewed after fitting;
  there was no separate automated feature-selection step.

## Data and file handling

- Raw files are `vehicles.csv` and `car_price_dataset.csv` in
  `data/raw/`. They are not included in Git; do not add large source
  datasets or private files to the public repository.
- The notebook writes cleaned data to `data/processed/`. Scripts use
  that directory for derived modelling tables. Keep raw data unchanged.
- Preserve the existing notebook sections, analysis and saved outputs.
  Change it only for a requested purpose; do not replace or reorganize
  it wholesale.
- Keep generated tables, figures and the executed notebook under
  `outputs/`. Do not delete prior results unless they are confirmed
  redundant and removal is requested.
- Do not scrape listings unless the user has approved it and the source
  terms permit the intended research use.

## Reproducible workflow

From the repository root, `python run_project.py` executes the cleaning
and EDA notebook to a separate copy, then runs data preparation, model
comparisons, tuning, feature diagnostics, anomaly analysis, SHAP and
plotting in order. It requires the two raw CSVs in `data/raw/` and the
packages in `requirements.txt`. It does not launch the interactive
terminal or Streamlit demos.

Important files:

- `notebooks/data_cleaning_&_EDA.ipynb`: cleaning and EDA.
- `src/prepare_model_data.py`: common vehicle table and vehicle age.
- `src/prepare_market_context.py` and `src/integrate_market_context.py`:
  market context preparation and joining by listing month.
- `src/model_preprocessing.py`: features, preprocessing and time splits.
- `src/price_prediction.py`,
  `src/compare_vehicle_and_market_aware_models.py`,
  `src/compare_model_method_experiments.py` and
  `src/tune_selected_models.py`: model evaluation.
- `src/analyze_model_features.py`, `src/analyze_price_anomalies.py`,
  `src/explain_model_predictions.py` and `src/plot_model_results.py`:
  diagnostics and figures.
- `src/sample_vehicle_check.py` and `src/sample_vehicle_app.py`:
  interactive demonstrations; run separately.

## Working style

- Inspect the notebook, data and current outputs before repeating work.
- Avoid data leakage: fit imputers, encoders and scalers on training
  rows only.
- Keep code readable, comments brief and natural, and paths reproducible.
- Preserve established names and structure unless a change is needed.
- When the user is running notebook cells interactively, provide one
  step at a time and wait for their output.
- Keep replies short, explain decisions plainly, and do not invent
  results or source facts.
