# AGENTS.md --- MSc Used-Vehicle Price Research

## Project context

Research topic: **Explainable Machine Learning for Detecting Abnormal
Used-Vehicle Prices in the Sri Lankan Market**.

The project estimates expected asking prices and identifies listings
with unusually high or low prices relative to model expectations. An
abnormal price is not proof of fraud.

Read `readme_info.md` before making project
changes. It records the research methodology, data decisions, progress
and known limitations.

## Working rules

1.  **Inspect before editing.** Review the repository, notebook,
    datasets and handover before proposing changes.
2.  **Do not repeat completed work.** Check the existing notebook and
    outputs first.
3.  **Do not overwrite the notebook** or reorganise it without explicit
    approval.
4.  **Preserve raw data.** Never modify source CSVs in place. Write
    derived files to `data/processed/`.
5.  **Dataset 2 price unit is unconfirmed.** Never assume lakhs or
    millions, and never silently convert `price_original`. Ask for
    confirmation or reliable source evidence first.
6.  **Do not force dataset integration.** Check price units, feature
    definitions, data types, periods and coverage before combining.
7.  **Market factors come later.** Prepare vehicle datasets first.
    Collect and join exchange-rate, vehicle-transfer and import-policy
    context only when vehicle data are ready and compatible.
8.  **Do not remove price outliers automatically.** Unusual asking
    prices are central to the research. Apply justified data-quality
    exclusions only, and document them.
9.  **Avoid data leakage.** Fit imputers, encoders, scalers and other
    preprocessing transformations on training data only; apply fitted
    transformations to validation/test data.
10. **Keep methods aligned with the proposal.** Compare Linear
    Regression, Random Forest and XGBoost or LightGBM; compare
    vehicle-only features with vehicle-plus-market-context features.
11. **Keep anomaly detection distinct from price prediction.** Use model
    residuals / relative deviations and compare with LOF. SHAP explains
    model predictions; it does not detect anomalies.
12. **Do not report precision, recall or F1 without suitable labelled
    evaluation data.**
13. **Avoid causal claims.** The two datasets cover separated periods,
    not a continuous time series.
14. **Respect data-source terms.** Do not scrape marketplace listings
    unless collection and research use are permitted and the user
    approves.
15. **Protect repository contents.** Do not commit large datasets,
    credentials or private information to a public repository. Recommend
    a private repository or suitable data storage when needed.

## Coding and communication preferences

-   Keep explanations short and practical.
-   Work one step at a time; do not generate a large pipeline unless
    requested.
-   When working interactively, provide one code cell at a time and wait
    for the user's output.
-   Put a short, natural comment at the start of every code cell.
-   Use clear names, reproducible paths and straightforward Python.
-   Add notebook headings only for meaningful new sections; preserve
    existing numbering.
-   Explain proposed changes before making major edits.
-   Do not invent data, results, source facts or assumptions.

## Initial Codex task

Do not edit files on the first pass. Inspect the repository, notebook
and datasets, then provide a concise status report: - Current project
structure. - Notebook sections and completed work. - Available raw and
processed datasets. - Any mismatch with the handover document. - Any
unresolved issue, especially Dataset 2's price unit. - One recommended
next step.

Wait for the user's approval before substantial changes.
