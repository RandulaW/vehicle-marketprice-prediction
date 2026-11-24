"""Run a sample asking-price check from the command line."""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_price_anomalies import LOF_CONTAMINATION, LOF_NEIGHBORS
from model_preprocessing import (
    FEATURES,
    MARKET_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)
from price_prediction import make_model


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "vehicle_market_model_data.csv"
MARKET_DATA_PATH = ROOT / "data" / "processed" / "market_context_monthly.csv"
THRESHOLD_PATH = ROOT / "outputs" / "results" / "residual_thresholds.csv"
RANDOM_STATE = 42


def ask_number(label: str, *, integer: bool = False, minimum: float = 0) -> float:
    """Prompt again until a valid non-negative number is entered."""
    while True:
        value = input(f"{label}: ").strip().replace(",", "")
        try:
            number = int(value) if integer else float(value)
        except ValueError:
            print("Enter a valid number.")
            continue
        if number < minimum:
            print(f"Enter a value of at least {minimum:g}.")
            continue
        return number


def ask_category(label: str, training: pd.Series) -> str:
    """Read a category and return the matching training label when available."""
    counts = training.dropna().astype(str).value_counts()
    known = counts.index.tolist()
    matches = {category.casefold(): category for category in known}
    examples = ", ".join(known[:5])
    value = input(f"{label} (examples: {examples}): ").strip()
    return matches.get(value.casefold(), value)


def make_vehicle_record(
    training: pd.DataFrame,
    market_data: pd.DataFrame,
    listing_month: str,
    vehicle_year: int,
    engine_cc: float,
    mileage_km: float,
    make: str,
    model: str,
    transmission: str,
    fuel_type: str,
    condition: str,
) -> tuple[pd.DataFrame, dict, list[str]]:
    """Build the model row and note any market or category limits."""
    market_rows = market_data.set_index("listing_month")
    if listing_month not in market_rows.index:
        raise ValueError("No market context is available for that listing month.")

    listing_year = int(listing_month[:4])
    if vehicle_year > listing_year:
        raise ValueError("Manufacturing year cannot be later than listing year.")

    context = market_rows.loc[listing_month]
    policy = str(context["import_policy_environment"])
    exchange_rate = float(context["usd_lkr_monthly_average"])
    warnings = []

    training_policies = set(training["import_policy_environment"].dropna())
    if policy not in training_policies:
        warnings.append(f"The model was not trained on the '{policy}' policy period.")

    rate_min = training["usd_lkr_monthly_average"].min()
    rate_max = training["usd_lkr_monthly_average"].max()
    if not rate_min <= exchange_rate <= rate_max:
        warnings.append(
            f"The exchange rate {exchange_rate:.1f} is outside the Dataset 1 "
            f"training range ({rate_min:.1f}-{rate_max:.1f})."
        )

    categories = {
        "make": make,
        "model": model,
        "transmission": transmission,
        "fuel_type": fuel_type,
        "condition": condition,
    }
    for column, value in categories.items():
        labels = training[column].dropna().astype(str)
        match = {label.casefold(): label for label in labels.unique()}.get(value.casefold())
        if match is None:
            warnings.append(f"'{value}' {column} was not seen during training.")
        else:
            categories[column] = match

    row = {
        "vehicle_age_years": listing_year - vehicle_year,
        "engine_cc": engine_cc,
        "mileage_km": mileage_km,
        **categories,
        "usd_lkr_monthly_average": exchange_rate,
        "import_policy_environment": policy,
    }
    return pd.DataFrame([row]), context.to_dict(), warnings


def get_vehicle_input(
    training: pd.DataFrame,
    market_data: pd.DataFrame,
) -> tuple[pd.DataFrame, float, str, dict, list[str]]:
    """Collect vehicle details and look up context for the listing month."""
    available_months = market_data["listing_month"].tolist()
    policy_months = market_data.groupby("import_policy_environment")[
        "listing_month"
    ].apply(list)

    print("Enter the listing month and vehicle details.")
    print("Market context months:", ", ".join(available_months))
    print("Policy-period labels recorded in project context:")
    for policy, months in policy_months.items():
        print(f"- {policy}: {', '.join(months)}")

    while True:
        listing_month = input("Listing month (YYYY-MM): ").strip()
        if listing_month in available_months:
            break
        print("No saved market context for that month. Choose one from the list above.")

    vehicle_year = ask_number("Manufacturing year", integer=True)
    engine_cc = ask_number("Engine capacity (cc)")
    mileage_km = ask_number("Mileage (km)")
    categories = {
        column: ask_category(label, training[column])
        for column, label in [
            ("make", "Make"),
            ("model", "Model"),
            ("transmission", "Transmission"),
            ("fuel_type", "Fuel type"),
            ("condition", "Condition"),
        ]
    }
    vehicle, context, warnings = make_vehicle_record(
        training,
        market_data,
        listing_month,
        vehicle_year,
        engine_cc,
        mileage_km,
        **categories,
    )
    asking_price = ask_number("Asking price in LKR (commas are allowed)", minimum=0.01)
    return vehicle, asking_price, listing_month, context, warnings


def fit_lof(training: pd.DataFrame):
    """Fit LOF on Dataset 1 vehicle age, engine capacity and mileage."""
    detector = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LocalOutlierFactor(
            n_neighbors=LOF_NEIGHBORS,
            contamination=LOF_CONTAMINATION,
            novelty=True,
        ),
    )
    detector.fit(training[NUMERIC_FEATURES])
    return detector


def original_feature(encoded_name: str) -> str:
    """Map a transformed feature back to its source column."""
    name = encoded_name.split("__", 1)[-1]
    for feature in FEATURES + MARKET_FEATURES:
        if name == feature or name.startswith(f"{feature}_"):
            return feature
    return name


def display_feature(feature: str, vehicle: pd.DataFrame) -> str:
    """Show the feature name together with the value entered by the user."""
    row = vehicle.iloc[0]
    if feature == "vehicle_age_years":
        return f"Vehicle age ({row[feature]:.0f} years)"
    if feature == "engine_cc":
        return f"Engine capacity ({row[feature]:,.0f} cc)"
    if feature == "mileage_km":
        return f"Mileage ({row[feature]:,.0f} km)"
    if feature == "usd_lkr_monthly_average":
        return f"USD/LKR rate ({row[feature]:.1f})"
    if feature == "import_policy_environment":
        return f"Policy period ({str(row[feature]).replace('_', ' ')})"
    value = str(row[feature]).title()
    return f"{feature.replace('_', ' ').title()} ({value})"


def load_demo_components() -> dict:
    """Fit the model once and prepare the reusable demo components."""
    data = pd.read_csv(DATA_PATH, parse_dates=["listing_date"])
    market_data = pd.read_csv(MARKET_DATA_PATH)
    training = data.loc[data["market_period"] == "earlier"].copy()
    columns = FEATURES + MARKET_FEATURES
    model = make_model(
        RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        scale_numeric=False,
        include_market_context=True,
    )
    model.fit(training[columns], training[TARGET])

    lof = fit_lof(training)
    thresholds = pd.read_csv(THRESHOLD_PATH)
    later_threshold = thresholds.loc[
        (thresholds["evaluation"] == "Dataset 2 later-period test")
        & (thresholds["model"] == "Random Forest")
    ].iloc[0]

    preprocessor = model.named_steps["preprocessing"]
    background_rows = training.sample(min(20, len(training)), random_state=RANDOM_STATE)
    background = preprocessor.transform(background_rows[columns])
    if hasattr(background, "toarray"):
        background = background.toarray()
    explainer = shap.TreeExplainer(
        model.named_steps["regression"],
        data=background,
        feature_perturbation="interventional",
    )

    return {
        "training": training,
        "market_data": market_data,
        "columns": columns,
        "model": model,
        "lof": lof,
        "explainer": explainer,
        "low_cutoff": float(later_threshold["low_relative_deviation"]),
        "high_cutoff": float(later_threshold["high_relative_deviation"]),
    }


def check_vehicle(components: dict, vehicle: pd.DataFrame, asking_price: float) -> dict:
    """Estimate a price and return the flags and SHAP contributions."""
    model = components["model"]
    expected_price = max(float(model.predict(vehicle[components["columns"]])[0]), 1.0)
    relative_deviation = asking_price / expected_price - 1
    difference = abs(relative_deviation)
    relation = "above" if relative_deviation >= 0 else "below"

    low_cutoff = components["low_cutoff"]
    high_cutoff = components["high_cutoff"]
    if relative_deviation > high_cutoff:
        price_result = f"Review alert: asking price is {difference:.1%} above the estimate."
    elif relative_deviation < low_cutoff:
        price_result = f"Review alert: asking price is {difference:.1%} below the estimate."
    else:
        price_result = f"No review alert: asking price is {difference:.1%} {relation} the estimate."

    vehicle_flag = components["lof"].predict(vehicle[NUMERIC_FEATURES])[0] == -1

    preprocessor = model.named_steps["preprocessing"]
    transformed = preprocessor.transform(vehicle[components["columns"]])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    explainer = components["explainer"]
    contributions = np.asarray(explainer.shap_values(transformed))[0]
    reference_price = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    explanation_error = abs(reference_price + contributions.sum() - expected_price)
    grouped = pd.Series(contributions, index=preprocessor.get_feature_names_out())
    grouped.index = [original_feature(name) for name in grouped.index]
    grouped = grouped.groupby(level=0).sum().sort_values(key=np.abs, ascending=False)

    return {
        "expected_price": expected_price,
        "asking_price": asking_price,
        "difference": difference,
        "relation": relation,
        "price_result": price_result,
        "vehicle_flag": vehicle_flag,
        "contributions": grouped.head(3),
        "reference_price": reference_price,
        "explanation_error": explanation_error,
    }


def print_result(result: dict, vehicle: pd.DataFrame, listing_month: str) -> None:
    """Print the result for the terminal demo."""
    print("\nPrice check")
    print(f"Model estimate: LKR {result['expected_price']:,.0f}")
    print(
        f"Asking price: LKR {result['asking_price']:,.0f} "
        f"({result['difference']:.1%} {result['relation']})"
    )
    print(f"Price review: {result['price_result']}")
    profile_result = (
        "An unusual vehicle profile was found."
        if result["vehicle_flag"]
        else "No unusual pattern was found in age, engine capacity and mileage."
    )
    print(f"Vehicle profile: {profile_result}")
    row = vehicle.iloc[0]
    print(
        f"Market context used: {listing_month}, USD/LKR "
        f"{row['usd_lkr_monthly_average']:.1f}, "
        f"policy period '{row['import_policy_environment']}'."
    )
    print("\nLargest SHAP contributions:")
    for feature, value in result["contributions"].items():
        sign = "+" if value >= 0 else "-"
        label = display_feature(feature, vehicle)
        print(f"- {label}: {sign}LKR {abs(value) / 1_000_000:.1f} million")
    print("Contributions are relative to SHAP's model reference, not actual discounts.")
    if result["explanation_error"] > 1:
        print("Note: SHAP contributions did not exactly add up to the prediction.")
    print("No review alert does not confirm that the asking price is fair.")


def main() -> None:
    # Run the terminal version of the same check used by the web demo.
    components = load_demo_components()
    vehicle, asking_price, listing_month, _, warnings = get_vehicle_input(
        components["training"],
        components["market_data"],
    )
    for warning in warnings:
        print(f"Note: {warning}")
    result = check_vehicle(components, vehicle, asking_price)
    print_result(result, vehicle, listing_month)


if __name__ == "__main__":
    main()
