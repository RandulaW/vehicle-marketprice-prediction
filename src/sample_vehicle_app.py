"""Small Streamlit interface for the vehicle-price research demo."""

import streamlit as st

from sample_vehicle_check import (
    check_vehicle,
    display_feature,
    load_demo_components,
    make_vehicle_record,
)


st.set_page_config(page_title="Vehicle price research demo")
st.title("Used-vehicle price check")
st.caption("Research demonstration for Sri Lankan vehicle listings")
st.info(
    "This tool screens asking prices for review. It does not confirm a fair "
    "price or prove that a listing is incorrect."
)


@st.cache_resource
def get_components():
    return load_demo_components()


with st.spinner("Loading the data and fitting the model..."):
    components = get_components()

training = components["training"]
market_data = components["market_data"]
months = market_data["listing_month"].tolist()

listing_month = st.selectbox("Listing month", months)
context = market_data.loc[market_data["listing_month"] == listing_month].iloc[0]
policy = context["import_policy_environment"]
exchange_rate = float(context["usd_lkr_monthly_average"])
st.caption(
    f"Market context: USD/LKR {exchange_rate:.1f}; "
    f"policy period: {str(policy).replace('_', ' ')}"
)

if policy not in set(training["import_policy_environment"].dropna()):
    st.warning("The model did not learn this policy period from Dataset 1.")

rate_min = training["usd_lkr_monthly_average"].min()
rate_max = training["usd_lkr_monthly_average"].max()
if not rate_min <= exchange_rate <= rate_max:
    st.warning(
        f"This exchange rate is outside Dataset 1's training range "
        f"({rate_min:.1f} to {rate_max:.1f}); treat the estimate cautiously."
    )

with st.form("vehicle_details"):
    listing_year = int(listing_month[:4])
    vehicle_year = st.number_input(
        "Manufacturing year",
        min_value=1900,
        max_value=listing_year,
        value=min(2015, listing_year),
        step=1,
    )
    engine_cc = st.number_input(
        "Engine capacity (cc)",
        min_value=0.0,
        value=1500.0,
        step=100.0,
    )
    mileage_km = st.number_input(
        "Mileage (km)",
        min_value=0.0,
        value=85000.0,
        step=5000.0,
    )

    left, right = st.columns(2)
    make = left.text_input("Make", value="Toyota")
    vehicle_model = right.text_input("Model", value="Aqua")
    transmission = left.selectbox(
        "Transmission",
        training["transmission"].dropna().value_counts().index.tolist(),
        format_func=lambda value: str(value).title(),
    )
    fuel_type = right.selectbox(
        "Fuel type",
        training["fuel_type"].dropna().value_counts().index.tolist(),
        format_func=lambda value: str(value).title(),
    )
    condition = left.selectbox(
        "Condition",
        training["condition"].dropna().value_counts().index.tolist(),
        format_func=lambda value: str(value).title(),
    )
    asking_price = st.number_input(
        "Asking price (LKR)",
        min_value=1.0,
        value=6500000.0,
        step=100000.0,
        format="%.0f",
    )
    submitted = st.form_submit_button("Check asking price", type="primary")

if submitted:
    vehicle, _, warnings = make_vehicle_record(
        training,
        market_data,
        listing_month,
        int(vehicle_year),
        engine_cc,
        mileage_km,
        make,
        vehicle_model,
        transmission,
        fuel_type,
        condition,
    )
    for warning in warnings:
        st.warning(warning)

    result = check_vehicle(components, vehicle, asking_price)
    st.subheader("Price check")
    first, second = st.columns(2)
    first.metric("Model estimate", f"LKR {result['expected_price']:,.0f}")
    second.metric(
        "Asking price",
        f"LKR {asking_price:,.0f}",
        f"{result['difference']:.1%} {result['relation']}",
    )
    st.write(f"**Price review:** {result['price_result']}")
    if result["vehicle_flag"]:
        st.warning("The age, engine capacity and mileage form an unusual profile.")
    else:
        st.write("No unusual age, engine capacity or mileage pattern was found.")

    st.subheader("Main SHAP contributions")
    st.write(f"Reference prediction: LKR {result['reference_price']:,.0f}")
    st.caption(
        "This is the model's starting prediction before the entered vehicle "
        "details are considered."
    )
    for feature, value in result["contributions"].items():
        label = display_feature(feature, vehicle)
        sign = "+" if value >= 0 else "−"
        st.write(f"{label}: {sign}LKR {abs(value) / 1_000_000:.2f} million")
    with st.expander("How to read this explanation"):
        st.write(
            "The reference prediction plus all feature contributions gives "
            "the model estimate. A plus sign means the feature moved the "
            "estimate up from the reference; a minus sign means it moved it "
            "down. These are model calculations, not real-world causes or "
            "discounts. Only the three largest contributions are shown here."
        )
    st.caption("A missing alert does not confirm that the asking price is fair.")
