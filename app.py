import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️", layout="wide")

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"

def find_file(name):
    p1 = ART / name
    p2 = BASE / name
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Missing required file: {name}")

@st.cache_resource
def load_assets():
    clf = joblib.load(find_file("final_delay_classifier.joblib"))
    reg = joblib.load(find_file("final_delay_regressor.joblib"))
    prep = joblib.load(find_file("preprocessing_artifacts.joblib"))
    scaler = joblib.load(find_file("standard_scaler.joblib"))
    with open(find_file("feature_columns.json"), "r", encoding="utf-8") as f:
        feature_columns = json.load(f)
    return clf, reg, prep, scaler, feature_columns

clf, reg, prep, scaler, FEATURE_COLUMNS = load_assets()
enc = prep["encoders"]

AIRLINES = sorted(enc["freq"]["AIRLINE"].keys())
ORIGINS = sorted(enc["freq"]["ORIGIN_AIRPORT"].keys())
DESTINATIONS = sorted(enc["freq"]["DESTINATION_AIRPORT"].keys())

st.title("✈️ U.S. Flight Delay Predictor")
st.caption("NTI — Machine Learning for Data Analysis | 2015 U.S. Flight Delays and Cancellations")
st.write(
    "Enter information known before departure. The app reproduces the saved "
    "feature encoding and scaling used by the trained models."
)

with st.form("flight_form"):
    st.subheader("Flight Information")

    c1, c2, c3 = st.columns(3)
    with c1:
        airline = st.selectbox("Airline", AIRLINES)
        origin = st.selectbox("Origin airport", ORIGINS)
        destination = st.selectbox("Destination airport", DESTINATIONS)
    with c2:
        month = st.slider("Month", 1, 12, 6)
        day = st.slider("Day", 1, 31, 15)
        dow = st.slider("Day of week (1=Mon, 7=Sun)", 1, 7, 1)
    with c3:
        dep_hour = st.slider("Scheduled departure hour", 0, 23, 10)
        arr_hour = st.slider("Scheduled arrival hour", 0, 23, 12)
        scheduled_time = st.number_input("Scheduled flight time (minutes)", 20.0, 1000.0, 120.0, 5.0)

    c4, c5 = st.columns(2)
    with c4:
        distance = st.number_input("Distance (miles)", 50.0, 6000.0, 800.0, 25.0)
        origin_hourly_flights = st.number_input(
            "Scheduled flights from origin in this hour", 1.0, 200.0, 10.0, 1.0,
            help="Operational schedule/congestion proxy used by the trained model."
        )
        origin_peak_pressure = st.number_input(
            "Origin peak pressure ratio", 0.05, 10.0, 1.0, 0.05,
            help="Hourly origin traffic divided by that airport's normal hourly traffic. 1.0 = normal."
        )
    with c5:
        aircraft_daily_leg = st.number_input("Aircraft leg number today", 1.0, 20.0, 1.0, 1.0)
        aircraft_daily_legs = st.number_input("Aircraft scheduled legs today", 1.0, 20.0, 4.0, 1.0)
        turnaround = st.number_input(
            "Scheduled turnaround before this flight (minutes)", 0.0, 720.0, 120.0, 5.0
        )

    submitted = st.form_submit_button("Predict Flight Delay", use_container_width=True)

def map_value(mapping, key, default):
    return float(mapping.get(key, default))

def make_features():
    route = f"{origin}_{destination}"

    row = {
        "MONTH": month,
        "DAY": day,
        "DAY_OF_WEEK": dow,
        "SCHEDULED_TIME": scheduled_time,
        "SCHED_DEP_HOUR": dep_hour,
        "SCHED_ARR_HOUR": arr_hour,
        "ORIGIN_HOURLY_FLIGHTS": origin_hourly_flights,
        "AIRCRAFT_DAILY_LEG": aircraft_daily_leg,
        "AIRCRAFT_DAILY_LEGS": aircraft_daily_legs,
        "IS_WEEKEND": int(dow in [6, 7]),
        "IS_PEAK_HOUR": int((6 <= dep_hour <= 9) or (16 <= dep_hour <= 20)),
        "IS_RED_EYE": int(dep_hour >= 22 or dep_hour < 5),
        "IS_OVERNIGHT_ARRIVAL": int(arr_hour < dep_hour),
        "HOUR_SIN": np.sin(2 * np.pi * dep_hour / 24),
        "HOUR_COS": np.cos(2 * np.pi * dep_hour / 24),
        "DOW_SIN": np.sin(2 * np.pi * dow / 7),
        "DOW_COS": np.cos(2 * np.pi * dow / 7),
        "IS_LONG_HAUL": int(distance > 1500),
        "SCHED_SPEED_MPH": distance / (scheduled_time / 60.0),
        "TURNAROUND_MIN": min(max(turnaround, 0.0), 720.0),
        "IS_TIGHT_TURNAROUND": int(turnaround < 45 and aircraft_daily_leg > 1),
        "LEG_PROGRESS": aircraft_daily_leg / max(aircraft_daily_legs, 1.0),
        "ORIGIN_PEAK_PRESSURE": origin_peak_pressure,
    }

    raw_cats = {
        "AIRLINE": airline,
        "ORIGIN_AIRPORT": origin,
        "DESTINATION_AIRPORT": destination,
        "ROUTE": route,
    }

    prior = float(enc["prior"])
    for col, value in raw_cats.items():
        row[f"{col}_FREQ"] = map_value(enc["freq"][col], value, 0.0)
        row[f"{col}_TE"] = map_value(enc["target"][col], value, prior)

    hubs = set(enc["hub_airports"])
    row["IS_HUB_ORIGIN"] = int(origin in hubs)
    row["IS_HUB_DEST"] = int(destination in hubs)
    row["IS_HUB_TO_HUB"] = int(origin in hubs and destination in hubs)

    X = pd.DataFrame([row])

    # Scale exactly the columns stored by the training notebook.
    scale_cols = prep["scale_cols"]
    X[scale_cols] = scaler.transform(X[scale_cols])

    # Exact final feature order expected by both saved models.
    X = X.reindex(columns=FEATURE_COLUMNS)
    return X

if submitted:
    if origin == destination:
        st.error("Origin and destination airports must be different.")
    elif aircraft_daily_leg > aircraft_daily_legs:
        st.error("Aircraft leg number cannot exceed total scheduled legs.")
    else:
        X = make_features()

        delay_prob = float(clf.predict_proba(X)[0, 1])
        delayed = int(clf.predict(X)[0])
        delay_minutes = float(reg.predict(X)[0])

        st.divider()
        st.subheader("Prediction Results")

        a, b, c = st.columns(3)
        a.metric("Classification", "DELAYED" if delayed else "ON-TIME")
        b.metric("Delay Probability", f"{delay_prob:.1%}")
        c.metric("Regression Estimate", f"{delay_minutes:.0f} min")

        st.progress(min(max(delay_prob, 0.0), 1.0))

        if delayed:
            st.warning(
                "Classification model: DELAYED — predicted probability is above the decision threshold."
            )
        else:
            st.success(
                "Classification model: ON-TIME — predicted probability is below the decision threshold."
            )

        st.info(
            f"Regression model: independently estimates the arrival delay at approximately "
            f"{delay_minutes:.0f} minutes. Because classification and regression are separate "
            "models trained for different targets, their outputs can occasionally differ."
        )

        with st.expander("Technical details"):
            st.write(f"Classifier threshold used during the project: {prep['delay_threshold']} minutes")
            st.write(f"Number of final model features: {len(FEATURE_COLUMNS)}")
            st.dataframe(X, use_container_width=True)

st.divider()
st.caption(
    "Academic ML project. Predictions are based on a model trained on a sample of 2015 U.S. domestic flights "
    "and should not be used as live airline operational advice."
)
