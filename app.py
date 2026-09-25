import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Flight Delay Intelligence", page_icon="✈️", layout="wide", initial_sidebar_state="collapsed")

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

# ---------- Dashboard styling ----------
st.markdown("""
<style>
:root{
    --gold:#D4AF37;
    --gold2:#F4D06F;
    --black:#0A0A0A;
    --paper:#FFFFFF;
}
.stApp{
    background:
      radial-gradient(circle at 82% 4%, rgba(212,175,55,.16), transparent 23rem),
      linear-gradient(180deg,#ffffff 0%,#fbfaf6 52%,#ffffff 100%);
    color:#111;
}
[data-testid="stHeader"]{background:rgba(255,255,255,.82);backdrop-filter:blur(12px)}
.block-container{max-width:1450px;padding-top:1.35rem;padding-bottom:3rem}
.hero{
    position:relative; overflow:hidden; min-height:330px; border-radius:28px;
    padding:52px 56px; display:flex; align-items:flex-end;
    background:
      linear-gradient(90deg,rgba(0,0,0,.92) 0%,rgba(0,0,0,.72) 43%,rgba(0,0,0,.30) 100%),
      url("https://images.unsplash.com/photo-1521727857535-28d2047314ac?auto=format&fit=crop&w=2200&q=85");
    background-size:cover; background-position:center;
    box-shadow:0 22px 60px rgba(0,0,0,.20); border:1px solid rgba(212,175,55,.45);
}
.hero:after{content:"";position:absolute;inset:0;border:1px solid rgba(255,255,255,.10);border-radius:28px;pointer-events:none}
.hero-content{position:relative;z-index:2;max-width:800px}
.eyebrow{color:var(--gold2);font-weight:800;letter-spacing:.18em;font-size:.78rem;margin-bottom:12px}
.hero h1{color:white;font-size:3.25rem;line-height:1.02;margin:0 0 14px;font-weight:850;letter-spacing:-.035em}
.hero p{color:rgba(255,255,255,.83);font-size:1.05rem;max-width:720px;margin:0}
.gold-line{width:72px;height:4px;background:var(--gold);border-radius:9px;margin:20px 0}
.section-kicker{color:#9b7818;font-size:.76rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase;margin-top:1.7rem}
.section-title{font-size:1.75rem;font-weight:850;color:#111;margin:.2rem 0 .25rem}
.section-copy{color:#666;margin-bottom:1rem}
div[data-testid="stForm"]{
    background:rgba(255,255,255,.94);border:1px solid #e8e0c6;border-radius:24px;
    padding:1.2rem 1.35rem 1.35rem;box-shadow:0 16px 45px rgba(0,0,0,.07)
}
div[data-testid="stForm"] h3{color:#111}
div[data-testid="stSelectbox"] label, div[data-testid="stNumberInput"] label, div[data-testid="stSlider"] label{
    font-weight:700;color:#202020
}
.stButton>button, div[data-testid="stFormSubmitButton"] button{
    background:linear-gradient(90deg,#0b0b0b,#242424)!important;color:#fff!important;
    border:1px solid #D4AF37!important;border-radius:12px!important;font-weight:800!important;
    min-height:48px;box-shadow:0 8px 20px rgba(0,0,0,.13)
}
.stButton>button:hover, div[data-testid="stFormSubmitButton"] button:hover{
    color:#F4D06F!important;border-color:#F4D06F!important;transform:translateY(-1px)
}
[data-testid="stMetric"]{
    background:#0c0c0c;border:1px solid rgba(212,175,55,.65);border-radius:18px;
    padding:20px 22px;box-shadow:0 12px 28px rgba(0,0,0,.12)
}
[data-testid="stMetricLabel"]{color:#D4AF37!important;font-weight:800}
[data-testid="stMetricValue"]{color:#fff!important}
div[data-testid="stProgress"] > div > div > div > div{background-color:#D4AF37}
.result-head{
    margin-top:.4rem;padding:20px 24px;border-left:5px solid #D4AF37;
    background:linear-gradient(90deg,#111,#242424);border-radius:16px;color:#fff
}
.result-head small{color:#D4AF37;font-weight:800;letter-spacing:.13em}
.result-head h2{color:#fff;margin:.2rem 0 0;font-size:1.8rem}
.footer-note{text-align:center;color:#777;font-size:.83rem;padding-top:8px}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-content">
    <div class="eyebrow">MACHINE LEARNING • FLIGHT OPERATIONS</div>
    <h1>Flight Delay<br>Intelligence</h1>
    <div class="gold-line"></div>
    <p>Estimate delay risk and expected arrival delay using the same preprocessing pipeline and trained models developed for the project.</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-kicker">Prediction workspace</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Flight details</div>', unsafe_allow_html=True)
st.markdown('<div class="section-copy">Enter the information available before departure, then run the prediction.</div>', unsafe_allow_html=True)

with st.form("flight_form"):
    st.markdown("### ✈ Flight information")

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

        st.markdown(
            '<div class="result-head"><small>MODEL OUTPUT</small><h2>Prediction results</h2></div>',
            unsafe_allow_html=True,
        )

        a, b, c = st.columns(3)
        a.metric("FLIGHT STATUS", "DELAYED" if delayed else "ON-TIME")
        b.metric("DELAY PROBABILITY", f"{delay_prob:.1%}")
        c.metric("ESTIMATED DELAY", f"{delay_minutes:.0f} min")

        st.progress(min(max(delay_prob, 0.0), 1.0))

        if delayed:
            st.warning("Higher delay risk detected — the classification model places this flight above its decision threshold.")
        else:
            st.success("Lower delay risk detected — the classification model places this flight below its decision threshold.")

        st.info(
            f"The regression model estimates an arrival delay of approximately {delay_minutes:.0f} minutes. "
            "Classification and regression are separate models, so their outputs can occasionally differ."
        )

        with st.expander("Model details"):
            st.write(f"Delay threshold used in the project: {prep['delay_threshold']} minutes")
            st.write(f"Features used by the final models: {len(FEATURE_COLUMNS)}")
            st.dataframe(X, use_container_width=True)

st.divider()
st.markdown(
    '<div class="footer-note"><b>Flight Delay Intelligence</b> · Machine Learning Project · '
    '2015 U.S. Domestic Flights<br>Academic prediction system — not live airline operational advice.</div>',
    unsafe_allow_html=True,
)
