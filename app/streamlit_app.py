import streamlit as st
import pandas as pd
import requests
from pathlib import Path

st.set_page_config(
    page_title="Ford Fleet AI Copilot",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "model_training_table.csv"

API_URL = "http://127.0.0.1:8000/predict/vehicle"

st.title("Ford Fleet AI Copilot")
st.write("Predicts 30-day vehicle failure risk and helps prioritize maintenance.")

df = pd.read_csv(DATA_PATH)

vehicle_ids = df["vehicle_id"].tolist()

st.sidebar.header("Controls")

selected_vehicle = st.sidebar.selectbox(
    "Select Vehicle",
    vehicle_ids
)

top_n = st.sidebar.slider(
    "Number of vehicles to scan",
    min_value=10,
    max_value=200,
    value=50,
    step=10
)

def call_prediction_api(vehicle_id):
    payload = {"vehicle_id": vehicle_id}

    response = requests.post(API_URL, json=payload)

    if response.status_code != 200:
        return None

    return response.json()

st.header("Single Vehicle Risk Check")

if st.button("Predict Selected Vehicle"):
    result = call_prediction_api(selected_vehicle)

    if result is None:
        st.error("API call failed. Make sure FastAPI is running.")
    else:
        col1, col2, col3 = st.columns(3)

        col1.metric("Vehicle ID", result["vehicle_id"])
        col2.metric("Failure Probability", result["failure_probability_percent"])
        col3.metric("Risk Level", result["risk_level"])

        st.json(result)

st.header("High-Risk Vehicle Ranking")

if st.button("Scan Fleet"):
    results = []

    with st.spinner("Calling model API for vehicles..."):
        for vehicle_id in vehicle_ids[:top_n]:
            result = call_prediction_api(vehicle_id)
            if result:
                results.append(result)

    if results:
        results_df = pd.DataFrame(results)

        high_risk_df = results_df.sort_values(
            "failure_probability",
            ascending=False
        )

        st.subheader("Top High-Risk Vehicles")
        st.dataframe(high_risk_df)

        st.subheader("Risk Level Breakdown")
        risk_counts = high_risk_df["risk_level"].value_counts()
        st.bar_chart(risk_counts)

        st.subheader("Failure Probability Distribution")
        st.line_chart(
            high_risk_df["failure_probability"].reset_index(drop=True)
        )
    else:
        st.error("No predictions returned. Check if FastAPI is running.")