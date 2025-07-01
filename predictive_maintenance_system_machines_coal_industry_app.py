import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load models
risk_model = joblib.load("risk_model.pkl")
type_model = joblib.load("type_model.pkl")
rul_model = joblib.load("rul_model.pkl")

# Load encoder and scaler
scaler = joblib.load("predictive_maintenace_scaler.pkl")
label_encoder = joblib.load("predictive_maintenace_encoder.pkl")

# Failure type logic - now using machine_type as string
def get_failure_type(row):
    reasons = []
    if row['risk_level'] == "Low Risk":
        return "None"
    if row['vibration'] > 5.0:
        reasons.append("Acoustic Fault")
    if row['temperature'] > 50:
        reasons.append("Overheating")
    if row['load'] > 1.5:
        reasons.append("Overload")
    if row['rpm'] > 1500:
        reasons.append("RPM Overspeed")
    if row['sound'] > 90:
        reasons.append("Acoustic Fault")
    if (row['machine_type'] == "Conveyor belt" and not (198 <= row['oil_quality'] <= 242)) or \
       (row['machine_type'] == "Crusher" and not (288 <= row['oil_quality'] <= 352)) or \
       (row['machine_type'] == "Loader" and not (41 <= row['oil_quality'] <= 75)):
        reasons.append("Lubrication Issue")
    if (row['machine_type'] == "Conveyor belt" and not (20 <= row['power_usage'] <= 200)) or \
       (row['machine_type'] == "Crusher" and not (60 <= row['power_usage'] <= 900)) or \
       (row['machine_type'] == "Loader" and not (110 <= row['power_usage'] <= 640)):
        reasons.append("Electrical Fault")
    if row['downtime_percentage'] > 20:
        reasons.append("Excess Downtime")
    return ", ".join(sorted(set(reasons))) if reasons else "None"

# App layout
st.title("🔧 Predictive Maintenance System for SECL")
tabs = st.tabs(["Manual Input", "Batch Upload", "Visualization & Filter"])

# --- Manual Input ---
with tabs[0]:
    st.header("🛠️ Manual Input Panel")

    machine_type = st.selectbox("Machine Type", ["Conveyor belt", "Crusher", "Loader"])
    vibration = st.slider("Vibration (mm/s)", 0.0, 10.0, 2.5)
    temperature = st.slider("Temperature (°C)", 25, 100, 50)
    load = st.slider("Load (T/m)", 0.1, 3.0, 1.2)
    rpm = st.slider("RPM", 500, 3000, 1200)
    sound = st.slider("Sound (dB)", 70, 110, 85)
    usage_minutes = st.slider("Usage Minutes", 60, 1440, 600)
    planned_op = st.selectbox("Planned Operating Time (min)", list(range(900, 1441, 60)))
    downtime = st.slider("Downtime (min)", 0, planned_op, 100)
    oil_quality = st.number_input("Oil Quality", value=220)
    power_usage = st.number_input("Power Usage", value=150)

    downtime_percentage = round((downtime / planned_op) * 100, 2)

    # Build input data (machine_type kept as string)
    input_data = pd.DataFrame([{
        "vibration": vibration,
        "temperature": temperature,
        "load": load,
        "rpm": rpm,
        "sound": sound,
        "usage_minutes": usage_minutes,
        "planned_operating_time": planned_op,
        "downtime_minutes": downtime,
        "downtime_percentage": downtime_percentage,
        "oil_quality": oil_quality,
        "power_usage": power_usage,
        "machine_type": machine_type
    }])

    if st.button("🔍 Predict Failure Risk, Type & RUL"):
        # Scale the input
        scaled_input = scaler.transform(input_data)

        # Predict
        risk = risk_model.predict(scaled_input)[0]
        risk_label = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}.get(risk)
        rul = int(rul_model.predict(scaled_input)[0])

        input_data["risk_level"] = risk_label
        failur
