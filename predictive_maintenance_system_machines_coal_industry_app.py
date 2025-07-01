
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained models and pre-processing tools
risk_model = joblib.load('risk_model.pkl')
type_model = joblib.load('type_model.pkl')
rul_model = joblib.load('rul_model.pkl')

# Load encoder or scaler 
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

def get_failure_type(row):
    reasons = []
    if row['failure_risk'] == 0:
        return "None"
    if row['vibration'] > 5.0:
        reasons.append("Vibration")
    if row['temperature'] > 50:
        reasons.append("Overheating")
    if row['load'] > 1.5:
        reasons.append("Overload")
    if row['rpm'] > 1500:
        reasons.append("RPM Overspeed")
    if row['sound'] > 90:
        reasons.append("Acoustic Fault")
    if (row['machine_type'] == "Conveyor belt" and not (198 <= row['oil_quality'] <= 242)) or        (row['machine_type'] == "Crusher" and not (288 <= row['oil_quality'] <= 352)) or        (row['machine_type'] == "Loader" and not (41 <= row['oil_quality'] <= 75)):
        reasons.append("Lubrication Issue")
    if (row['machine_type'] == "Conveyor belt" and not (20 <= row['power_usage'] <= 200)) or        (row['machine_type'] == "Crusher" and not (60 <= row['power_usage'] <= 900)) or        (row['machine_type'] == "Loader" and not (110 <= row['power_usage'] <= 640)):
        reasons.append("Electrical Fault")
    if row['downtime_percentage'] > 20:
        reasons.append("Excess Downtime")
    return ", ".join(reasons) if reasons else "None"

st.title("Predictive Maintenance System - SECL")
tabs = st.tabs(["🔍 Manual Input", "📂 Batch Upload", "📊 Visualization & Filter"])

with tabs[0]:
    st.header("🔧 Manual Machine Input")
    
    machine_type = st.selectbox("Machine Type", ["Conveyor belt", "Crusher", "Loader"])
    vibration = st.slider("Vibration (mm/s)", 0.0, 10.0, 2.5)
    temperature = st.slider("Temperature (°C)", 25, 100, 45)
    load = st.slider("Load (T/m)", 0.1, 3.0, 1.0)
    rpm = st.slider("RPM", 500, 3000, 1200)
    sound = st.slider("Sound Level (dB)", 70, 100, 85)
    usage_minutes = st.slider("Usage Minutes", 60, 1440, 800)
    planned_op_time = st.slider("Planned Operating Time (min)", 900, 1440, 1200)
    downtime_minutes = st.slider("Downtime (min)", 0, planned_op_time, 100)
    oil_quality = st.number_input("Oil Quality", value=220)
    power_usage = st.number_input("Power Usage", value=150)
    
    downtime_percentage = (downtime_minutes / planned_op_time) * 100

    input_data = pd.DataFrame([{
        'machine_type': machine_type,
        'vibration': vibration,
        'temperature': temperature,
        'load': load,
        'rpm': rpm,
        'sound': sound,
        'usage_minutes': usage_minutes,
        'planned_operating_time': planned_op_time,
        'downtime_minutes': downtime_minutes,
        'downtime_percentage': downtime_percentage,
        'oil_quality': oil_quality,
        'power_usage': power_usage
    }])

    if st.button("Predict Failure Risk, Type & RUL"):
        risk = risk_model.predict(input_data)[0]
        rul = rul_model.predict(input_data)[0]
        input_data['failure_risk'] = risk
        failure_type = get_failure_type(input_data.iloc[0])

        risk_label = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}.get(risk, "Unknown")

        st.success(f"📉 Failure Risk: {risk_label}")
        st.warning(f"⚠️ Failure Type(s): {failure_type}")
        st.info(f"⏳ Estimated RUL: {int(rul)} minutes")

with tabs[1]:
    st.header("📁 Upload CSV for Batch Prediction")
    csv_file = st.file_uploader("Upload CSV", type=["csv"])
    if csv_file:
        df = pd.read_csv(csv_file)
        df['downtime_percentage'] = df['downtime_minutes'] / df['planned_operating_time'] * 100
        df['failure_risk'] = risk_model.predict(df)
        df['rul'] = rul_model.predict(df)
        df['failure_type'] = df.apply(get_failure_type, axis=1)

        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions as CSV", data=csv, file_name="predicted_output.csv")

with tabs[2]:
    st.header("📊 Visualize & Filter Predictions")
    st.info("Upload a CSV in Batch tab to enable charts and filtering.")

    if 'df' in locals():
        risk_counts = df['failure_risk'].map({0: 'Low', 1: 'Medium', 2: 'High'}).value_counts()
        st.subheader("Risk Level Distribution")
        st.bar_chart(risk_counts)

        filter_option = st.selectbox("Filter", ["All", "Only Risky", "Only High Risk (RUL < 1000)"])
        
        if filter_option == "Only Risky":
            st.dataframe(df[df['failure_risk'] != 0])
        elif filter_option == "Only High Risk (RUL < 1000)":
            st.dataframe(df[(df['failure_risk'] == 2) & (df['rul'] < 1000)])
        else:
            st.dataframe(df)
    else:
        st.warning("Please upload a CSV in the 'Batch Upload' tab first.")
