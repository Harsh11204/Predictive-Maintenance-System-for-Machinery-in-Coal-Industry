import streamlit as st
import pandas as pd
import joblib

# Load models and scaler
risk_model = joblib.load("risk_model.pkl")
rul_model = joblib.load("rul_model.pkl")
scaler = joblib.load("predictive_maintenance_scaler.pkl")  # Ensure this matches the actual file name

# Failure detection logic
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

# Streamlit app
st.title("🔧 Predictive Maintenance System for SECL")

tabs = st.tabs(["Manual Input", "Batch Upload", "Visualization"])

# --- Manual Input Tab ---
with tabs[0]:
    st.header("🛠️ Manual Input")

    machine_type = st.selectbox("Machine Type", ["Conveyor belt", "Crusher", "Loader"])
    vibration = st.slider("Vibration", 0.0, 10.0, 2.5)
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
        scaled_input = scaler.transform(input_data)
        risk = risk_model.predict(scaled_input)[0]
        rul = int(rul_model.predict(scaled_input)[0])
        risk_label = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}[risk]
        input_data["risk_level"] = risk_label
        failure_type = get_failure_type(input_data.iloc[0])

        st.success(f"🧠 Risk Level: {risk_label}")
        st.warning(f"⚠️ Failure Type: {failure_type}")
        st.info(f"⏳ Remaining Useful Life: {rul} minutes")

# --- Batch Upload Tab ---
with tabs[1]:
    st.header("📂 Batch CSV Upload & Prediction")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "downtime_percentage" not in df.columns:
            df["downtime_percentage"] = df["downtime_minutes"] / df["planned_operating_time"] * 100

        # Predict using scaled data
        scaled_features = scaler.transform(df)
        df["risk"] = risk_model.predict(scaled_features)
        df["risk_level"] = df["risk"].map({0: "Low Risk", 1: "Medium Risk", 2: "High Risk"})
        df["rul"] = rul_model.predict(scaled_features)
        df["failure_type"] = df.apply(get_failure_type, axis=1)

        st.dataframe(df.head())

        csv_download = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Result CSV", data=csv_download, file_name="predicted_output.csv")

# --- Visualization Tab ---
with tabs[2]:
    st.header("📊 Risk Level Overview")

    if 'df' in locals():
        st.subheader("📈 Risk Distribution")
        st.bar_chart(df["risk_level"].value_counts())

        st.subheader("🔎 Filter Machines")
        choice = st.selectbox("Show:", ["All", "Only Risky", "Only High Risk with RUL < 1000"])

        if choice == "Only Risky":
            st.dataframe(df[df["risk_level"] != "Low Risk"])
        elif choice == "Only High Risk with RUL < 1000":
            st.dataframe(df[(df["risk_level"] == "High Risk") & (df["rul"] < 1000)])
        else:
            st.dataframe(df)
    else:
        st.info("Upload a CSV in the 'Batch Upload' tab to see visualizations.")
