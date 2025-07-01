import streamlit as st
import pandas as pd
import joblib

# --- Load Models ---
risk_model = joblib.load("risk_model.pkl")
rul_model = joblib.load("rul_model.pkl")
type_model = joblib.load("type_model.pkl")
scaler = joblib.load("predictive_maintenance_scaler.pkl")

# --- App Layout ---
st.title("🔧 Predictive Maintenance System for Machineries in Coal Industry")
tabs = st.tabs(["Manual Input", "Batch Upload", "Visualization"])

# --- Tab 1: Manual Input ---
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
        # Scale and predict using raw values
        scaled_input = scaler.transform(input_data)
        X = scaled_input  # already array

        risk = risk_model.predict(X)[0]
        rul = int(rul_model.predict(X)[0])
        failure_type = type_model.predict(X)[0]
        risk_label = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}[risk]

        st.success(f"🧠 Risk Level: **{risk_label}**")
        st.warning(f"⚠️ Failure Type (Model): **{failure_type}**")
        st.info(f"⏳ Remaining Useful Life: **{rul} minutes**")

# --- Tab 2: Batch Upload ---
with tabs[1]:
    st.header("📂 Batch Upload")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "machine_type" not in df.columns:
            st.error("CSV must include 'machine_type' column (as string).")
        else:
            df["downtime_percentage"] = df["downtime_minutes"] / df["planned_operating_time"] * 100

            scaled = scaler.transform(df)
            X = scaled

            df["risk"] = risk_model.predict(X)
            df["risk_level"] = df["risk"].map({0: "Low Risk", 1: "Medium Risk", 2: "High Risk"})
            df["rul"] = rul_model.predict(X)
            df["failure_type"] = type_model.predict(X)

            st.dataframe(df.head())
            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", csv_out, file_name="predicted_output.csv")

# --- Tab 3: Visualization ---
with tabs[2]:
    st.header("📊 Visualization")

    if 'df' in locals():
        st.subheader("📈 Risk Level Distribution")
        st.bar_chart(df["risk_level"].value_counts())

        st.subheader("🔍 Filter Machines")
        filter_option = st.selectbox("Filter", ["All", "Only Risky", "High Risk with RUL < 1000"])

        if filter_option == "Only Risky":
            st.dataframe(df[df["risk_level"] != "Low Risk"])
        elif filter_option == "High Risk with RUL < 1000":
            st.dataframe(df[(df["risk_level"] == "High Risk") & (df["rul"] < 1000)])
        else:
            st.dataframe(df)
    else:
        st.info("Please upload a CSV in the 'Batch Upload' tab to enable visualization.")
