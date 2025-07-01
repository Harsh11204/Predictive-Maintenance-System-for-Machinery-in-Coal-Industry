import streamlit as st
import pandas as pd
import joblib

# --- Load Models and Scaler ---
risk_model = joblib.load("risk_model.pkl")
rul_model = joblib.load("rul_model.pkl")
type_model = joblib.load("type_model.pkl")
scaler = joblib.load("predictive_maintenance_scaler.pkl")

# --- Machine type mapping (used during training) ---
machine_type_mapping = {"Conveyor belt": 0, "Crusher": 1, "Loader": 2}

# --- Feature order (must match training) ---
FEATURE_ORDER = [
    'vibration', 'temperature', 'load', 'rpm', 'sound',
    'usage_minutes', 'planned_operating_time', 'downtime_minutes',
    'oil_quality', 'power_usage', 'machine_type', 'downtime_percentage'
]

# --- App Layout ---
st.title("🔧 Predictive Maintenance System for Machineries in Coal Industry")
tabs = st.tabs(["Manual Input", "Batch Upload", "Filter Machines"])

# ------------------- TAB 1: Manual Input -------------------
with tabs[0]:
    st.header("🛠️ Manual Input")

    machine_type = st.selectbox("Machine Type", list(machine_type_mapping.keys()))
    vibration = st.number_input("Vibration (mm/s)", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
    temperature = st.number_input("Temperature (°C)", min_value=25, max_value=100, value=50)
    load = st.number_input("Load (T/m)", min_value=0.1, max_value=3.0, value=1.2, step=0.1)
    rpm = st.number_input("RPM (revs/min)", min_value=500, max_value=3000, value=1200, step=100)
    sound = st.number_input("Sound (dB)", min_value=70, max_value=110, value=85, step=1)
    usage_minutes = st.number_input("Usage Minutes (min)", min_value=60, max_value=1440, value=600, step=30)
    planned_op = st.selectbox("Planned Operating Time (min)", list(range(900, 1441, 60)))
    downtime = st.number_input("Downtime (min)", min_value=0, max_value=planned_op, value=100, step=10)
    oil_quality = st.number_input("Oil Quality (cSt)", value=220)
    power_usage = st.number_input("Power Usage (W)", value=150)

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
        "oil_quality": oil_quality,
        "power_usage": power_usage,
        "machine_type": machine_type_mapping[machine_type],
        "downtime_percentage": downtime_percentage
    }])

    input_data = input_data[FEATURE_ORDER]

    if st.button("🔍 Predict"):
        try:
            st.write("📊 Model input preview:", input_data)
            scaled_input = scaler.transform(input_data)
            risk_label = risk_model.predict(scaled_input)[0]
            rul = int(rul_model.predict(scaled_input)[0])
            failure_type = type_model.predict(scaled_input)[0]

            st.success(f"🧠 Risk Level: **{risk_label}**")
            st.warning(f"⚠️ Failure Type: **{failure_type}**")
            st.info(f"⏳ Remaining Useful Life: **{rul} minutes**")

        except Exception as e:
            st.error(f"❌ Prediction failed:\n\n{e}")
            st.stop()

# ------------------- TAB 2: Batch Upload + Visualization -------------------
with tabs[1]:
    st.header("📂 Batch Upload")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "machine_type" not in df.columns:
            st.error("CSV must include 'machine_type' column (as string).")
        else:
            df["machine_type"] = df["machine_type"].map(machine_type_mapping)

            if df["machine_type"].isnull().any():
                st.error("CSV contains unknown machine types. Use only: Conveyor belt, Crusher, Loader.")
            else:
                df["downtime_percentage"] = df["downtime_minutes"] / df["planned_operating_time"] * 100

                try:
                    df_model = df[FEATURE_ORDER].copy()
                    scaled = scaler.transform(df_model)

                    df["risk_level"] = risk_model.predict(scaled)
                    df["rul"] = rul_model.predict(scaled).astype(int)
                    df["failure_type"] = type_model.predict(scaled)

                    st.dataframe(df.head())
                    csv_out = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Results", csv_out, "predicted_output.csv")

                    # Save in session for Tab 3
                    st.session_state["batch_df"] = df

                    st.subheader("📊 Risk Distribution")
                    st.bar_chart(df["risk_level"].value_counts())

                except Exception as e:
                    st.error(f"❌ Prediction failed:\n\n{e}")

# ------------------- TAB 3: Filter Machines -------------------
with tabs[2]:
    st.header("🔍 Filter Machines")

    df = st.session_state.get("batch_df")
    if df is not None:
        filter_option = st.selectbox("Filter", [
            "All", 
            "Low Risk Only", 
            "Medium Risk Only", 
            "High Risk Only", 
            "Only Risky (Med + High)", 
            "High Risk with RUL < 1000"
        ])

        if filter_option == "Low Risk Only":
            st.dataframe(df[df["risk_level"] == "Low Risk"])
        elif filter_option == "Medium Risk Only":
            st.dataframe(df[df["risk_level"] == "Medium Risk"])
        elif filter_option == "High Risk Only":
            st.dataframe(df[df["risk_level"] == "High Risk"])
        elif filter_option == "Only Risky (Med + High)":
            st.dataframe(df[df["risk_level"].isin(["Medium Risk", "High Risk"])])
        else:
            st.dataframe(df)
    else:
        st.info("Please upload a CSV in the 'Batch Upload' tab to enable filtering.")
