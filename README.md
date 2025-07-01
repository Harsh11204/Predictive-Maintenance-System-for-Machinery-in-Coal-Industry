# ⚙️ Predictive Maintenance System for Machinery in Coal Industry

A Streamlit-based Web App to predict **Failure Risk**, **Failure Type**, and **Remaining Useful Life (RUL)** of industrial machines. Built as part of an internship at **SECL (South Eastern Coalfields Limited), IT Department**. The system helps proactively detect risky machinery and supports better maintenance planning in coal operations.

## 📂 Dataset Format (for Upload)

The dataset **must** be in **CSV** format and should contain **at least** the following columns:

| Column Name              | Type    | Description                                            |
| ------------------------ | ------- | ------------------------------------------------------ |
| `machine_type`           | String  | Type of machine (`Conveyor belt`, `Crusher`, `Loader`) |
| `vibration`              | Float   | Vibration level in mm/s                                |
| `temperature`            | Integer | Temperature in °C                                      |
| `load`                   | Float   | Load in T/m                                            |
| `rpm`                    | Integer | Machine speed in RPM                                   |
| `sound`                  | Integer | Sound level in decibels (dB)                           |
| `usage_minutes`          | Integer | Minutes machine was in use                             |
| `planned_operating_time` | Integer | Total planned operation time (in minutes)              |
| `downtime_minutes`       | Integer | Downtime experienced (in minutes)                      |
| `oil_quality`            | Integer | Oil quality indicator (depends on machine type)        |
| `power_usage`            | Integer | Electrical power usage (depends on machine type)       |

## 🎮 Features & Usage

### ✅ 1. Manual Data Input Panel

* Enter real-time values for a single machine.
* Predicts:

  * 📉 Failure Risk (`Low`, `Medium`, `High`)
  * ⚠️ Failure Type(s) (e.g., Overheating, Acoustic Fault)
  * ⏳ Remaining Useful Life (RUL)

### ✅ 2. Batch Data Input Panel

* Upload dataset with multiple machines.
* Returns predictions for all 3 targets.
* 📥 Option to download the results as a CSV file.

### ✅ 3. Filter Machines by Risk Level

* 📊 View pie/bar charts of:

  * Healthy vs Risky machine count
  * Failure type distribution
* Helps identify which failure types are most frequent.

### ✅ 4. Visual Analytics

* Filter machines based on:

  * Only Risky machines (Medium + High)
  * Only High Risk with RUL < threshold (e.g., <1000 mins)
  * View all machines

## ⚙️ Technologies Used

* **Python 3.13.0**
* **Streamlit**
* **Scikit-learn**
* **Pandas, NumPy**
* **Matplotlib, Seaborn**
* **Joblib** (for model persistence)

## 📧 Contact

**Developed by:** Harsh Verma
**Email:** [harshverma11204@gmail.com](mailto:harshverma11204@gmail.com)
**LinkedIn:** [https://www.linkedin.com/in/harsh-verma-1b3857287](https://www.linkedin.com/in/harsh-verma-1b3857287)
