
import os
import csv
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Radar | Fraud Control",
    page_icon="💳",
    layout="wide"
)

# =========================================================
# CUSTOM CSS (Stripe-like Styling)
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

[data-testid="metric-container"] {
    background-color: #111827;
    border: 1px solid #1f2937;
    padding: 15px;
    border-radius: 12px;
}

.stButton>button {
    background-color: #635bff;
    color: white;
    border-radius: 10px;
    border: none;
    height: 3em;
    width: 100%;
    font-weight: 600;
}

.stButton>button:hover {
    background-color: #5145cd;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL REGISTRY
# =========================================================

MODEL_LOG_PATH = os.path.join("models", "prediction_log.csv")
DEFAULT_MODEL_PATH = "fraud_detection_model.pkl"

def format_datetime(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

def get_all_model_files():
    model_files = []

    if os.path.exists(DEFAULT_MODEL_PATH):
        model_files.append(DEFAULT_MODEL_PATH)

    if os.path.isdir("models"):
        for file_name in sorted(os.listdir("models")):
            if file_name.endswith(".pkl"):
                path = os.path.join("models", file_name)
                if path not in model_files:
                    model_files.append(path)

    return model_files

@st.cache_data(ttl=120)
def get_model_registry():
    registry = []

    for path in get_all_model_files():
        try:
            stat = os.stat(path)
            registry.append({
                "name": os.path.splitext(os.path.basename(path))[0],
                "file": path,
                "size_kb": round(stat.st_size / 1024, 1),
                "last_updated": format_datetime(stat.st_mtime),
            })
        except OSError:
            continue

    return registry

@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    return joblib.load(model_path)


def ensure_log_file():
    os.makedirs(os.path.dirname(MODEL_LOG_PATH), exist_ok=True)
    if not os.path.exists(MODEL_LOG_PATH):
        with open(MODEL_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "model_name",
                "mode",
                "prediction",
                "probability",
                "records",
                "details"
            ])


def log_prediction(model_name, mode, prediction, probability, records=1, details=""):
    ensure_log_file()
    timestamp = datetime.now().isoformat(timespec="seconds")

    with open(MODEL_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            model_name,
            mode,
            prediction,
            f"{probability:.5f}" if probability is not None else "",
            records,
            details,
        ])


def load_prediction_log():
    if not os.path.exists(MODEL_LOG_PATH):
        return pd.DataFrame(columns=["timestamp", "model_name", "mode", "prediction", "probability", "records", "details"])

    return pd.read_csv(MODEL_LOG_PATH)

model_registry = get_model_registry()
selected_model_name = st.sidebar.selectbox(
    "Select model",
    [m["name"] for m in model_registry] if model_registry else ["fraud_detection_model"],
)
selected_model = next(
    (m for m in model_registry if m["name"] == selected_model_name),
    {"file": DEFAULT_MODEL_PATH, "name": "fraud_detection_model"},
)
model = load_model(selected_model["file"])

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("💳 Radar")

    st.caption("Fraud Intelligence System")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Transactions",
            "Risk Engine",
            "Reports",
            "Model Tracking",
            "About"
        ]
    )

    st.markdown("---")

    st.success("System Status: Operational")
    st.caption(f"Loaded model: {selected_model_name}")

    st.markdown("---")
    st.caption(f"Loaded model: {selected_model_name}")

    if st.button("Refresh Model List"):
        st.experimental_rerun()

# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.title("Fraud Monitoring Dashboard")

    st.caption("Real-time payment intelligence layer")

    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Volume", "1.2M")
    c2.metric("Fraud Events", "2,341")
    c3.metric("Fraud Rate", "0.19%")
    c4.metric("Model Accuracy", "99.8%")

    st.markdown("---")

    fraud_data = pd.DataFrame({
        "Type": ["Legitimate", "Fraud"],
        "Count": [99800, 200]
    })

    transaction_data = pd.DataFrame({
        "Amount": np.random.normal(120, 40, 1000)
    })

    left, right = st.columns([1, 1])

    with left:

        st.subheader("Fraud Distribution")

        pie_fig = px.pie(
            fraud_data,
            names="Type",
            values="Count",
            hole=0.65
        )

        st.plotly_chart(
            pie_fig,
            use_container_width=True
        )

    with right:

        st.subheader("Transaction Volume")

        hist_fig = px.histogram(
            transaction_data,
            x="Amount",
            nbins=40
        )

        st.plotly_chart(
            hist_fig,
            use_container_width=True
        )

# =========================================================
# TRANSACTIONS PAGE
# =========================================================

elif page == "Transactions":

    st.title("Transaction Inspector")

    st.caption("Single and batch fraud prediction engine")

    tabs = st.tabs([
        "Single Prediction",
        "Batch Prediction"
    ])

    # =====================================================
    # SINGLE PREDICTION
    # =====================================================

    with tabs[0]:

        input_data = []

        cols = st.columns(3)

        for i in range(1, 29):

            with cols[(i - 1) % 3]:

                value = st.number_input(
                    f"V{i}",
                    value=0.0,
                    step=0.0001,
                    format="%.4f",
                    key=f"single_{i}"
                )

                input_data.append(value)

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

        input_data.append(amount)

        if st.button("Run Fraud Analysis"):

            try:

                data = np.array(input_data).reshape(1, -1)

                prediction = model.predict(data)[0]

                probability = None

                if hasattr(model, "predict_proba"):

                    probability = model.predict_proba(data)[0][1]

                st.subheader("Decision Engine")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Fraud Score",
                    f"{probability:.2%}"
                )

                col2.metric(
                    "Decision",
                    "FRAUD" if prediction == 1 else "LEGIT"
                )

                risk = (
                    "HIGH"
                    if probability > 0.7
                    else "MEDIUM"
                    if probability > 0.3
                    else "LOW"
                )

                col3.metric(
                    "Risk Level",
                    risk
                )

                st.progress(float(probability))

                if prediction == 1:

                    st.error(
                        "⚠ High Risk Transaction Detected"
                    )

                log_prediction(
                    selected_model_name,
                    mode="single",
                    prediction=int(prediction),
                    probability=float(probability) if probability is not None else None,
                    records=1,
                    details="Single transaction"
                )
            except Exception as e:

                st.error(f"Prediction Error: {e}")

    # =====================================================
    # BATCH PREDICTION
    # =====================================================

    with tabs[1]:

        st.subheader("Batch Fraud Detection")

        uploaded_file = st.file_uploader(
            "Upload Transaction CSV",
            type=["csv"]
        )

        st.info("""
        CSV must contain:
        V1 to V28 + Amount
        """)

        if uploaded_file is not None:

            try:

                batch_df = pd.read_csv(uploaded_file)

                st.write("Uploaded Dataset Preview")

                st.dataframe(
                    batch_df.head(),
                    use_container_width=True
                )

                required_columns = [
                    f"V{i}" for i in range(1, 29)
                ] + ["Amount"]

                missing = [
                    col for col in required_columns
                    if col not in batch_df.columns
                ]

                if missing:

                    st.error(
                        f"Missing columns: {missing}"
                    )

                else:

                    if st.button(
                        "Run Batch Fraud Detection"
                    ):

                        X_batch = batch_df[
                            required_columns
                        ]

                        predictions = model.predict(
                            X_batch
                        )

                        probabilities = model.predict_proba(
                            X_batch
                        )[:, 1]

                        batch_df["Prediction"] = np.where(
                            predictions == 1,
                            "FRAUD",
                            "LEGIT"
                        )

                        batch_df["Fraud_Probability"] = (
                            probabilities
                        )

                        batch_df["Risk_Level"] = np.where(
                            probabilities > 0.7,
                            "HIGH",
                            np.where(
                                probabilities > 0.3,
                                "MEDIUM",
                                "LOW"
                            )
                        )

                        st.success(
                            "Batch prediction completed"
                        )

                        total = len(batch_df)

                        fraud_cases = (
                            batch_df["Prediction"]
                            == "FRAUD"
                        ).sum()

                        legit_cases = (
                            batch_df["Prediction"]
                            == "LEGIT"
                        ).sum()

                        c1, c2, c3 = st.columns(3)

                        c1.metric(
                            "Total Transactions",
                            total
                        )

                        c2.metric(
                            "Fraud Cases",
                            fraud_cases
                        )

                        c3.metric(
                            "Legitimate Cases",
                            legit_cases
                        )

                        st.dataframe(
                            batch_df.head(20),
                            use_container_width=True
                        )

                        csv = batch_df.to_csv(
                            index=False
                        )

                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name="fraud_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                        log_prediction(
                            selected_model_name,
                            mode="batch",
                            prediction=int(fraud_cases),
                            probability=float(probabilities.mean()),
                            records=int(total),
                            details=f"Batch run: {legit_cases} legit, {fraud_cases} fraud"
                        )

            except Exception as e:

                st.error(f"Error: {e}")

# =========================================================
# RISK ENGINE PAGE
# =========================================================

elif page == "Risk Engine":

    st.title("Risk Engine Analytics")

    st.caption("Fraud pattern intelligence")

    synthetic_data = pd.DataFrame({
        "Amount": np.random.normal(120, 40, 1000),
        "Class": np.random.choice(
            ["Legitimate", "Fraud"],
            1000,
            p=[0.98, 0.02]
        )
    })

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("Amount Distribution")

        fig1 = px.box(
            synthetic_data,
            x="Class",
            y="Amount",
            color="Class"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with c2:

        st.subheader("Fraud Density")

        fig2 = px.histogram(
            synthetic_data,
            x="Amount",
            color="Class",
            nbins=50
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

# =========================================================
# REPORTS PAGE
# =========================================================

elif page == "Reports":

    st.title("Reports & Export")

    st.caption("Audit logs and fraud reports")

    report_df = pd.DataFrame({
        "Transaction_ID": range(1001, 1021),
        "Fraud_Score": np.round(
            np.random.uniform(0, 1, 20),
            2
        ),
        "Decision": np.random.choice(
            ["LEGIT", "FRAUD"],
            20,
            p=[0.95, 0.05]
        ),
        "Amount": np.round(
            np.random.uniform(10, 5000, 20),
            2
        )
    })

    st.dataframe(
        report_df,
        use_container_width=True
    )

    csv = report_df.to_csv(index=False)

    st.download_button(
        label="Export Report",
        data=csv,
        file_name="fraud_report.csv",
        mime="text/csv",
        use_container_width=True
    )

# =========================================================
# MODEL TRACKING PAGE
# =========================================================

elif page == "Model Tracking":

    st.title("Model Tracking")
    st.caption("View available models and prediction usage logs")

    registry_df = pd.DataFrame(model_registry)
    st.subheader("Registered Models")
    st.dataframe(registry_df, use_container_width=True)

    log_df = load_prediction_log()
    st.subheader("Prediction Log")

    if log_df.empty:
        st.info("No prediction usage has been logged yet.")
    else:
        st.metric("Total Logged Entries", len(log_df))
        st.write(log_df.tail(20))

    if not log_df.empty:
        logs_by_model = log_df.groupby("model_name").size().reset_index(name="runs")
        st.subheader("Runs by Model")
        st.dataframe(logs_by_model, use_container_width=True)

    if st.button("Clear Tracking Log"):
        if os.path.exists(MODEL_LOG_PATH):
            os.remove(MODEL_LOG_PATH)
        st.success("Prediction log cleared. Refresh the page.")

# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "About":

    st.title("About This System")

    st.markdown("""
    ### 💳 Fraud Detection Intelligence Platform

    This platform uses Machine Learning models to detect
    fraudulent credit card transactions in real time.

    ### Technologies Used

    - Python
    - Streamlit
    - Scikit-learn
    - Plotly
    - Joblib

    ### Models

    - Logistic Regression
    - Random Forest
    - XGBoost

    ### Features

    - Real-time fraud prediction
    - Batch fraud screening
    - Stripe-style dashboard
    - Risk analytics
    - Fraud scoring engine
    - Exportable reports
    """)
