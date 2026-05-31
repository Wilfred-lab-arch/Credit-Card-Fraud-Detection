import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Radar Fraud Intelligence",
    page_icon="💳",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

[data-testid="metric-container"] {
    background-color: #111827;
    border: 1px solid #1f2937;
    padding: 15px;
    border-radius: 12px;
}

.stButton > button {
    background-color: #635bff;
    color: white;
    border-radius: 10px;
    border: none;
    height: 3em;
    width: 100%;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #5145cd;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL PIPELINE
# =====================================================

@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("fraud_pipeline.pkl")

    model = pipeline["model"]
    scaler = pipeline["scaler"]
    features = pipeline["features"]

    return model, scaler, features


try:
    model, scaler, FEATURE_COLUMNS = load_artifacts()
except Exception as e:
    st.error(f"Failed to load model pipeline: {e}")
    st.stop()

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("💳 Radar")
    st.caption("Fraud Intelligence Platform")

    st.success("🟢 Operational")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Single Prediction",
            "Batch Prediction",
            "Analytics",
            "Reports",
            "About"
        ]
    )

    st.markdown("---")

    uploaded_dataset = st.file_uploader(
        "Upload Dataset",
        type=["csv"]
    )

# =====================================================
# LOAD DATASET
# =====================================================

df = None

if uploaded_dataset is not None:
    try:
        df = pd.read_csv(uploaded_dataset)
    except Exception as e:
        st.error(str(e))

# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title("💳 Radar Fraud Intelligence")
    st.caption("Real-time Fraud Monitoring")

    if df is None:
        st.info("Upload a dataset to view analytics.")

    else:
        fraud_count = (df["Class"] == 1).sum()
        legit_count = (df["Class"] == 0).sum()
        total = len(df)
        fraud_rate = (fraud_count / total) * 100

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Transactions", f"{total:,}")
        c2.metric("Fraud Cases", fraud_count)
        c3.metric("Fraud Rate", f"{fraud_rate:.4f}%")
        c4.metric("Legitimate", legit_count)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            fraud_df = pd.DataFrame({
                "Type": ["Legitimate", "Fraud"],
                "Count": [legit_count, fraud_count]
            })

            fig = px.pie(
                fraud_df,
                names="Type",
                values="Count",
                hole=0.65
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write(df.columns.tolist())
            fig = px.histogram(df, x="scaled_amount")
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# SINGLE PREDICTION
# =====================================================

elif page == "Single Prediction":

    st.title("🔍 Transaction Inspector")
    st.subheader("Single Fraud Prediction")

    values = []

    with st.expander("Enter Transaction Features", expanded=True):

        cols = st.columns(3)

        for i in range(1, 29):

            with cols[(i - 1) % 3]:
                v = st.number_input(
                    f"V{i}",
                    value=0.0,
                    format="%.4f"
                )
                values.append(v)

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=100.0
        )

    if st.button("Run Fraud Analysis"):

        scaled_amount = scaler.transform([[amount]])[0][0]

        # Build safe feature mapping
        input_dict = {}

        for i in range(1, 29):
            input_dict[f"V{i}"] = values[i - 1]

        input_dict["scaled_amount"] = scaled_amount

        input_df = pd.DataFrame([input_dict])[FEATURE_COLUMNS]

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("---")

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={"text": "Fraud Risk"},
                gauge={"axis": {"range": [0, 100]}}
            )
        )

        st.plotly_chart(gauge, use_container_width=True)

        if prediction == 1:
            st.error("🚨 Fraudulent Transaction Detected")
        else:
            st.success("✅ Transaction Approved")

# =====================================================
# BATCH PREDICTION
# =====================================================

elif page == "Batch Prediction":

    st.title("📂 Batch Fraud Detection")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:

        batch_df = pd.read_csv(file)
        st.dataframe(batch_df.head())

        if st.button("Run Batch Prediction"):

            missing = [
                c for c in FEATURE_COLUMNS
                if c not in batch_df.columns
            ]

            if missing:
                st.error(f"Missing columns: {missing}")
                st.stop()

            X = batch_df[FEATURE_COLUMNS]

            pred = model.predict(X)
            prob = model.predict_proba(X)[:, 1]

            batch_df["Prediction"] = np.where(
                pred == 1,
                "FRAUD",
                "LEGIT"
            )

            batch_df["Fraud_Probability"] = prob

            st.success("Analysis Complete")
            st.dataframe(batch_df.head(50))

            csv = batch_df.to_csv(index=False)

            st.download_button(
                "Download Results",
                csv,
                "fraud_predictions.csv",
                "text/csv"
            )

# =====================================================
# ANALYTICS
# =====================================================

elif page == "Analytics":

    st.title("📊 Fraud Analytics")

    if df is None:
        st.warning("Upload dataset first.")

    else:
        analytics_df = df.copy()

        analytics_df["Class"] = analytics_df["Class"].map({
            0: "Legitimate",
            1: "Fraud"
        })

        metric_col = None

        if "scaled_amount" in analytics_df.columns:
            metric_col = "scaled_amount"
        elif "Amount" in analytics_df.columns:
            metric_col = "Amount"

        if metric_col is None:
            st.error("Neither 'Amount' nor 'scaled_amount' exists in the dataset.")
        else:
            fig1 = px.box(
                analytics_df,
                x="Class",
                y=metric_col,
                color="Class"
            )
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.histogram(
                analytics_df,
                x=metric_col,
                color="Class"
            )
            st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# REPORTS
# =====================================================

elif page == "Reports":

    st.title("📄 Reports")

    if df is None:
        st.warning("Upload dataset first.")

    else:
        sample = df.sample(min(50, len(df)))

        missing = [
            c for c in FEATURE_COLUMNS
            if c not in sample.columns
        ]

        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        X = sample[FEATURE_COLUMNS]

        pred = model.predict(X)
        prob = model.predict_proba(X)[:, 1]

        sample["Decision"] = np.where(
            pred == 1,
            "FRAUD",
            "LEGIT"
        )

        sample["Fraud_Score"] = prob

        st.dataframe(sample)

        csv = sample.to_csv(index=False)

        st.download_button(
            "Export Report",
            csv,
            "fraud_report.csv",
            "text/csv"
        )

# =====================================================
# ABOUT
# =====================================================

elif page == "About":

    st.title("ℹ️ About")

    st.markdown("""
### Radar Fraud Intelligence Platform

Features:
- Real-time fraud prediction
- Batch transaction screening
- Risk scoring
- Fraud analytics
- Exportable reports
- Machine Learning powered

### Technology Stack
- Python
- Streamlit
- Scikit-Learn
- XGBoost
- Plotly
- Joblib
""")

    st.subheader("Loaded Model Type")
    st.code(type(model).__name__)