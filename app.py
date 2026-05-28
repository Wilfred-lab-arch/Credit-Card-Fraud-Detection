

import os
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
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

body {
    background-color: #0f172a;
    color: #f8fafc;
}

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
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# FEATURE COLUMNS
# =========================================================

FEATURE_COLUMNS = [
    f"V{i}" for i in range(1, 29)
] + ["scaled_amount"]

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_path = "fraud_detection_model.pkl"

    if not os.path.exists(model_path):

        st.error(
            f"Model file not found: {model_path}"
        )

        st.stop()

    return joblib.load(model_path)

model = load_model()

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
            "About"
        ]
    )

    st.markdown("---")

    st.subheader("Upload Dataset")

    uploaded_dataset = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"]
    )

    st.markdown("---")

    st.success("System Status: Operational")

# =========================================================
# LOAD DATASET
# =========================================================

df = None

if uploaded_dataset is not None:

    try:

        df = pd.read_csv(uploaded_dataset)

        required_columns = (
            [f"V{i}" for i in range(1, 29)]
            + ["scaled_amount", "Class"]
        )

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:

            st.error(
                f"Missing dataset columns: "
                f"{missing_columns}"
            )

            st.stop()

        # =================================================
        # METRICS
        # =================================================

        fraud_count = (
            df["Class"] == 1
        ).sum()

        legit_count = (
            df["Class"] == 0
        ).sum()

        total_transactions = len(df)

        fraud_rate = (
            fraud_count / total_transactions
        ) * 100

    except Exception as e:

        st.error(f"Dataset Error: {e}")

# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.title("Fraud Monitoring Dashboard")

    st.caption(
        "Real-time payment intelligence layer"
    )

    if df is None:

        st.warning(
            "Please upload a fraud dataset "
            "from the sidebar."
        )

        st.stop()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    c2.metric(
        "Fraud Cases",
        f"{fraud_count:,}"
    )

    c3.metric(
        "Fraud Rate",
        f"{fraud_rate:.4f}%"
    )

    c4.metric(
        "Legitimate",
        f"{legit_count:,}"
    )

    st.markdown("---")

    fraud_data = pd.DataFrame({
        "Type": [
            "Legitimate",
            "Fraud"
        ],
        "Count": [
            legit_count,
            fraud_count
        ]
    })

    transaction_data = df[["scaled_amount"]]

    left, right = st.columns(2)

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

        st.subheader(
            "Scaled Amount Distribution"
        )

        hist_fig = px.histogram(
            transaction_data,
            x="scaled_amount",
            nbins=50
        )

        st.plotly_chart(
            hist_fig,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

# =========================================================
# TRANSACTIONS PAGE
# =========================================================

elif page == "Transactions":

    st.title("Transaction Inspector")

    st.caption(
        "Single and batch fraud prediction engine"
    )

    tabs = st.tabs([
        "Single Prediction",
        "Batch Prediction"
    ])

    # =====================================================
    # SINGLE PREDICTION
    # =====================================================

    with tabs[0]:

        st.subheader("Single Fraud Prediction")

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

        scaled_amount = st.number_input(
            "scaled_amount",
            value=0.0000,
            step=0.0001,
            format="%.4f"
        )

        input_data.append(scaled_amount)

        if st.button("Run Fraud Analysis"):

            try:

                data = pd.DataFrame(
                    [input_data],
                    columns=FEATURE_COLUMNS
                )

                prediction = model.predict(
                    data
                )[0]

                probability = 0.0

                if hasattr(
                    model,
                    "predict_proba"
                ):

                    probability = (
                        model.predict_proba(
                            data
                        )[0][1]
                    )

                st.subheader(
                    "Decision Engine"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Fraud Score",
                    f"{probability:.2%}"
                )

                c2.metric(
                    "Decision",
                    (
                        "FRAUD"
                        if prediction == 1
                        else "LEGIT"
                    )
                )

                risk = (
                    "HIGH"
                    if probability > 0.7
                    else "MEDIUM"
                    if probability > 0.3
                    else "LOW"
                )

                c3.metric(
                    "Risk Level",
                    risk
                )

                st.progress(
                    float(probability)
                )

                if prediction == 1:

                    st.error(
                        "⚠ High Risk "
                        "Transaction Detected"
                    )

                else:

                    st.success(
                        "✓ Transaction Approved"
                    )

            except Exception as e:

                st.error(
                    f"Prediction Error: {e}"
                )

    # =====================================================
    # BATCH PREDICTION
    # =====================================================

    with tabs[1]:

        st.subheader("Batch Fraud Detection")

        uploaded_file = st.file_uploader(
            "Upload Batch CSV",
            type=["csv"]
        )

        st.info("""
Required Columns:
V1 → V28 + scaled_amount
""")

        if uploaded_file is not None:

            try:

                batch_df = pd.read_csv(
                    uploaded_file
                )

                required_batch_columns = (
                    [f"V{i}" for i in range(1, 29)]
                    + ["scaled_amount"]
                )

                missing_columns = [
                    col
                    for col in required_batch_columns
                    if col not in batch_df.columns
                ]

                if missing_columns:

                    st.error(
                        f"Missing columns: "
                        f"{missing_columns}"
                    )

                else:

                    st.dataframe(
                        batch_df.head(),
                        use_container_width=True
                    )

                    if st.button(
                        "Run Batch Fraud Detection"
                    ):

                        X_batch = batch_df[
                            FEATURE_COLUMNS
                        ]

                        predictions = (
                            model.predict(
                                X_batch
                            )
                        )

                        probabilities = (
                            model.predict_proba(
                                X_batch
                            )[:, 1]
                        )

                        batch_df["Prediction"] = (
                            np.where(
                                predictions == 1,
                                "FRAUD",
                                "LEGIT"
                            )
                        )

                        batch_df[
                            "Fraud_Probability"
                        ] = probabilities

                        batch_df["Risk_Level"] = (
                            np.where(
                                probabilities > 0.7,
                                "HIGH",
                                np.where(
                                    probabilities > 0.3,
                                    "MEDIUM",
                                    "LOW"
                                )
                            )
                        )

                        st.success(
                            "Batch prediction completed"
                        )

                        total = len(batch_df)

                        fraud_cases = (
                            batch_df[
                                "Prediction"
                            ] == "FRAUD"
                        ).sum()

                        legit_cases = (
                            batch_df[
                                "Prediction"
                            ] == "LEGIT"
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
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name=(
                                "fraud_predictions.csv"
                            ),
                            mime="text/csv",
                            use_container_width=True
                        )

            except Exception as e:

                st.error(f"Error: {e}")

# =========================================================
# RISK ENGINE PAGE
# =========================================================

elif page == "Risk Engine":

    st.title("Risk Engine Analytics")

    st.caption(
        "Fraud pattern intelligence"
    )

    if df is None:

        st.warning(
            "Please upload a dataset."
        )

        st.stop()

    analytics_df = df.copy()

    analytics_df["Class"] = (
        analytics_df["Class"].map({
            0: "Legitimate",
            1: "Fraud"
        })
    )

    c1, c2 = st.columns(2)

    with c1:

        st.subheader(
            "Scaled Amount Distribution"
        )

        fig1 = px.box(
            analytics_df,
            x="Class",
            y="scaled_amount",
            color="Class"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with c2:

        st.subheader(
            "Fraud Density"
        )

        fig2 = px.histogram(
            analytics_df,
            x="scaled_amount",
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

    st.caption(
        "Audit logs and fraud reports"
    )

    if df is None:

        st.warning(
            "Please upload a dataset."
        )

        st.stop()

    report_df = df.sample(
        min(20, len(df))
    ).copy()

    X_report = report_df[
        FEATURE_COLUMNS
    ]

    report_probabilities = (
        model.predict_proba(
            X_report
        )[:, 1]
    )

    report_predictions = (
        model.predict(
            X_report
        )
    )

    report_df["Fraud_Score"] = np.round(
        report_probabilities,
        4
    )

    report_df["Decision"] = np.where(
        report_predictions == 1,
        "FRAUD",
        "LEGIT"
    )

    report_df = report_df[
        [
            "scaled_amount",
            "Decision",
            "Fraud_Score"
        ]
    ]

    st.dataframe(
        report_df,
        use_container_width=True
    )

    csv = report_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Export Report",
        data=csv,
        file_name="fraud_report.csv",
        mime="text/csv",
        use_container_width=True
    )

# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "About":

    st.title("About This System")

    st.markdown("""
### 💳 Fraud Detection Intelligence Platform

This platform uses Machine Learning
models to detect fraudulent credit
card transactions in real time.

### Technologies Used

- Python
- Streamlit
- Scikit-learn
- Plotly
- Joblib

### Features

- Real-time fraud prediction
- Batch fraud screening
- Dataset uploads
- Fraud analytics
- Risk scoring
- Exportable reports
- Interactive dashboard

### Supported Dataset Format

Dataset must contain:

- V1 → V28
- scaled_amount
- Class
""")

