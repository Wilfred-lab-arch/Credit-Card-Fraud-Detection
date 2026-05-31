
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Fraud Detection Intelligence",
    page_icon="🛡️",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_artifacts():
    try:
        pipeline = joblib.load("fraud_pipeline.pkl")

        model = pipeline["model"]
        scaler = pipeline["scaler"]
        features = pipeline["features"]

        return model, scaler, features

    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

model, scaler, FEATURE_COLUMNS = load_artifacts()

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:

    st.title("🛡️ Fraud Radar")

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "💳 Transaction Analyzer",
            "⚡ Risk Engine",
            "📈 Analytics",
            "📄 Reports",
            "ℹ️ About"
        ]
    )

    uploaded_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"]
    )

# ==========================================
# LOAD DATASET
# ==========================================
df = None

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        if "Amount" not in df.columns:
            st.error("Amount column not found.")
            st.stop()

        df["Amount"] = pd.to_numeric(
            df["Amount"],
            errors="coerce"
        )

        df["Amount"] = df["Amount"].fillna(
            df["Amount"].median()
        )

        # Apply same scaling used during training
        df["scaled_amount"] = scaler.transform(
            df[["Amount"]]
        )[:, 0]

        # Drop original Amount and Time
        df.drop(
            columns=["Amount", "Time"],
            inplace=True,
            errors="ignore"
        )

        st.success("Dataset loaded successfully")

    except Exception as e:
        st.error(f"Dataset error: {e}")


# ==========================================
# DASHBOARD
# ==========================================
if page == "📊 Dashboard":

    st.title("📊 Fraud Detection Dashboard")

    if df is None:
        st.info("Upload a dataset to begin.")
    else:

        if "Class" in df.columns:

            fraud = (df["Class"] == 1).sum()
            legit = (df["Class"] == 0).sum()
            total = len(df)
            fraud_percentage = (fraud / total) * 100

            # 1. KPI Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Transactions", f"{total:,}")
            c2.metric("Fraud Cases", f"{fraud:,}", delta=f"{fraud} detected", delta_color="inverse")
            c3.metric("Legitimate", f"{legit:,}")
            c4.metric("Fraud Rate", f"{fraud_percentage:.3f}%")

            # Warning Banner for Class Imbalance
            st.warning(
                f"⚠️ **Extreme Class Imbalance Detected:** Fraudulent cases make up only **{fraud_percentage:.3f}%** "
                "of the dataset. Standard accuracy is misleading here. The system uses specialized thresholding to prioritize Recall."
            )

            # 2. Side-by-Side Visualizations
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                # Donut Chart for general distribution
                fig_pie = px.pie(
                    names=["Legitimate", "Fraud"],
                    values=[legit, fraud],
                    hole=0.6,
                    title="Overall Transaction Mix",
                    color_discrete_sequence=["#2ecc71", "#e74c3c"]
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with chart_col2:
                # Log-scaled Bar Chart to visually expose the tiny fraud class
                fig_bar = px.bar(
                    x=["Legitimate", "Fraud"],
                    y=[legit, fraud],
                    title="Class Volume Comparison (Log Scale)",
                    labels={'x': 'Transaction Type', 'y': 'Count (Log Scale)'},
                    color=["Legitimate", "Fraud"],
                    color_discrete_sequence=["#2ecc71", "#e74c3c"]
                )
                fig_bar.update_layout(yaxis_type="log", showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

        else:
            # Fallback if the uploaded data doesn't have ground-truth 'Class' labels
            st.info("📊 Uploaded dataset does not contain historical 'Class' labels. Use the **Transaction Analyzer** to scan for risks.")


# ==========================================
# TRANSACTION ANALYZER
# ==========================================
elif page == "💳 Transaction Analyzer":

    st.title("💳 Transaction Analysis")

    single_tab, batch_tab = st.tabs(
        ["Single Transaction", "Batch Prediction"]
    )

    with single_tab:

        st.subheader("Single Transaction Analysis")

        cols = st.columns(3)
        values = []

        for i in range(1, 29):
            with cols[(i - 1) % 3]:
                values.append(
                    st.number_input(
                        f"V{i}",
                        value=0.0,
                        format="%.6f",
                        key=f"single_v{i}"
                    )
                )

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=100.0,
            key="single_amount"
        )

        if st.button("Predict Transaction"):

            scaled_amount = scaler.transform(
                pd.DataFrame([[amount]], columns=["Amount"])
            )[0][0]

            input_data = {
                f"V{i}": values[i - 1]
                for i in range(1, 29)
            }

            input_data["scaled_amount"] = scaled_amount

            input_df = pd.DataFrame([input_data])
            input_df = input_df[FEATURE_COLUMNS]

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            risk_score = probability * 100

            st.metric("Fraud Probability", f"{risk_score:.2f}%")

            if prediction == 1:
                st.error("🚨 Fraudulent Transaction")
            else:
                st.success("✅ Legitimate Transaction")

    with batch_tab:

        st.subheader("Batch Fraud Detection")

        batch_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            key="batch_upload"
        )

        if batch_file is not None:

            batch_df = pd.read_csv(batch_file)

            st.write("Preview")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("Run Batch Prediction"):

                batch_df["Amount"] = pd.to_numeric(
                    batch_df["Amount"],
                    errors="coerce"
                )

                batch_df["Amount"] = batch_df["Amount"].fillna(
                    batch_df["Amount"].median()
                )

                batch_df["scaled_amount"] = scaler.transform(
                    batch_df[["Amount"]]
                )[:, 0]

                batch_df.drop(
                    columns=["Amount", "Time"],
                    inplace=True,
                    errors="ignore"
                )

                prediction_df = batch_df[FEATURE_COLUMNS]

                batch_df["Prediction"] = model.predict(prediction_df)

                batch_df["Fraud_Probability"] = model.predict_proba(
                    prediction_df
                )[:, 1]

                batch_df["Risk_Level"] = np.where(
                    batch_df["Fraud_Probability"] >= 0.80,
                    "High",
                    np.where(
                        batch_df["Fraud_Probability"] >= 0.50,
                        "Medium",
                        "Low"
                    )
                )

                c1, c2, c3 = st.columns(3)

                c1.metric("Total", len(batch_df))
                c2.metric("Fraud", (batch_df["Prediction"] == 1).sum())
                c3.metric("Legit", (batch_df["Prediction"] == 0).sum())

                st.dataframe(batch_df.head(100), use_container_width=True)

                csv = batch_df.to_csv(index=False)

                st.download_button(
                    "⬇ Download Results",
                    csv,
                    "fraud_predictions.csv",
                    "text/csv"
                )

# ==========================================
# RISK ENGINE
# ==========================================
elif page == "⚡ Risk Engine":

    st.title("⚡ Risk Engine")

    st.metric(
        "System Status",
        "ACTIVE"
    )

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=72,
            title={
                "text": "Current Risk Level"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                }
            }
        )
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

# ==========================================
# ANALYTICS
# ==========================================
elif page == "📈 Analytics":

    st.title("📈 Analytics")

    if df is None:
        st.info("Upload dataset first.")

    else:

        if "scaled_amount" in df.columns:

            fig = px.histogram(
                df,
                x="scaled_amount",
                nbins=50,
                title="Scaled Amount Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

# ==========================================
# REPORTS
# ==========================================
elif page == "📄 Reports":

    st.title("📄 Reports")

    if df is None:
        st.info("Upload dataset first.")

    else:

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(100),
            use_container_width=True
        )

        csv = df.to_csv(index=False)

        st.download_button(
            label="⬇ Download Dataset",
            data=csv,
            file_name="fraud_report.csv",
            mime="text/csv"
        )

# ==========================================
# ABOUT
# ==========================================
elif page == "ℹ️ About":

    st.title("ℹ️ About")

    st.markdown("""
    ### Fraud Detection Intelligence

    This application uses a Machine Learning model
    trained on credit card transaction data to:

    - Detect fraudulent transactions
    - Analyze transaction risk
    - Visualize fraud trends
    - Generate reports

    **Model Features**
    - V1 – V28
    - scaled_amount

    **Preprocessing**
    - RobustScaler on Amount
    - Amount removed
    - Time removed
    """)








    
