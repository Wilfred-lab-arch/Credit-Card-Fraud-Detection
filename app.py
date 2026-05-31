
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

        # 1. Handle Raw Data (Contains 'Amount')
        if "Amount" in df.columns:
            df["Amount"] = pd.to_numeric(
                df["Amount"],        
                errors="coerce"
            )

            median_amount = float(df["Amount"].median())
            df["Amount"] = df["Amount"].fillna(median_amount)

            # High-speed NumPy scaling setup
            raw_amounts = df["Amount"].values.reshape(-1, 1)
            df["scaled_amount"] = scaler.transform(raw_amounts)[:, 0]

            # Drop original columns cleanly
            df.drop(
                columns=["Amount", "Time"],
                inplace=True,
                errors="ignore"
            )
            st.success("✅ Raw dataset loaded and scaled successfully!")

        # 2. Handle Pre-Cleaned/Balanced Data (Contains 'scaled_amount' or 'Scaled_Amount')
        elif "scaled_amount" in df.columns or "Scaled_Amount" in df.columns:
            if "Scaled_Amount" in df.columns:
                df.rename(columns={"Scaled_Amount": "scaled_amount"}, inplace=True)
            st.success("✅ Pre-cleaned dataset loaded successfully!")

        # 3. Fail-safe if neither exists
        else:
            st.error("❌ Missing Data Structure: Neither 'Amount' nor 'scaled_amount' column was found in this file.")
            st.stop()

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

            # Dynamic Banner based on the actual distribution
            if fraud_percentage < 5.0:
                st.error(
                    f"⚠️ **Extreme Class Imbalance (Heavily Legitimate):** Fraudulent cases make up only **{fraud_percentage:.3f}%** "
                    "of the dataset. Standard accuracy is misleading here. The system uses specialized thresholding to prioritize Recall."
                )
            elif fraud_percentage > 60.0:
                st.error(
                    f"🚨 **Extreme Critical Spike (Heavily Fraudulent):** An overwhelming **{fraud_percentage:.3f}%** of uploaded transactions "
                    "are flagged as ground-truth Fraud. Your network is currently under massive attack simulation."
                )
            else:
                st.success(
                    f"⚖️ **Balanced Dataset Mix:** Fraudulent cases make up **{fraud_percentage:.3f}%** of this sample. "
                    "This is an optimized testing distribution."
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

    if df is None:
        # If no dataset has been uploaded yet, keep it at a baseline normal state
        system_status = "NORMAL"
        risk_value = 0.0
        status_color = "normal"
        gauge_color = "#2ecc71"  # Clean green
        st.info("ℹ️ Upload a dataset in the sidebar to see live system telemetry.")
    else:
        # Calculate live risk from the uploaded dataset
        if "Class" in df.columns:
            fraud_cases = (df["Class"] == 1).sum()
            total_cases = len(df)
            risk_value = (fraud_cases / total_cases) * 100
        else:
            # Fallback if dataset doesn't have ground truth labels
            risk_value = 15.0  # Simulated nominal baseline

        # Dynamically calculate the risk categories and UI styles
        if risk_value < 5.0:
            system_status = "SECURE (LOW RISK)"
            status_color = "normal"
            gauge_color = "#2ecc71"  # Green
        elif risk_value <= 50.0:
            system_status = "ELEVATED THREAT PROFILE"
            status_color = "off"
            gauge_color = "#f39c12"  # Amber/Yellow
        else:
            system_status = "CRITICAL UNDER ATTACK"
            status_color = "inverse"
            gauge_color = "#e74c3c"  # Red

    # 1. Show dynamic state indicators
    col1, col2 = st.columns(2)
    col1.metric("System Status", system_status, delta=f"{risk_value:.2f}% Threat Index", delta_color=status_color)
    col2.metric("Active Node monitoring", "ONLINE" if df is not None else "STANDBY")

    # 2. Build the live telemetry Gauge
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_value,
            title={
                "text": "Current Live Risk Level (%)",
                "font": {"size": 20}
            },
            number={'valueformat': '.2f', 'suffix': '%'},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "gray"
                },
                "bar": {"color": gauge_color},
                "bgcolor": "rgba(0,0,0,0.05)",
                "steps": [
                    {"range": [0, 5], "color": "rgba(46, 204, 113, 0.1)"},
                    {"range": [5, 50], "color": "rgba(243, 156, 18, 0.1)"},
                    {"range": [50, 100], "color": "rgba(231, 76, 60, 0.1)"}
                ]
            }
        )
    )

    gauge.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        height=350
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


