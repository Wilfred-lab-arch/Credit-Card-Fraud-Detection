# 💳 Credit Card Fraud Detection

An end-to-end machine learning project that detects fraudulent credit card transactions and exposes predictions via a Streamlit web app.

---



# -📌 Project Overview

Credit card fraud is a major challenge in the financial industry, resulting in significant losses for banks, businesses, and customers. This project leverages machine learning techniques to identify potentially fraudulent transactions and provide real-time fraud risk assessment.

The solution covers the complete machine learning lifecycle, from data preprocessing and model training to deployment and user interaction.

---
#🚀 Features

Machine Learning
Data Cleaning and Preprocessing
Feature Scaling
Class Imbalance Handling using SMOTE
Logistic Regression Model
Random Forest Model
Model Comparison and Evaluation
Fraud Probability Prediction
Fraud Detection Analytics
Fraud Risk Scoring
Behavioral Fraud Detection
Detection of Multiple Large Transactions within Short Intervals
Fraud Trend Visualization
Transaction Monitoring Dashboard
Web Application
Interactive Streamlit Dashboard
Single Transaction Prediction
Batch CSV Upload and Prediction
Download Prediction Results
Fraud Analytics Dashboard
User-Friendly FinTech Interface

# -🛠️ Tech Stack

Programming Language
Python
Data Processing
Pandas
NumPy
Machine Learning
Scikit-learn
Imbalanced-learn (SMOTE)
Random Forest


**Key features**
- Single-transaction prediction and batch CSV prediction
- Streamlit dashboard for visualization and monitoring
- Data preprocessing pipeline (scaling, duplicate removal)
- Class imbalance handling with SMOTE
- Model training and evaluation (Logistic Regression, Random Forest)
- Saved model for inference (`fraud_pipeline.pkl`)

---

## Quick Start

Prerequisites
- Python 3.10+ (recommended)
- A virtual environment (optional but recommended)

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the Streamlit app locally:

```powershell
streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
```

Open the forwarded port `8501` in your browser to view the dashboard.

---

## Files and Structure
- `app.py` — Streamlit application and inference endpoints
- `fraud_pipeline.pkl` — Serialized preprocessing + model pipeline used by the app
- `Data/` — dataset files and notebooks
  - `creditcard.csv` — original dataset (if present)
  - `cleaned_creditcard_data.csv` — cleaned dataset used for training
  - `CREDIT CARD FRAUD DETECTION - 2.ipynb` — exploratory notebook
- `.devcontainer/` — local devcontainer config (ignored by git)
- `requirements.txt` — Python dependencies

---

## Data

This project uses the anonymized credit card transactions dataset with a `Class` target where `0` = legitimate and `1` = fraud. The dataset is highly imbalanced, so evaluation emphasizes recall and other robust metrics.

If you need to retrain the model, use the notebook in `Data/` to reproduce preprocessing and training steps.

---

## How it works (high level)
1. Load and clean the dataset
2. Scale features and apply SMOTE to address class imbalance
3. Train models and evaluate (precision, recall, F1, ROC-AUC)
4. Save best model pipeline to `fraud_pipeline.pkl`
5. `app.py` loads the pipeline for inference and provides a Streamlit UI for predictions and analytics

---

## Notes on Deployment & Security
- The devcontainer is for local development only and is ignored by git.
- When deploying publicly, do not disable CORS/XSRF protections; the flags used above are convenient for local development only.

---

## Contributing
- Open an issue or submit a pull request with improvements.

---

## License & Contact
Specify your license here (e.g. MIT) and add contact info or a project owner email if desired.

---

If you'd like, I can also add a short `CONTRIBUTING.md` and a license file. 
