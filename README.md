💳 Credit Card Fraud Detection System

An end-to-end Machine Learning and FinTech Analytics project designed to detect fraudulent credit card transactions using classification algorithms and deploy predictions through an interactive Streamlit web application.

📌 Project Overview

Credit card fraud is a major challenge in the financial industry, resulting in significant losses for banks, businesses, and customers. This project leverages machine learning techniques to identify potentially fraudulent transactions and provide real-time fraud risk assessment.

The solution covers the complete machine learning lifecycle, from data preprocessing and model training to deployment and user interaction.

//🚀 Features
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

//🛠️ Tech Stack
Programming Language
Python
Data Processing
Pandas
NumPy
Machine Learning
Scikit-learn
Imbalanced-learn (SMOTE)
Random Forest
Logistic Regression
Visualization
Plotly
Matplotlib
Deployment
Streamlit
GitHub
Streamlit Community Cloud

//
// 📂 Project Structure
credit-card-fraud-detection/
│
├── data/
│   ├── creditcard.csv
│   └── cleaned_creditcard.csv
│
├── models/
│   ├── best_fraud_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── screenshots/


//🔄 Machine Learning Workflow
Data Collection
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Scaling
        ↓
Train-Test Split
        ↓
SMOTE Balancing
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Best Model Selection
        ↓
Model Saving
        ↓
Streamlit Deployment


//📊 Dataset

The project uses the Credit Card Fraud Detection dataset containing anonymized transaction features.

Target Variable
Class	Description
0	Legitimate Transaction
1	Fraudulent Transaction
Dataset Challenges
Highly Imbalanced Data
Very Few Fraud Cases
Need for Specialized Evaluation Metrics


//🧹 Data Preprocessing

The following preprocessing steps were performed:

Missing Value Analysis
Duplicate Removal
Feature Scaling using StandardScaler
Feature Selection
Train-Test Split
Class Balancing using SMOTE

//- 🤖 Models Used
Logistic Regression

Used as the baseline model for fraud classification.

Random Forest

Selected as the final model due to stronger fraud detection performance and improved recall.

//📈 Model Evaluation Metrics

The models were evaluated using:

Precision
Recall
F1 Score
Confusion Matrix
ROC-AUC Score

// -Why Recall Matters

In fraud detection, missing fraudulent transactions can be more costly than generating false alarms.

Therefore, Recall was prioritized during model selection.
