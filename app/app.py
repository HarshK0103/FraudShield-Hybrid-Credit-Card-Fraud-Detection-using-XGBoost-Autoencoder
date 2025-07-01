import os
import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Set your base directory
base_dir = os.getcwd()

# File paths
model_path = os.path.join(base_dir, 'xgb_model.pkl')
auto_path = os.path.join(base_dir, 'autoencoder_model.keras')
scaler_path = os.path.join(base_dir, 'scaler.pkl')

# Load models and scaler
xgb_model = joblib.load(model_path)
autoencoder = load_model(auto_path)
scaler = joblib.load(scaler_path)

# Streamlit UI
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection")
st.write("Upload a CSV file with credit card transactions to detect potential fraud.")

uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])

# Define required features
expected_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.subheader("🔎 Data Preview")
        st.dataframe(df.head())

        # Validate required columns
        missing_cols = [col for col in expected_columns if col not in df.columns]
        extra_cols = [col for col in df.columns if col not in expected_columns + ['Class']]

        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.stop()

        if extra_cols:
            st.warning(f"⚠️ Extra columns detected: {extra_cols}. They will be ignored.")

        X = df[expected_columns].copy()
        original = X.copy()

        # Scale and rename Time & Amount
        if 'Time' in X.columns and 'Amount' in X.columns:
            X['scaled_time'] = scaler.transform(X[['Time']])
            X['scaled_amount'] = scaler.transform(X[['Amount']])
            X = X.drop(columns=['Time', 'Amount'])

        # Reorder columns as per training data
        expected_columns = ['scaled_time', 'scaled_amount'] + [f'V{i}' for i in range(1, 29)]
        X = X[expected_columns]

        # XGBoost prediction
        xgb_probs = xgb_model.predict_proba(X)[:, 1]

        # Autoencoder reconstruction error
        reconstructions = autoencoder.predict(X)
        mse = np.mean(np.square(X - reconstructions), axis=1)
        mse_scaled = (mse - mse.min()) / (mse.max() - mse.min())

        # Hybrid Risk Score
        risk_score = ((xgb_probs * 0.5) + (mse_scaled * 0.5)) * 100
        risk_score = np.round(risk_score, 2)

        # Final prediction
        hybrid_preds = (risk_score > 50).astype(int)

        # Prepare results
        result_df = original.copy()
        result_df["Fraud Probability (%)"] = np.round(xgb_probs * 100, 2)
        result_df["Anomaly Score (%)"] = np.round(mse_scaled * 100, 2)
        result_df["Hybrid Risk Score (%)"] = risk_score
        result_df["Prediction"] = hybrid_preds

        st.subheader("📊 Prediction Results Table")
        st.dataframe(result_df)

        # Download button
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", data=csv, file_name="fraud_detection_results.csv", mime='text/csv')

        # Summary Metrics
        total_txns = len(result_df)
        total_frauds = result_df['Prediction'].sum()
        fraud_percent = (total_frauds / total_txns) * 100

        st.subheader("📊 Summary Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", total_txns)
        col2.metric("Predicted Frauds", total_frauds)
        col3.metric("Fraud %", f"{fraud_percent:.2f}%")

        # Pie Chart
        st.subheader("🧩 Fraud vs Non-Fraud Pie Chart")
        fig1, ax1 = plt.subplots()
        labels = ['Non-Fraud', 'Fraud']
        sizes = [total_txns - total_frauds, total_frauds]
        colors = ['#00cc96', '#ef553b']
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax1.axis('equal')
        st.pyplot(fig1)

        # Histogram of Risk Scores
        st.subheader("📈 Distribution of Hybrid Risk Scores")
        fig2, ax2 = plt.subplots()
        ax2.hist(risk_score, bins=30, color='#636EFA', edgecolor='black')
        ax2.set_xlabel("Hybrid Risk Score (%)")
        ax2.set_ylabel("Number of Transactions")
        ax2.set_title("Risk Score Distribution")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"⚠️ Error while processing: {str(e)}")

else:
    st.info("📁 Please upload a valid CSV file to begin.")