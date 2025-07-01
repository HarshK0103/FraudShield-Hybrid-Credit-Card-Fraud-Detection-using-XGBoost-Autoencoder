<h1 align="center">💳 FraudShield</h1>
<h3 align="center">🔐 Hybrid Credit Card Fraud Detection using XGBoost + Autoencoder</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python">
  <img src="https://img.shields.io/badge/Machine%20Learning-XGBoost-success?logo=xgboost">
  <img src="https://img.shields.io/badge/Deep%20Learning-Autoencoder-orange?logo=tensorflow">
  <img src="https://img.shields.io/badge/Streamlit-App-red?logo=streamlit">
</p>

---

## 🚀 Overview

**FraudShield** is an intelligent fraud detection system that blends the strengths of **XGBoost** (supervised learning) and **Autoencoder** (unsupervised learning) to identify fraudulent credit card transactions with high accuracy and interpretability.

> 🧠 Inspired by real-world banking scenarios, FraudShield enhances fraud prediction by calculating a **hybrid risk score** that balances classification and anomaly detection.

---

## 🎯 Objectives

- ✅ Detect subtle fraud patterns beyond classical models  
- ✅ Use anomaly reconstruction and classification synergy  
- ✅ Build an interactive, ready-to-test **Streamlit web app**  
- ✅ Provide downloadable results with risk insights  

---

## 📐 Project Architecture

```
Dataset (creditcard.csv)
        │
        ▼
[Data Preprocessing]
 - Standard Scaling (Time, Amount)
 - Class Balancing (optional)
        │
        ├─────────────┬─────────────┐
        ▼                           ▼
[XGBoost Classifier]       [Autoencoder (Anomaly Detection)]
        │                           │
        └───────► Hybrid Risk Scoring ◄────────
                        ▼
               Final Fraud Prediction
```

---

## 🧠 Models Used

### 🔹 XGBoost Classifier
- Trained on labeled fraud data
- Outputs fraud probability
- Great for known patterns


### 🔸 Autoencoder
- Trained only on non-fraud transactions
- Reconstructs data to detect anomalies
- Detects new, unseen fraud types


### ⚖️ Hybrid Risk Score

We compute a hybrid score like this:

```python
Hybrid Score = 0.5 * XGBoost Probability + 0.5 * Autoencoder Anomaly
```

Fraud prediction is positive if the hybrid score > 50%.

---

## 🧪 Dataset Info

- 📦 Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 📊 284,807 transactions, 492 frauds (~0.17%)
- 🔢 Features: `Time`, `V1`–`V28`, `Amount`, `Class`
- 🚫 Only a small `sample.csv` is included for testing

---

## 📐 Project Structure

```
FraudShield/
│
├── app/
│   ├── app.py                          # Streamlit app logic
│   ├── xgb_model.pkl                   # Trained XGBoost classifier
│   ├── autoencoder_model.keras         # Trained Autoencoder model
│   └── scaler.pkl                      # StandardScaler for preprocessing
│
├── models/
│   ├── xgb_model.pkl
│   ├── autoencoder_model.keras
│   └── scaler.pkl
│
├── notebook/
│   └── Credit card Fraud.ipynb         # Training and evaluation notebook
│
├── data/
│   └── sample.csv                      # Example test file for predictions
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/FraudShield.git
   cd FraudShield
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app/app.py
   ```

4. **Upload test data**  
   Use the `sample.csv` from `/data/` or your own formatted file.

> ⚠️ Full training dataset is not included due to size. You can download it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).


---

## 🖥️ Streamlit Web App

### 🔹 Features

- 📁 Upload `.csv` file of transactions  
- 📊 View hybrid fraud risk per row  
- 📉 View histogram of risk scores  
- 🧾 Download predictions as CSV  
- ✅ Easy to test using `sample.csv`  


---

## 📈 Results

- ✔️ Classification report from XGBoost  
- ✔️ Anomaly scores from Autoencoder  
- ✔️ Hybrid model combines both for stronger generalization  
- ✔️ Visual output: histogram + result table  

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).  
Feel free to use, modify, and distribute it with credit.

---

## 🙋‍♂️ Author

**Harsh Karekar**  
B.Tech in ECE | ML & AI Enthusiast | VIT Bhopal  
📫 [LinkedIn](https://www.linkedin.com/in/harsh-karekar-01h6910a04/) | 💻 [GitHub](https://github.com/HarshK0103)

---

## ⭐️ Show Your Support!

If you found this project helpful, please consider giving it a ⭐️ on GitHub — it really motivates me to build more!

---
