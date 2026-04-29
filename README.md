# 📊 Customer Churn Intelligence Dashboard

A production-ready Machine Learning project that predicts customer churn and provides actionable insights through an interactive web dashboard.

---

## 🚀 Live Demo

👉 *(Add your deployed Streamlit link here)*
Example: https://nishantdubey-churn.streamlit.app

---

## 📌 Problem Statement

Customer churn is a critical challenge for subscription-based businesses.
Retaining customers is significantly more cost-effective than acquiring new ones.

**Goal:**
Predict whether a customer is likely to churn and enable proactive retention strategies.

---

## 🧠 Solution Overview

This project builds an end-to-end ML system:

* Data preprocessing & feature engineering
* Supervised learning (classification)
* Model deployment using Streamlit
* Interactive dashboard with risk scoring

---

## 🏗️ Project Architecture

```bash
customer-churn-project/
│
├── src/
│   ├── app.py              # Streamlit dashboard
│   ├── model.py            # Training pipeline
│
├── models/
│   └── pipeline.pkl        # Trained ML pipeline
│
├── data/
│   └── churn.csv           # Dataset
│
├── assets/
│   ├── logo.png
│   └── profile.jpg
│
├── .streamlit/
│   └── config.toml         # UI theme
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn (Pipeline, Random Forest)
* Streamlit (Dashboard UI)

---

## 🤖 Machine Learning Approach

### Problem Type:

* Binary Classification

### Target Variable:

* `Churn` (Yes / No)

### Model:

* Random Forest Classifier

### Pipeline Includes:

* Missing value handling
* Feature scaling
* One-hot encoding
* Model training

---

## 📊 Key Features

* 📈 Churn probability prediction
* 🚨 Risk classification (High / Medium / Low)
* 💡 Business recommendations
* 📊 Interactive dashboard
* 🎨 SaaS-style UI

---

## 📸 Dashboard Preview

*(Add screenshot here)*

---

## ▶️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/nishant1529/customer-churn-project.git
cd customer-churn-project
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate environment

```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Train model (optional)

```bash
python src/model.py
```

### 6. Run the app

```bash
python -m streamlit run src/app.py
```

---

## 🌍 Deployment

This app is deployed using **Streamlit Cloud**.

To deploy:

1. Push code to GitHub
2. Connect repo to Streamlit Cloud
3. Select `src/app.py`
4. Deploy

---

## 💡 Business Impact

* Identify high-risk customers early
* Reduce churn through targeted interventions
* Improve customer retention strategy

---

## 🔮 Future Improvements

* SHAP explainability (model interpretability)
* Advanced models (XGBoost / LightGBM)
* Real-time data integration
* Customer segmentation

---

## 👤 Author

**Nishant Dubey**

* LinkedIn: https://linkedin.com/in/nishantdubey
* GitHub: https://github.com/nishant1529

---

## ⭐ If You Like This Project

Give it a ⭐ on GitHub and feel free to connect!
