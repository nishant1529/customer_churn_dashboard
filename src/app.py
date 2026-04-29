import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import base64

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent
pipeline = pickle.load(open(BASE_DIR / "models" / "pipeline.pkl", "rb"))

def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logoImg = get_base64_image("assets/D&D.png")

# ---------------- SIDEBAR (PROFILE + INPUTS) ----------------

st.sidebar.markdown(
    f"""
    <div style="text-align:center; margin-top:-20px;">
        <img src="data:image/png;base64,{logoImg}" 
             width="100" 
             style="border-radius:50%; border:2px solid #4CAF50;">
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("## 👤 Nishant Dubey")
st.sidebar.markdown("""
**Tech Enthusiast**  
📧 nishant.dubey@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/nishantdubey)  
""")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Customer Inputs")

# Inputs
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.number_input("Monthly Charges", value=70.0)
total = st.sidebar.number_input("Total Charges", value=1000.0)

contract = st.sidebar.selectbox("Contract", ["Month-to-month","One year","Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL","Fiber optic","No"])

gender = st.sidebar.selectbox("Gender", ["Male","Female"])
partner = st.sidebar.selectbox("Partner", ["Yes","No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes","No"])

phone = st.sidebar.selectbox("Phone Service", ["Yes","No"])
multiple = st.sidebar.selectbox("Multiple Lines", ["Yes","No","No phone service"])
online_sec = st.sidebar.selectbox("Online Security", ["Yes","No","No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes","No","No internet service"])
device = st.sidebar.selectbox("Device Protection", ["Yes","No","No internet service"])
tech = st.sidebar.selectbox("Tech Support", ["Yes","No","No internet service"])
tv = st.sidebar.selectbox("Streaming TV", ["Yes","No","No internet service"])
movies = st.sidebar.selectbox("Streaming Movies", ["Yes","No","No internet service"])
payment = st.sidebar.selectbox("Payment Method", [
    "Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"
])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes","No"])

predict_btn = st.sidebar.button("🚀 Predict Churn")

# ---------------- MAIN DASHBOARD ----------------

st.title("📊 Customer Churn Intelligence Dashboard")
st.markdown("### Predict churn risk and take proactive retention actions")

# Build input
input_df = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "Contract": contract,
    "InternetService": internet,
    "gender": gender,
    "Partner": partner,
    "Dependents": dependents,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "OnlineSecurity": online_sec,
    "OnlineBackup": online_backup,
    "DeviceProtection": device,
    "TechSupport": tech,
    "StreamingTV": tv,
    "StreamingMovies": movies,
    "PaymentMethod": payment,
    "PaperlessBilling": paperless
}])

# ---------------- PREDICTION ----------------

if predict_btn:
    prob = pipeline.predict_proba(input_df)[0][1]
    pred = pipeline.predict(input_df)[0]

    col1, col2, col3 = st.columns(3)

    # KPI Cards
    col1.metric("Churn Probability", f"{prob:.2%}")
    col2.metric("Risk Level", "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low")
    col3.metric("Customer Status", "⚠️ At Risk" if pred else "✅ Safe")

    st.markdown("---")

    # Progress bar
    st.subheader("📈 Risk Score")
    st.progress(float(prob))

    # Recommendation Section
    st.subheader("💡 Recommended Action")

    if prob > 0.7:
        st.error("Immediate retention action required (discount / call)")
    elif prob > 0.4:
        st.warning("Engage customer with offers")
    else:
        st.success("Customer is stable")

    # Simple Visualization
    st.subheader("📊 Risk Breakdown")
    chart_df = pd.DataFrame({
        "Category": ["Churn Risk", "Retention Probability"],
        "Value": [prob, 1-prob]
    })
    st.bar_chart(chart_df.set_index("Category"))