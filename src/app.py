import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import base64
import json
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from utils import apply_theme, tooltip

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Churn Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1, h2, h3 {
    font-weight: 600;
}
.metric-card {
    background-color: #161B22;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD ----------------
BASE_DIR = Path(__file__).resolve().parent.parent

pipeline = pickle.load(open(BASE_DIR / "models" / "pipeline.pkl", "rb"))
metrics = json.load(open(BASE_DIR / "models" / "metrics.json"))
results_df = pd.read_csv(BASE_DIR / "models" / "model_comparison.csv")
X_test, y_test = pickle.load(open(BASE_DIR / "models" / "test_data.pkl", "rb"))
df = pd.read_csv(BASE_DIR / "data" / "Customer_Churn_Datasheet.csv")

# SIDEBAR
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logoImg = get_base64_image("assets/D&D.png")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# ---------------- HEADER ----------------
st.title("🚀 Churn Intelligence")
st.markdown("### 📊 Customer Retention Analytics Platform")

st.markdown("""
### **🧠Executive Insight:**  
Customers with low tenure and month-to-month contracts show the highest churn probability.  
Early intervention can significantly reduce revenue leakage.
""")

# ---------------- KPI ROW ----------------
k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Customers", len(df))
k2.metric("Churn Rate", f"{(df['Churn']=='Yes').mean():.2%}")
k3.metric("Model Accuracy", f"{metrics['accuracy']:.2%}")
k4.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")

st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.markdown(
    f"""
    <div style="text-align:center; margin-top:-20px;">
        <img src="data:image/png;base64,{logoImg}" 
             width="50" 
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

st.sidebar.header("⚙️ Model Controls")

threshold = st.sidebar.slider("💡Decision Threshold", 0.1, 0.9, 0.5)
predict_btn = st.sidebar.button("🚀 Run Prediction")

apply_theme()
# ---------------- INPUT SECTION ----------------
st.subheader("📥 Customer Profile")

c1, c2, c3 = st.columns(3)

with c1:
    tenure = st.slider("Tenure", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
    gender = st.selectbox("Gender", ["Male","Female"])

with c2:
    monthly = st.number_input("Monthly Charges", 10.0)
    internet = st.selectbox("Internet", ["DSL","Fiber optic","No"])
    partner = st.selectbox("Partner", ["Yes","No"])

with c3:
    total = st.number_input("Total Charges", 10.0)
    dependents = st.selectbox("Dependents", ["Yes","No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"
    ])

# Hidden but required fields
phone = "Yes"
multiple = "No"
online_sec = "No"
online_backup = "No"
device = "No"
tech = "No"
tv = "No"
movies = "No"
paperless = "Yes"

# ---------------- INPUT DF ----------------
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

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs([
    "🔍 Prediction",
    "💡 Drivers",
    "📊 Model Performance"
])

# ---------------- TAB 1 ----------------
with tab1:

    if predict_btn:

        prob = pipeline.predict_proba(input_df)[0][1]
        pred = int(prob >= threshold)

        c1, c2, c3 = st.columns(3)

        c1.metric("Churn Probability", f"{prob:.2%}")
        c2.metric("Risk Segment",
                  "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low")
        c3.metric("Decision",
                  "⚠️ At Risk" if pred else "✅ Safe")

        st.progress(prob)

        st.markdown("### 💡 Recommendation")

        if prob > 0.7:
            st.error("Immediate intervention recommended")
        elif prob > 0.4:
            st.warning("Targeted engagement advised")
        else:
            st.success("No action required")

# ---------------- TAB 2 ----------------
with tab2:

    st.subheader("💡 Key Drivers of Churn")

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_

    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    fig = px.bar(
        feat_df,
        x="Importance",
        y="Feature",
        orientation="h",
        hover_data=["Importance"],
        title="Feature Importance"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **🧠 Insights:**
    - 📉 Contract type strongly influences churn  
    - ⏳ Lower tenure correlates with higher churn  
    - 💰 Pricing impacts retention behavior  
    """)

    st.markdown("---")

    # ---------------- CONFUSION MATRIX ----------------
    st.subheader("Confusion Matrix")

    cm = metrics["confusion_matrix"]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted No", "Predicted Yes"],
        y=["Actual No", "Actual Yes"],
        colorscale="Blues",
        showscale=True,
        hovertemplate="Value: %{z}<extra></extra>"
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- ROC CURVE ----------------
    st.subheader("ROC Curve")

    y_prob = pipeline.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random'))

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---------------- CHURN DISTRIBUTION ----------------
    st.subheader("Churn Distribution")

    fig = px.pie(df, names="Churn", title="Churn Distribution", hover_data=["Churn"])
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Gender Distribution  ----------------
    st.subheader("Churn Distribution by Gender")

    fig = px.histogram(df, x="gender", color="Churn", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Payment Method Distribution  ----------------
    st.subheader("Churn Distribution by Payment Methods")

    fig = px.histogram(df, x="PaymentMethod", color="Churn", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- CONTRACT VS CHURN ----------------
    st.subheader("Contract vs Churn")

    fig = px.histogram(df, x="Contract", color="Churn", barmode="group",hover_data=df.columns)
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- TENURE ----------------
    st.subheader("Tenure Distribution")

    fig = px.histogram(df, x="tenure", color="Churn", nbins=30)
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- INTERNET SERVICE ----------------
    st.subheader("Internet Service vs Churn")

    fig = px.histogram(df, x="InternetService", color="Churn", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- CHARGES ----------------
    st.subheader("Monthly Charges vs Churn")

    fig = px.box(df, x="Churn", y="MonthlyCharges")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Relationship between the Total Charges & Churn ----------------
    st.subheader("Relationship between the Total Charges & Churn")

    fig = px.histogram(df, x="TotalCharges", color="Churn", nbins=30, marginal="box")   # adds extra distribution insight
    st.plotly_chart(fig, use_container_width=True)

    clean_df = df.dropna(subset=["TotalCharges"])
    hist_data = [
        clean_df[clean_df["Churn"] == "Yes"]["TotalCharges"],
        clean_df[clean_df["Churn"] == "No"]["TotalCharges"]
    ]

    group_labels = ["Churn", "No Churn"]

    fig = ff.create_distplot(hist_data, group_labels, show_hist=True)
    st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 3 ----------------
with tab3:

    st.subheader("💰 Business Impact Simulator")

    cost_fp = st.slider("Cost of False Positive ($)", 0, 500, 50)
    cost_fn = st.slider("Cost of False Negative ($)", 0, 1000, 500)

    cm = metrics["confusion_matrix"]

    fp = cm[0][1]
    fn = cm[1][0]

    total_cost = fp * cost_fp + fn * cost_fn

    st.metric("Total Estimated Cost", f"${total_cost:,}")

    st.info("Reducing missed churn customers has highest financial impact")

    st.subheader("📈 Model Comparison")

    st.dataframe(results_df)

    fig = px.bar(
        results_df,
        x="Model",
        y=["Accuracy","Precision","Recall","F1 Score"],
        barmode="group"
    )

    st.plotly_chart(fig, use_container_width=True)