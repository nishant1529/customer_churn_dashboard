import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import pickle
from pathlib import Path
import base64
import json
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from utils import apply_theme, tooltip
import streamlit.components.v1 as components  # type: ignore

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

st.markdown("""
    <style>
    div[data-testid="stTabs"] button {
        font-size: 18px !important;
        padding: 10px 20px !important;
        border-radius: 8px !important;
        margin-right: 8px;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
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
report_path = BASE_DIR / "assets" / "customer_churn_report.html"   # adjust if needed
confusion_matrices = json.load(open(BASE_DIR / "models" / "confusion_matrices.json"))
business_costs = json.load(open(BASE_DIR / "models" / "business_costs.json"))
lift_table = pd.read_json(BASE_DIR / "models" / "lift_table.json")

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
with st.expander("📥 Customer Profile", expanded=False):

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
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Prediction",
    "💡 Insight",
    "📊 Model Performance",
    "📄 Report"
])

# ---------------- TAB 1 ----------------
with tab1:

    if predict_btn:
        st.info("""
            📌 **Threshold Strategy (0.3):**
            Lower threshold is used to capture more churners.
            Missing a churner is more costly than a false alarm.
        """)

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

    # ================= EXECUTIVE SUMMARY =================
    st.markdown("""
    ### 📌 Executive Summary

    - Churn is primarily driven by **contract type, payment method, and tenure**
    - **Early-stage customers** are most at risk
    - **Auto-payment and long-term contracts** improve retention
    """)

    # ================= BUSINESS CONTEXT =================
    st.markdown("""
    ### 🎯 Business Metric Focus

    - **Recall is critical** → missing churners = revenue loss  
    - Precision is secondary → false alerts are cheaper  
    - Model optimized to identify **high-risk customers early**
    """)

    # ================= CLASS DISTRIBUTION =================
    st.markdown("---")
    st.subheader("📊 Class Distribution")

    fig = px.histogram(df, x="Churn", title="Customer Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    - Dataset is imbalanced (~65% retained, ~35% churn)
    - Model uses **class_weight='balanced'**
    """)

    # ================= MODEL PERFORMANCE =================
    st.markdown("---")
    st.subheader("📊 Model Performance")

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")

    # Confusion Matrix
    cm = metrics["confusion_matrix"]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted No", "Predicted Yes"],
        y=["Actual No", "Actual Yes"],
        colorscale="Blues",
        hovertemplate="Value: %{z}<extra></extra>"
    ))
    st.plotly_chart(fig, use_container_width=True)

    # ROC Curve
    y_prob = pipeline.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Baseline'))

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("📌 Threshold tuned to 0.3 → improves recall and captures more churners")

    # ================= FEATURE IMPORTANCE =================
    st.markdown("---")
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
        title="Top Drivers of Churn"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **🧠 Insights:**
    - 📉 Contract type strongly influences churn  
    - ⏳ Lower tenure correlates with higher churn  
    - 💰 Pricing impacts retention behavior  
    """)

    # ================= CUSTOMER BEHAVIOR =================
    st.markdown("---")
    st.markdown("## 🔍 Customer Behavior Drivers")

    # 1️⃣ Overall churn
    st.subheader("📊 Churn Distribution")

    fig = px.pie(df, names="Churn")
    st.plotly_chart(fig, use_container_width=True)

    # 2️⃣ Contract (MOST IMPORTANT)
    st.subheader("📉 Contract vs Churn")

    fig = px.histogram(df, x="Contract", color="Churn", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insight:** Month-to-month customers churn the most  
    **Action:** Promote long-term contracts
    """)

    # 3️⃣ Tenure
    st.subheader("⏳ Tenure Distribution")

    fig = px.histogram(df, x="tenure", color="Churn", nbins=30)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insight:** Early-stage customers churn more  
    **Action:** Improve onboarding experience
    """)

    # 4️⃣ Payment
    st.subheader("💳 Payment Method vs Churn")

    fig = px.histogram(df, x="PaymentMethod", color="Churn", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insight:** Electronic check users churn more  
    **Action:** Encourage auto-pay adoption
    """)

    # 5️⃣ Charges
    st.subheader("💰 Charges vs Churn")

    fig = px.box(df, x="Churn", y="MonthlyCharges")
    st.plotly_chart(fig, use_container_width=True)

    # 6️⃣ Gender (least important)
    st.subheader("👥 Gender vs Churn")

    fig = px.histogram(df, x="gender", color="Churn", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("👉 Gender is NOT a strong driver")

    # ================= CORRELATION (LAST) =================
    st.markdown("---")
    st.subheader("🔗 Feature Correlation Heatmap")

    corr = df.select_dtypes(include="number").corr()

    fig = px.imshow(corr, text_auto=True, color_continuous_scale="Purples")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insight:** Tenure and Total Charges are strongly correlated  
    **Takeaway:** No severe multicollinearity issues
    """)

# ---------------- TAB 3 ----------------
with tab3:

    st.subheader("💰 Business Impact Simulator")

    cost_fp = st.slider("Cost of False Positive ($)", 0, 500, 50)
    cost_fn = st.slider("Cost of False Negative ($)", 0, 1000, 500)

    cm = metrics["confusion_matrix"]

    fp = cm[0][1]
    fn = cm[1][0]

    total_cost = fp * cost_fp + fn * cost_fn

    st.metric("Total Estimated loss", f"${total_cost:,}")

    st.info("Reducing missed churn customers has highest financial impact")

    results_df["Churn Captured (%)"] = (results_df["Recall"] * 100).round(1)

    st.dataframe(results_df)

    best_model = results_df.iloc[-1]

    st.success(f"""
    🏆 Best Model: {best_model['Model']}  
    Captures {best_model['Churn Captured (%)']}% of churners
    """)


    st.subheader("📈 Model Comparison")

    st.dataframe(results_df)

    fig = px.bar(
        results_df,
        x="Model",
        y=["Accuracy","Precision","Recall","F1 Score", "ROC-AUC"],
        barmode="group"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ================= MODEL COMPARISON =================
    st.markdown("---")
    st.subheader("📊 Model Comparison (Churn Detection Effectiveness)")

    # Sort by Recall (business priority)
    mc_df = results_df.sort_values(by="Recall")

    fig = px.bar(
        mc_df,
        x="Recall",
        y="Model",
        orientation="h",
        text=mc_df["Recall"].round(2),
        title="Model Comparison Based on Recall (Higher = Better)",
        hover_data={
            "Accuracy": True,
            "Precision": True,
            "Recall": True,
            "F1 Score": True
        }
    )

    fig.update_layout(
        xaxis_title="Recall (Churn Detection Rate)",
        yaxis_title="",
        height=400
    )

    fig.update_traces(textposition="outside")

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    ### 💼 Business Interpretation

    - **Recall is prioritized** because missing churners leads to revenue loss  
    - Random Forest shows the **highest churn detection capability**  
    - Lower-performing models miss more high-risk customers  

    👉 This directly impacts retention strategy effectiveness
    """)

    mc_df = results_df.sort_values(by="F1 Score")

    fig = px.bar(
        mc_df,
        x="F1 Score",
        y="Model",
        orientation="h",
        text=mc_df["F1 Score"].round(2),
        title="Model Comparison (Balanced Performance - F1 Score)",
        hover_data=["Accuracy", "Precision", "Recall"]
    )

    fig.update_traces(textposition="outside")

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    ### 💼 Model Selection Insight

    - Logistic Regression shows very high recall but low precision → over-predicting churn  
    - Random Forest and Gradient Boosting provide better balance  
    - **F1 Score is used for final model selection**

    👉 Best model balances detection + accuracy of predictions
    """)

    st.markdown("---")
    st.subheader("🧩 Confusion Matrix Comparison")

    cols = st.columns(2)

    for idx, (model_name, cm) in enumerate(confusion_matrices.items()):

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted No", "Predicted Yes"],
            y=["Actual No", "Actual Yes"],
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues"
        ))

        fig.update_layout(
            title=model_name,
            height=350
        )

        with cols[idx % 2]:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("💰 Business Cost Comparison")

    cost_df = pd.DataFrame({
        "Model": business_costs.keys(),
        "Total Loss ($)": business_costs.values()
    })

    cost_df = cost_df.sort_values(
        by="Total Loss ($)",
        ascending=True
    )

    fig = px.bar(
        cost_df,
        x="Total Loss ($)",
        y="Model",
        orientation="h",
        text="Total Loss ($)",
        title="Estimated Business Loss by Model"
    )

    fig.update_traces(textposition="outside")

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(cost_df, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Lift Chart Analysis")

    fig = px.line(
        lift_table,
        x="decile",
        y="lift",
        markers=True,
        title="Lift by Decile"
    )

    fig.add_hline(
        y=1,
        line_dash="dash"
    )

    fig.update_layout(
        xaxis_title="Decile (1 = Highest Risk Customers)",
        yaxis_title="Lift"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig = px.line(
        lift_table,
        x="decile",
        y="cum_lift",
        markers=True,
        title="Cumulative Lift Chart"
    )

    fig.add_hline(
        y=1,
        line_dash="dash"
    )

    fig.update_layout(
        xaxis_title="Decile",
        yaxis_title="Cumulative Lift"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
        ### 💼 Business Interpretation

        - Top deciles contain customers with highest churn probability
        - A lift > 1 means the model performs better than random targeting
        - Higher lift in early deciles indicates strong ranking capability
        - Marketing teams can prioritize retention campaigns efficiently

        👉 This directly improves campaign ROI and reduces retention costs.
    """)

# ---------------- TAB 4 ----------------
with tab4:
    images = {
        "/assets/pipeline.png": BASE_DIR / "assets" / "pipeline.png",
        "/assets/model_comparison.png": BASE_DIR / "assets" / "model_comparison.png",
        "/assets/hyperparameter.png": BASE_DIR / "assets" / "hyperparameter.png",
        "/assets/roc_curve.png": BASE_DIR / "assets" / "roc_curve.png",
        "/assets/confusion_matrix.png": BASE_DIR / "assets" / "confusion_matrix.png",
        "/assets/feature_importance.png": BASE_DIR / "assets" / "feature_importance.png",
        "/assets/churn_distribution_pie.png": BASE_DIR / "assets" / "churn_distribution_pie.png",
        "/assets/contract_vs_churn.png": BASE_DIR / "assets" / "contract_vs_churn.png",
        "/assets/tenure_distribution.png": BASE_DIR / "assets" / "tenure_distribution.png",
        "/assets/churn_distribution_payment_method.png": BASE_DIR / "assets" / "churn_distribution_payment_method.png",
        "/assets/cost_by_model.png": BASE_DIR / "assets" / "cost_by_model.png",
        "/assets/cumulative_lift_chart.png": BASE_DIR / "assets" / "cumulative_lift_chart.png",
        "/assets/lift_chart_analysis_by_decile.png": BASE_DIR / "assets" / "lift_chart_analysis_by_decile.png",
        "/assets/newplot.png": BASE_DIR / "assets" / "newplot.png",
        "/assets/all_confusion_matrices.png": BASE_DIR / "assets" / "all_confusion_matrices.png"
    }
    st.subheader("📄 Full Report")
    
    report_path = BASE_DIR / "assets" / "customer_churn_report.html"

    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            html = f.read()
            for key, path in images.items():
                img_base64 = get_base64_image(path)
                html = html.replace(key, f"data:image/png;base64,{img_base64}")
        components.html(html, height=1200, scrolling=True)
    else:
        st.error("Report not found")        