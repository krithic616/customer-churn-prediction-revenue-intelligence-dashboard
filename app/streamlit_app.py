import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import time

st.set_page_config(page_title="Customer Churn Intelligence", layout="wide")

# ================= UI STYLE =================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}
[data-testid="stSidebar"] {
    background: #020617;
}
.title {
    font-size: 34px;
    font-weight: 700;
    color: #e2e8f0;
}
.subtitle {
    color: #94a3b8;
    margin-bottom: 20px;
}
.section {
    margin-top: 40px;
    margin-bottom: 15px;
    font-size: 22px;
    font-weight: 600;
    color: #e2e8f0;
}
.metric-card {
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    color: white;
}
.kpi-customers {
    background: linear-gradient(135deg, #2563eb, #1e40af);
}
.kpi-churn {
    background: linear-gradient(135deg, #f97316, #c2410c);
}
.kpi-revenue {
    background: linear-gradient(135deg, #22c55e, #166534);
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown('<div class="title">Customer Churn Prediction & Revenue Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict churn, analyze revenue risk, generate insights and actions</div>', unsafe_allow_html=True)

# ================= LOAD =================
@st.cache_data
def load_data():
    return pd.read_csv("data/churn_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("model/churn_model.pkl")

model = load_model()
model_columns = joblib.load("model/columns.pkl")

# ================= SIDEBAR =================
st.sidebar.header("Filters")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# ================= DATA LOAD =================
if uploaded_file:
    with st.spinner("Processing dataset... Please wait"):
        progress = st.progress(0)
        for i in range(1, 101):
            time.sleep(0.01)
            progress.progress(i)
        df = pd.read_csv(uploaded_file)
    st.success("Custom dataset loaded successfully.")
else:
    df = load_data()

# ================= VALIDATION =================
required_cols = ["customer_id", "age", "tenure", "monthly_charges", "contract_type", "payment_method"]

for col in required_cols:
    if col not in df.columns:
        st.error(f"Missing column: {col}")
        st.stop()

if "churn" not in df.columns:
    df["churn"] = 0

# ================= TOP PANELS =================
st.markdown("""
<div style="background: linear-gradient(90deg, #3b82f6, #1e3a8a);
padding: 20px; border-radius: 14px; margin-top: 20px; margin-bottom: 15px; color: white;">
<h4>📊 Data Requirements</h4>
Provide a valid customer dataset to generate churn predictions and revenue intelligence.
<b>Required fields:</b>
<ul>
<li>customer_id</li>
<li>age</li>
<li>tenure</li>
<li>monthly_charges</li>
<li>contract_type</li>
<li>payment_method</li>
</ul>
⚠️ CSV format required • Column names must match exactly
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(90deg, #1e293b, #334155);
padding: 15px; border-radius: 14px; margin-bottom: 25px; color: white;">
📘 <b>Demo Mode</b><br>
Showing sample dataset. Upload your own dataset to generate personalized churn insights and revenue intelligence.
</div>
""", unsafe_allow_html=True)

# ================= FILTERS =================
contract_filter = st.sidebar.multiselect("Contract Type", df["contract_type"].unique(), default=df["contract_type"].unique())
payment_filter = st.sidebar.multiselect("Payment Method", df["payment_method"].unique(), default=df["payment_method"].unique())
search_id = st.sidebar.text_input("Search Customer ID")

df = df[
    (df["contract_type"].isin(contract_filter)) &
    (df["payment_method"].isin(payment_filter))
]

if search_id:
    df = df[df["customer_id"].astype(str).str.contains(search_id)]

# ================= MODEL =================
with st.spinner("Running AI churn prediction engine..."):
    X = df.drop(["churn", "customer_id"], axis=1)
    X = pd.get_dummies(X)

    for col in model_columns:
        if col not in X:
            X[col] = 0

    X = X[model_columns]

    df["Churn_Probability"] = model.predict_proba(X)[:, 1]
    df["Revenue_Risk"] = df["Churn_Probability"] * df["monthly_charges"]

def risk_category(p):
    if p > 0.7:
        return "High Risk"
    elif p > 0.4:
        return "Medium Risk"
    return "Low Risk"

df["Risk_Category"] = df["Churn_Probability"].apply(risk_category)

# ================= KPI =================
st.markdown('<div class="section">Executive Overview</div>', unsafe_allow_html=True)

total_customers = len(df)
avg_churn = df["Churn_Probability"].mean()
total_risk = df["Revenue_Risk"].sum()
high_risk_count = df[df["Risk_Category"] == "High Risk"].shape[0]

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"<div class='metric-card kpi-customers'><h3>Total Customers</h3><h2>{total_customers}</h2></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card kpi-revenue'><h3>Revenue Risk</h3><h2>₹{int(total_risk)}</h2></div>", unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    st.markdown(f"<div class='metric-card kpi-churn'><h3>Avg Churn</h3><h2>{round(avg_churn,2)}</h2></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-card' style='background:linear-gradient(135deg,#7c3aed,#4c1d95);'><h3>High Risk Customers</h3><h2>{high_risk_count}</h2></div>", unsafe_allow_html=True)

# ================= ALERT =================
st.markdown(f"""
<div style="margin-top:20px;padding:18px;border-radius:14px;text-align:center;
background:#065f46;color:#d1fae5;font-weight:600;">
Customer base stable | High Risk Customers: {high_risk_count}
</div>
""", unsafe_allow_html=True)

# ================= CHARTS =================
st.markdown('<div class="section">Analytics Dashboard</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
c1.plotly_chart(px.histogram(df, x="Churn_Probability", template="plotly_dark"), use_container_width=True)
c2.plotly_chart(
    px.bar(df.groupby("Risk_Category")["Revenue_Risk"].sum().reset_index(),
    x="Risk_Category", y="Revenue_Risk", color="Risk_Category", template="plotly_dark"),
    use_container_width=True
)