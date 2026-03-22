import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Customer Churn Intelligence", layout="wide")

# ================= UI =================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}
[data-testid="stSidebar"] {
    background: #020617;
}
.title {font-size:34px;font-weight:700;color:#e2e8f0;}
.subtitle {color:#94a3b8;margin-bottom:20px;}
.section {margin-top:40px;font-size:22px;font-weight:600;color:#e2e8f0;}
.metric-card {padding:25px;border-radius:16px;text-align:center;color:white;}
.kpi-customers {background:linear-gradient(135deg,#2563eb,#1e40af);}
.kpi-churn {background:linear-gradient(135deg,#f97316,#c2410c);}
.kpi-revenue {background:linear-gradient(135deg,#22c55e,#166534);}
.data-box {padding:20px;border-radius:14px;margin-bottom:20px;}
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

# ================= DATA =================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Custom dataset loaded successfully.")
else:
    df = load_data()

# ================= DATA REQUIREMENTS =================
if uploaded_file is None:
    st.markdown("""
    <div class="data-box" style="background: linear-gradient(135deg,#3b82f6,#1e3a8a);">
    <h4>📊 Data Requirements</h4>
    Required fields:
    <ul>
    <li>customer_id</li>
    <li>age</li>
    <li>tenure</li>
    <li>monthly_charges</li>
    <li>contract_type</li>
    <li>payment_method</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ================= VALIDATION =================
required_cols = ["customer_id","age","tenure","monthly_charges","contract_type","payment_method"]

for col in required_cols:
    if col not in df.columns:
        st.error(f"Missing column: {col}")
        st.stop()

if "churn" not in df.columns:
    df["churn"] = 0

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
X = df.drop(["churn","customer_id"], axis=1)
X = pd.get_dummies(X)

for col in model_columns:
    if col not in X:
        X[col] = 0

X = X[model_columns]

df["Churn_Probability"] = model.predict_proba(X)[:,1]
df["Revenue_Risk"] = df["Churn_Probability"] * df["monthly_charges"]

def risk_category(p):
    if p > 0.7: return "High Risk"
    elif p > 0.4: return "Medium Risk"
    return "Low Risk"

df["Risk_Category"] = df["Churn_Probability"].apply(risk_category)

# ================= KPI =================
st.markdown('<div class="section">Executive Overview</div>', unsafe_allow_html=True)

col1,col2,col3 = st.columns(3)

col1.markdown(f"<div class='metric-card kpi-customers'><h3>Total Customers</h3><h2>{len(df)}</h2></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card kpi-churn'><h3>Avg Churn</h3><h2>{round(df['Churn_Probability'].mean(),2)}</h2></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card kpi-revenue'><h3>Revenue Risk</h3><h2>₹{int(df['Revenue_Risk'].sum())}</h2></div>", unsafe_allow_html=True)

# ================= AI INSIGHTS =================
st.markdown('<div class="section">AI Insights Engine</div>', unsafe_allow_html=True)

insights = [
    f"{df.groupby('contract_type')['Revenue_Risk'].sum().idxmax()} contracts drive highest risk.",
    f"{df.groupby('payment_method')['Revenue_Risk'].sum().idxmax()} users have highest exposure.",
    f"Top 20% customers contribute {int((df.nlargest(int(0.2*len(df)),'Revenue_Risk')['Revenue_Risk'].sum()/df['Revenue_Risk'].sum())*100)}% risk.",
    f"{df[df['tenure']<6].shape[0]} low-tenure customers are churn drivers."
]

for i in insights:
    st.info(i)

# ================= ACTIONS =================
st.markdown('<div class="section">Recommended Actions</div>', unsafe_allow_html=True)

actions = [
    f"Convert {df[(df['Risk_Category']=='High Risk') & (df['contract_type']=='Monthly')].shape[0]} users to yearly plans.",
    f"Onboard {df[df['tenure']<6].shape[0]} new customers better.",
    f"Target {df[df['monthly_charges']>df['monthly_charges'].mean()].shape[0]} high-paying users.",
    f"Immediate retention for {df[df['Churn_Probability']>0.7].shape[0]} users."
]

for a in actions:
    st.success(a)

# ================= CHARTS =================
st.markdown('<div class="section">Analytics Dashboard</div>', unsafe_allow_html=True)

col4,col5 = st.columns(2)

col4.plotly_chart(px.histogram(df, x="Churn_Probability", nbins=40, color_discrete_sequence=["#3b82f6"]), use_container_width=True)

col5.plotly_chart(
    px.bar(df.groupby("Risk_Category")["Revenue_Risk"].sum().reset_index(),
           x="Risk_Category", y="Revenue_Risk",
           color="Risk_Category",
           color_discrete_map={"High Risk":"#ef4444","Medium Risk":"#f59e0b","Low Risk":"#22c55e"}),
    use_container_width=True
)

# ================= TABLES =================
st.markdown('<div class="section">Customer Intelligence Tables</div>', unsafe_allow_html=True)

st.subheader("Top High Risk Customers")
st.dataframe(df.sort_values("Revenue_Risk", ascending=False).head(10))

st.subheader("Full Customer Data")
st.dataframe(df.head(100))

# ================= LIVE PREDICTION =================
st.markdown('<div class="section">Live Prediction Engine</div>', unsafe_allow_html=True)

with st.form("predict_form"):
    age = st.number_input("Age", 18, 80, 30)
    tenure = st.number_input("Tenure", 1, 72, 12)
    monthly = st.number_input("Monthly Charges", 100.0, 10000.0, 2000.0)

    contract = st.selectbox("Contract Type", ["Monthly", "Yearly"])
    payment = st.selectbox("Payment Method", ["UPI", "Credit Card", "Debit Card"])

    submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([{
            "age": age,
            "tenure": tenure,
            "monthly_charges": monthly,
            "contract_type": contract,
            "payment_method": payment
        }])

        input_df = pd.get_dummies(input_df)

        for col in model_columns:
            if col not in input_df:
                input_df[col] = 0

        input_df = input_df[model_columns]

        prob = model.predict_proba(input_df)[0][1]

        if prob > 0.7:
            st.error(f"High Risk | Probability: {round(prob,2)}")
        elif prob > 0.4:
            st.warning(f"Medium Risk | Probability: {round(prob,2)}")
        else:
            st.success(f"Low Risk | Probability: {round(prob,2)}")