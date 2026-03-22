import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Customer Churn Intelligence", layout="wide")


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


st.markdown('<div class="title">Customer Churn Prediction & Revenue Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict churn, analyze revenue risk, generate insights and actions</div>', unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv("data/churn_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("model/churn_model.pkl")

model = load_model()
model_columns = joblib.load("model/columns.pkl")


st.sidebar.header("Filters")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data()

required_cols = ["customer_id", "age", "tenure", "monthly_charges", "contract_type", "payment_method"]

for col in required_cols:
    if col not in df.columns:
        st.error(f"Missing column: {col}")
        st.stop()

if "churn" not in df.columns:
    df["churn"] = 0


contract_filter = st.sidebar.multiselect("Contract Type", df["contract_type"].unique(), default=df["contract_type"].unique())
payment_filter = st.sidebar.multiselect("Payment Method", df["payment_method"].unique(), default=df["payment_method"].unique())
search_id = st.sidebar.text_input("Search Customer ID")

df = df[
    (df["contract_type"].isin(contract_filter)) &
    (df["payment_method"].isin(payment_filter))
]

if search_id:
    df = df[df["customer_id"].astype(str).str.contains(search_id)]


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


df["Priority_Score"] = df["Churn_Probability"] * df["monthly_charges"]
df["Potential_Revenue_Saved"] = df["Revenue_Risk"] * 0.6


def churn_reason(row):
    reasons = []
    if row["tenure"] < 6:
        reasons.append("Low tenure")
    if row["monthly_charges"] > df["monthly_charges"].mean():
        reasons.append("High charges")
    if row["contract_type"] == "Monthly":
        reasons.append("Monthly plan")
    if row["Churn_Probability"] > 0.7:
        reasons.append("Critical risk")
    return ", ".join(reasons) if reasons else "Stable"

df["Churn_Reason"] = df.apply(churn_reason, axis=1)


st.markdown('<div class="section">Executive Overview</div>', unsafe_allow_html=True)

total_customers = len(df)
avg_churn = df["Churn_Probability"].mean()
total_risk = df["Revenue_Risk"].sum()

col1, col2, col3 = st.columns(3)

col1.markdown(f"<div class='metric-card kpi-customers'><h3>Total Customers</h3><h2>{total_customers}</h2></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card kpi-churn'><h3>Avg Churn</h3><h2>{round(avg_churn,2)}</h2></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card kpi-revenue'><h3>Revenue Risk</h3><h2>₹{int(total_risk)}</h2></div>", unsafe_allow_html=True)


high_risk_count = df[df["Risk_Category"] == "High Risk"].shape[0]

st.markdown(f"""
<div style="margin-top:20px;padding:18px;border-radius:14px;text-align:center;
background:#065f46;color:#d1fae5;font-weight:600;">
Customer base stable | High Risk Customers: {high_risk_count}
</div>
""", unsafe_allow_html=True)


st.markdown('<div class="section">AI Insights Engine</div>', unsafe_allow_html=True)

insights = []
insights.append(f"{df.groupby('contract_type')['Revenue_Risk'].sum().idxmax()} contracts contribute highest revenue risk.")
insights.append(f"{df.groupby('payment_method')['Revenue_Risk'].sum().idxmax()} users show highest revenue exposure.")

top_20 = df.nlargest(int(0.2 * len(df)), "Revenue_Risk")["Revenue_Risk"].sum()
total = df["Revenue_Risk"].sum()
insights.append(f"Top 20% customers contribute {int((top_20/total)*100)}% of revenue risk.")

low_tenure = df[df["tenure"] < 6].shape[0]
insights.append(f"{low_tenure} low-tenure customers are major churn drivers.")

for i in insights:
    st.info(i)


st.markdown('<div class="section">Recommended Actions</div>', unsafe_allow_html=True)

actions = []

actions.append(f"Convert {df[(df['Risk_Category']=='High Risk') & (df['contract_type']=='Monthly')].shape[0]} high-risk monthly users to yearly plans.")
actions.append(f"Launch onboarding program for {df[df['tenure']<6].shape[0]} new customers.")
actions.append(f"Offer pricing optimization or discounts to {df[df['monthly_charges']>df['monthly_charges'].mean()].shape[0]} high-charge users.")
actions.append(f"Immediate retention campaign for {df[df['Churn_Probability']>0.7].shape[0]} critical-risk customers.")

for a in actions:
    st.success(a)


st.markdown('<div class="section">Executive Download Report</div>', unsafe_allow_html=True)

report_df = df[[
    "customer_id",
    "Churn_Probability",
    "Risk_Category",
    "Revenue_Risk",
    "Priority_Score",
    "Potential_Revenue_Saved",
    "Churn_Reason"
]].copy()

def map_action(row):
    if row["Risk_Category"] == "High Risk":
        return "Immediate retention (call + offer)"
    elif row["Risk_Category"] == "Medium Risk":
        return "Targeted engagement campaign"
    else:
        return "Maintain relationship"

report_df["Recommended_Action"] = report_df.apply(map_action, axis=1)

csv = report_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Executive Customer Report",
    data=csv,
    file_name="executive_churn_report.csv",
    mime="text/csv"
)


st.markdown('<div class="section">Analytics Dashboard</div>', unsafe_allow_html=True)

col4, col5 = st.columns(2)

col4.plotly_chart(px.histogram(df, x="Churn_Probability"), use_container_width=True)

col5.plotly_chart(
    px.bar(df.groupby("Risk_Category")["Revenue_Risk"].sum().reset_index(),
    x="Risk_Category", y="Revenue_Risk", color="Risk_Category"),
    use_container_width=True
)


st.markdown('<div class="section">Customer Intelligence Tables</div>', unsafe_allow_html=True)

st.subheader("Top High Risk Customers")
st.dataframe(df.sort_values(by="Priority_Score", ascending=False).head(10))

st.subheader("Customer Data")
st.dataframe(df.head(100))


st.markdown('<div class="section">Live Prediction Engine</div>', unsafe_allow_html=True)

with st.form("predict"):
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
            st.error(f"High Risk Customer | Probability: {round(prob,2)}")
        elif prob > 0.4:
            st.warning(f"Medium Risk Customer | Probability: {round(prob,2)}")
        else:
            st.success(f"Low Risk Customer | Probability: {round(prob,2)}")
