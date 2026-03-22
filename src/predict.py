import pandas as pd
import joblib

from feature_engineering import prepare_features

model = joblib.load("model/churn_model.pkl")

df = pd.read_csv("data/churn_data.csv")

X, _ = prepare_features(df)

df["Churn_Probability"] = model.predict_proba(X)[:,1]

df["Revenue_Risk"] = df["Churn_Probability"] * df["TotalCharges"]

df["Risk_Category"] = pd.cut(
    df["Churn_Probability"],
    bins=[0, 0.3, 0.7, 1],
    labels=["Low", "Medium", "High"]
)

df.to_csv("data/predictions.csv", index=False)

print("Predictions ready")