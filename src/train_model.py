import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model(df):

    X = df.drop(["churn", "customer_id"], axis=1)
    y = df["churn"]

    X = pd.get_dummies(X)

    joblib.dump(X.columns.tolist(), "model/columns.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Model Accuracy:", acc)

    joblib.dump(model, "model/churn_model.pkl")