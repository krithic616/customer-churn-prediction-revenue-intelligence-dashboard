import pandas as pd
import numpy as np

def load_data():

    np.random.seed(42)

    n = 1000

    data = pd.DataFrame({
        "customer_id": range(1, n+1),
        "age": np.random.randint(18, 70, n),
        "tenure": np.random.randint(1, 72, n),
        "monthly_charges": np.random.uniform(300, 5000, n),
        "contract_type": np.random.choice(["Monthly", "Yearly"], n),
        "payment_method": np.random.choice(["Credit Card", "UPI", "Debit Card"], n),
    })

    data["total_charges"] = data["monthly_charges"] * data["tenure"]

    data["churn"] = (
        (data["tenure"] < 12) &
        (data["monthly_charges"] > 2000)
    ).astype(int)

    return data


def save_data():
    df = load_data()
    df.to_csv("data/churn_data.csv", index=False)
    print("Dataset created and saved!")


if __name__ == "__main__":
    save_data()