import pandas as pd

def create_features(df):

    df["avg_value_per_month"] = df["total_charges"] / (df["tenure"] + 1)

    df["high_value_customer"] = (df["total_charges"] > 50000).astype(int)

    df["long_term_customer"] = (df["tenure"] > 24).astype(int)

    df["contract_type"] = df["contract_type"].map({
        "Monthly": 0,
        "Yearly": 1
    })

    df["payment_method"] = df["payment_method"].map({
        "Credit Card": 0,
        "Debit Card": 1,
        "UPI": 2
    })

    return df