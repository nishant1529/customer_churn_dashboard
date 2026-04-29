import pandas as pd

def preprocess(df):
    df = df.copy()

    # Drop safely
    df.drop("customerID", axis=1, inplace=True, errors="ignore")

    # Numeric conversion
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Target handling
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].replace({"Yes":1, "No":0})

    # Binary encoding
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].replace({"Yes":1, "No":0, "Male":1, "Female":0})

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    return df