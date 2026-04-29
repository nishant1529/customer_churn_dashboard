from pathlib import Path
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Customer_Churn_Datasheet.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

def train():
    df = pd.read_csv(DATA_PATH)

    # Drop ID
    df.drop("customerID", axis=1, inplace=True, errors="ignore")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Target
    df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Column groups
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    pipeline.fit(X_train, y_train)

    # Save pipeline
    with open(MODEL_DIR / "pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print("✅ Pipeline saved successfully!")

if __name__ == "__main__":
    train()