from pathlib import Path
import pandas as pd
import pickle
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Customer_Churn_Datasheet.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

def train():
    df = pd.read_csv(DATA_PATH)

    # Drop unwanted columns
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    # Convert numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Target
    df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Feature groups
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Pipelines
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    

    categorical_transformer = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"))
    ])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, None],
        "model__min_samples_split": [2, 5] 
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    # 🔥 Replace pipeline with best model
    pipeline = grid.best_estimator_
    #metrics["best_params"] = grid.best_params_

    print("Best Parameters:", grid.best_params_)

    # Save pipeline
    with open(MODEL_DIR / "pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print("✅ Pipeline saved successfully!")

    # Predictions
    # y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.3).astype(int)

    # Metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    accuracy = (y_pred == y_test).mean()
    cm = confusion_matrix(y_test, y_pred)

    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc)

    # Save metrics
    metrics = {
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "best_params": grid.best_params_
    }

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f)
    
    print("✅ metrics saved successfully!")

    # Save test data
    with open(MODEL_DIR / "test_data.pkl", "wb") as f:
        pickle.dump((X_test, y_test), f)

    print("✅ test data saved successfully!")

    model_results = []

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
        "Random Forest": RandomForestClassifier(class_weight="balanced")
    }

    for name, model in models.items():
        model_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        model_pipeline.fit(X_train, y_train)

        # y_pred = model_pipeline.predict(X_test)
        y_prob = model_pipeline.predict_proba(X_test)[:,1]
        y_pred = (y_prob >= 0.3).astype(int)

        model_results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred)
        })

    # Save results
    results_df = pd.DataFrame(model_results)
    results_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

    print("✅ Model comparison saved!")

    # Cross validation
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
    print("Cross-validation ROC-AUC:", scores)
    print("CV ROC-AUC:", scores.mean())

if __name__ == "__main__":
    train()