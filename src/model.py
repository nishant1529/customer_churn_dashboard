from pathlib import Path
import pandas as pd  # type: ignore[import]
import pickle
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Customer_Churn_Datasheet.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

def calculate_business_cost(y_true, y_pred,
                            cost_fp=50,
                            cost_fn=500):

    cm = confusion_matrix(y_true, y_pred)

    fp = cm[0][1]
    fn = cm[1][0]

    total_cost = (fp * cost_fp) + (fn * cost_fn)

    return total_cost

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

    confusion_matrices = {}

    models = {
        "KNN": Pipeline([
            ("preprocessor", preprocessor),
            ("model", KNeighborsClassifier())
        ]),

        "Decision Tree": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(max_depth=5))
        ]),

        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(max_depth=7))
        ]),

        "Dummy": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DummyClassifier())
        ])
    }
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm.tolist()

    # SAVE JSON
    with open(MODEL_DIR / "confusion_matrices.json", "w") as f:
        json.dump(confusion_matrices, f)

    print("✅ Confusion matrices saved!")

    # ================= BUSINESS COST COMPARISON =================

    business_costs = {}

    models = {
        "Random Forest": models.get("Random Forest"),
        "Decision Tree": models.get("Decision Tree"),
        "KNN": models.get("KNN"),
        "Dummy": models.get("Dummy")
    }

    for name, model in models.items():
        preds = model.predict(X_test)
        cost = calculate_business_cost(y_test, preds)
        business_costs[name] = float(cost)

    # SAVE JSON
    with open(MODEL_DIR / "business_costs.json", "w") as f:
        json.dump(business_costs, f)

    print("✅ Business costs saved!")

    # OPTIONAL CSV
    cost_df = pd.DataFrame.from_dict(
        business_costs,
        orient='index',
        columns=['Total Loss ($)']
    )

    cost_df.to_csv(MODEL_DIR / "business_costs.csv")

    # ================= LIFT CHART DATA =================

    # Use Random Forest probabilities
    rf_probs = models.get("Random Forest").predict_proba(X_test)[:, 1]

    lift_df = pd.DataFrame({
        "actual": y_test.reset_index(drop=True),
        "pred_prob": rf_probs
    })

    # Sort highest probability first
    lift_df = lift_df.sort_values(
        "pred_prob",
        ascending=False
    ).reset_index(drop=True)

    # Create deciles
    lift_df["decile"] = pd.qcut(
        lift_df.index,
        10,
        labels=False
    ) + 1

    # Aggregate
    lift_table = lift_df.groupby("decile").agg(
        total_users=("actual", "count"),
        conversions=("actual", "sum"),
        avg_pred_prob=("pred_prob", "mean")
    ).reset_index()

    # Conversion rate
    lift_table["churn_rate"] = (
        lift_table["conversions"] /
        lift_table["total_users"]
    )

    # Overall rate
    overall_churn_rate = lift_df["actual"].mean()

    # Lift
    lift_table["lift"] = (
        lift_table["churn_rate"] /
        overall_churn_rate
    )

    # Cumulative metrics
    lift_table["cum_users"] = lift_table["total_users"].cumsum()

    lift_table["cum_churn"] = (
        lift_table["conversions"].cumsum()
    )

    lift_table["cum_churn_rate"] = (
        lift_table["cum_churn"] /
        lift_table["cum_users"]
    )

    lift_table["cum_lift"] = (
        lift_table["cum_churn_rate"] /
        overall_churn_rate
    )

    # SAVE
    lift_table.to_json(
        MODEL_DIR / "lift_table.json",
        orient="records"
    )

    print("✅ Lift chart data saved!")

    # Train, number of estimators, depth and samples split
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

    # Replace pipeline with best model
    pipeline = grid.best_estimator_


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

    # print(classification_report(y_test, y_pred))
    # print("ROC-AUC:", roc_auc)

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
        "Random Forest": RandomForestClassifier(class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier()
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
            "F1 Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
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