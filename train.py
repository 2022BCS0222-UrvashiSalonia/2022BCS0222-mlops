import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import argparse
import json
import os

STUDENT_NAME = "Urvashi Salonia"
ROLL_NO = "2022BCS0222"

mlflow.set_experiment("2022BCS0222_experiment")

def train(dataset_path, model_type, n_estimators, max_depth, features):
    df = pd.read_csv(dataset_path)
    
    X = df[features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        mlflow.log_param("dataset", dataset_path)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", str(features))
        mlflow.log_param("roll_no", ROLL_NO)

        if model_type == "rf":
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        else:
            model = LogisticRegression(max_iter=200, random_state=42)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        metrics = {
            "name": STUDENT_NAME,
            "roll_no": ROLL_NO,
            "model_type": model_type,
            "accuracy": acc,
            "f1_score": f1,
            "features": features
        }
        os.makedirs("models", exist_ok=True)
        with open("models/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/iris_v2.csv")
    parser.add_argument("--model_type", default="rf")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--features", nargs="+", default=[
        "sepal length (cm)", "sepal width (cm)",
        "petal length (cm)", "petal width (cm)"
    ])
    args = parser.parse_args()
    train(args.dataset, args.model_type, args.n_estimators, args.max_depth, args.features)