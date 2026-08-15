import os
import pickle
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score, roc_auc_score
)
import json
import pandas as pd
import mlflow
import mlflow.sklearn
import logging

logger = logging.getLogger("evaluation")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("evaluation.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def evaluate_model(model, test:pd.DataFrame, y_test:pd.Series) -> dict:
    y_proba = model.predict_proba(test)[:,1]
    y_pred = (y_proba>=0.4).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    r_a_score = roc_auc_score(y_test, y_proba)

    return { 
    "accuracy_score":accuracy,
    "recall_score":recall,
    "precision_score":precision,
    "f1_score":f1,
    "roc_auc_score":r_a_score
        }

def main():
    with open ("models/best_model.pickle", 'rb') as f:
        model = pickle.load(f)
    try:
        y_test = pd.read_csv("data/model_data/y_test.csv").squeeze()
        logger.debug("Successfully loaded data")
    except FileNotFoundError:
        logger.error("file not found")
        raise
    except Exception as e:
        logger.error("An unexcepted error occur: %s", e)
        raise
    test = pd.read_csv("data/model_data/test.csv")
    metrics = evaluate_model(model, test, y_test)
    os.makedirs("reports", exist_ok=True)
    with open ("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Fraud-detection")
    with mlflow.start_run():
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Error during evaluation")
        raise

