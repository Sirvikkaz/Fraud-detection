from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
from src.prediction import prediction
import logging
import csv
import os
from datetime import datetime

logger = logging.getLogger("app")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("app.log")
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Prediction log file
PREDICTION_LOG = "reports/prediction_log.csv"
os.makedirs("reports", exist_ok=True)

def log_prediction(transaction_data: dict, prediction_label: str, prediction_value: int):
    """Log each prediction to a CSV file for monitoring."""
    file_exists = os.path.isfile(PREDICTION_LOG)
    with open(PREDICTION_LOG, "a", newline="") as f:
        fieldnames = ["timestamp", "Amount", "Time", "prediction", "label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "Amount": transaction_data.get("Amount"),
            "Time": transaction_data.get("Time"),
            "prediction": prediction_value,
            "label": prediction_label
        })

app = FastAPI(
    title="API for fraud detection",
    description="An API that takes transactions as data input and flags transactions as fraudulent or not",
    version="0.1.0"
)

try:
    with open("models/best_model.pickle", "rb") as f:
        model = pickle.load(f)
        logger.debug("Model loaded successfully")
except FileNotFoundError:
    logger.error("Model not found")
    raise
except Exception:
    logger.exception("Unexpected error")
    raise

try:
    with open("models/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
        logger.debug("Preprocessor loaded successfully")
except FileNotFoundError:
    logger.error("Preprocessor not found")
    raise
except Exception:
    logger.exception("Unexpected error")
    raise

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0)


transaction = {
    "Time": 406, "V1": -1.359807, "V2": -0.072781, "V3": 2.536347,
    "V4": 1.378155, "V5": -0.338321, "V6": 0.462388, "V7": 0.239599,
    "V8": 0.098698, "V9": 0.363787, "V10": 0.090794, "V11": -0.551600,
    "V12": -0.617801, "V13": -0.991390, "V14": -0.311169, "V15": 1.468177,
    "V16": -0.470401, "V17": 0.207971, "V18": 0.025791, "V19": 0.403993,
    "V20": 0.251412, "V21": -0.018307, "V22": 0.277838, "V23": -0.110474,
    "V24": 0.066928, "V25": 0.128539, "V26": -0.189115, "V27": 0.133558,
    "V28": -0.021053, "Amount": 149.62
}

warmed_up = False

@app.get("/")
def root():
    global warmed_up
    if not warmed_up:
        try:
            transactions = pd.DataFrame([transaction])
            prediction(model, transactions, preprocessor)
            warmed_up = True
        except Exception:
            logger.exception("Model warm-up failed")
            return {"status": "Model not ready"}
    return {"message": "Model ready!"}


@app.post("/predict")
async def predict_transaction(transaction: Transaction):
    transaction_data = transaction.model_dump()
    df = pd.DataFrame([transaction_data])

    try:
        pred, _ = prediction(model, df, preprocessor)
        label = "Fraud" if pred[0] == 1 else "Normal"
        prediction_value = int(pred[0])

        # Log prediction
        print(f"Logging prediction to {PREDICTION_LOG}")
        print(f"Reports folder exists: {os.path.exists('reports')}")
        
        log_prediction(transaction_data, label, prediction_value)
        logger.info(f"Prediction made — Amount: {transaction_data['Amount']} | Label: {label}")

        return {
            "prediction": prediction_value,
            "label": label
        }

    except Exception as e:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Prediction failed")