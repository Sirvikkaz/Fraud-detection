from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from src.prediction import prediction
app = FastAPI (
    title="API for fraud detection",
    description = "An API, that takes transactions as data input and flags transactions as fraudulent or not",
    version="0.1.0"
)

with open("models/best_model.pickle", "rb") as f:
    model = pickle.load(f)
with open("models/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

class Transaction(BaseModel):
    Time:float
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
    Amount: float


transaction = {
    "Time": 406,
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536347,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.551600,
    "V12": -0.617801,
    "V13": -0.991390,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount": 149.62
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
        except Exception as e:
            return {
                "status": "Model not ready",
                "error": str(e)
            }
    return {
        "message":"Model ready!"
        }

@app.post("/predict")
async def predict_transaction(transaction:Transaction):
    df = pd.DataFrame([transaction.model_dump()])
    # df = df[["Time", "Amount", "V1", "V2", "V3", "V4", "V5", "V6", "V7", 
    #          "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", 
    #          "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", 
    #          "V26", "V27", "V28"]]

    try:
        pred, _ = prediction(model, df, preprocessor)
        return {
            "prediction": int(pred[0]),
            "label": "Fraud" if pred[0] == 1 else "Normal"
        }
    
    except Exception as e:
        return {
            "error":str(e)
        }

    