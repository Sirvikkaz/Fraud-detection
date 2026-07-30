import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def load_model() -> RandomForestClassifier:
    with open ("models/best_model.pickle", 'rb') as f:
        model = pickle.load(f)
        print("model function ran succesfully")
        return model

def prediction(model: RandomForestClassifier):
    prediction_data = pd.read_csv("data/prediction_data/prediction.csv")
    proba = model.predict_proba(prediction_data)[:, 1]
    pred = (proba>=0.4).astype(int)
    print("prediction function ran succesfully")
    return pred, proba

def save_prediction(pred, proba) -> None:
    os.makedirs("reports", exist_ok=True)

    prediction_df = pd.DataFrame({
        "prediction": pred,
        "probability": proba
    })

    prediction_df.to_csv("reports/predictions_outcome.csv", index=False)

    print("Predictions saved successfully.")

def main():
    model = load_model()
    pred, proba = prediction(model)
    save_prediction(pred, proba)
    print("The prediction for this transcation is: ", pred)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)