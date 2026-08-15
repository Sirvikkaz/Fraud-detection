import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger("prediction")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("prediction.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model() -> RandomForestClassifier:
    try:
        with open ("models/best_model.pickle", 'rb') as f:
            model = pickle.load(f)
            logger.debug("model function ran successfully")
            return model
    except FileNotFoundError:
        logger.error("Model file not found")
        raise
    except Exception:
        logger.exception("Unexpected error loading model")
        raise

def load_preprocessor():
    try:
        with open ("models/preprocessor.pkl", 'rb') as f:
            preprocessor = pickle.load(f)
            logger.debug("preprocessor function ran successfully")
            return preprocessor
    except FileNotFoundError:
        logger.error("Preprocessor file not found")
        raise
    except Exception:
        logger.exception("Unexpected error loading the preprocessor")
        raise
    
def prediction(model: RandomForestClassifier, prediction_data:pd.DataFrame, preprocessor):
    #prediction_data = pd.read_csv("data/prediction_data/prediction.csv")
    prediction_data = preprocessor.transform(prediction_data)
    prediction_data = pd.DataFrame(
        prediction_data,
        columns = model.feature_names_in_
    )
    proba = model.predict_proba(prediction_data)[:, 1]
    pred = (proba>=0.4).astype(int)
    logger.debug("prediction function ran successfully")
    return pred, proba

def save_prediction(pred, proba) -> None:
    os.makedirs("reports", exist_ok=True)

    prediction_df = pd.DataFrame({
        "prediction": pred,
        "probability": proba
    })

    prediction_df.to_csv("reports/predictions_outcome.csv", index=False)

    logger.debug("Predictions saved successfully.")

def main():
    model = load_model()
    preprocessor = load_preprocessor()
    try:
        prediction_data = pd.read_csv("data/prediction_data/prediction.csv")
        logger.debug("Successfully loaded data")
    except FileNotFoundError:
        logger.error("prediction data file not found")
        raise
    except Exception:
        logger.exception("Unexpected error loading data")
        raise
    pred, proba = prediction(model, prediction_data, preprocessor)
    save_prediction(pred, proba)
    logger.info("The prediction for this transcation is: %s", pred)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unexpected error")
        raise