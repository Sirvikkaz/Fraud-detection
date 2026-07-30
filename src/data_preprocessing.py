import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle

def preprocess_data(path):
    # Load data
    df = pd.read_csv(path)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Features & target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Scale only Time and Amount
    preprocessor = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), ["Time", "Amount"])
        ],
        remainder="passthrough"
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    X_train = pd.DataFrame(
        X_train,
        columns=feature_names
    )
    X_test = pd.DataFrame(
        X_test,
        columns = feature_names
    )
    return X_train, X_test, y_train, y_test, preprocessor

def save_data(X_train: pd.DataFrame, X_test: pd.DataFrame, Y_train: pd.DataFrame, Y_test: pd.DataFrame, data_path: str) -> None:
    try:
        data_path = os.path.join(data_path, 'model_data')
        os.makedirs(data_path, exist_ok=True)
        pd.DataFrame(X_train).to_csv(os.path.join(data_path, "train.csv"), index=False)
        pd.DataFrame(Y_train).to_csv(os.path.join(data_path, "y_train.csv"), index=False)
        pd.DataFrame(Y_test).to_csv(os.path.join(data_path, "y_test.csv"), index=False)
        pd.DataFrame(X_test).to_csv(os.path.join(data_path, "test.csv"), index=False)
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the data.")
        print(e)
        raise


os.makedirs("models", exist_ok=True)
path = "data/creditcard.csv"
def main():
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(path)
    save_data(X_train, X_test, y_train, y_test, 'data')
    pickle.dump(preprocessor, open("models/preprocessor.pkl", "wb"))
if __name__ == "__main__":
    main()
