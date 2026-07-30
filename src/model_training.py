import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import pickle
import mlflow
import mlflow.sklearn
import json

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split':[2,5],
    'min_samples_leaf':[1,2],
    'class_weight':[None, 'balanced']
}
rf = RandomForestClassifier(random_state=42)

# grid = GridSearchCV(
#     estimator=rf,
#     scoring='f1',
#     param_grid=param_grid,
#     n_jobs =-1,
#     cv=5,
#     verbose=2
# )

random_search = RandomizedSearchCV(
    estimator=rf,
    scoring='f1',
    param_distributions=param_grid,
    n_jobs=-1,
    n_iter = 10,
    cv = 5, 
    verbose = 2,
    random_state=42
)
model = RandomForestClassifier(random_state=42, class_weight="balanced")
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
def model_train(X_train:pd.DataFrame, y_train:pd.Series)->RandomForestClassifier:
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Fraud-detection")
    with mlflow.start_run():
        #Train
        random_search.fit(X_train, y_train)
        model = random_search.best_estimator_
        #Log best params
        best_params = random_search.best_params_
        
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        print(random_search.best_score_)
        #Log best cv score
        mlflow.log_metric("best_cv_f1_score", random_search.best_score_)
        
        #log model 
        mlflow.sklearn.log_model(model, "Random-Forest-Model")

        with open("models/best_model.pickle", "wb") as f:
            pickle.dump(model, f)
        
        print(f"Best params{random_search.best_params_}")
        print(f"Best CV F1: {random_search.best_score_}")
        return model 



def main():
    X_train = pd.read_csv("data/model_data/train.csv")
    y_train = pd.read_csv("data/model_data/y_train.csv").squeeze() #added to convert y_train to Series
    model_train(X_train, y_train)

if __name__ == "__main__":
    main()

