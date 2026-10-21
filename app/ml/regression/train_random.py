
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURE_DIR = PROCESSED_DATA_DIR / "selected_features"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


class Random:
    def __init__(self):
        self.model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        min_samples_leaf=1,
        n_jobs=-1,
        )

        self._model_files = {
            "laon_amount": "random_laon_amount.pkl",
        }

    
    def train(self, target, model, X_train, y_train):
       
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

  
    def train_and_save(self, target):
         
      
        file="features_loan_amount_lasso_set_e.json"

        with open(FEATURE_DIR / file) as f:
            features = json.load(f)


        X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train_loan_amount.csv")[features]
        X_test  = pd.read_csv(PROCESSED_DATA_DIR / "X_test_loan_amount.csv")[features]
        y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train_loan_amount.csv").squeeze()
        y_test  = pd.read_csv(PROCESSED_DATA_DIR / "y_test_loan_amount.csv").squeeze()

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100

        model_file = "random_laon_amount.pkl"
        joblib.dump(self.model, MODELS_DIR / model_file)


        print("Regression Evaluation Metrics")
        print(f"MAE:  {mae:.2f}")
        print(f"MSE:  {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2:   {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Saved: {model_file}")

        return self.model


def main():
    trainer = Random()
    trainer.train_and_save("loan_amount")
    
    

if __name__ == "__main__":
    main()
