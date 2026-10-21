import sys
from pathlib import Path
import json
import joblib
sys.path.append(str(Path(__file__).resolve().parents[3]))

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from app.core.config import X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()


        #feautre_set
        self.feature_dir = PROCESSED_DATA_DIR/"selected_features"


    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)

    def get_model(self):
        return self.model

    def load_data(self,target,feature_set):
        if target == "default":
            X_train = self.X_train_default[feature_set]
            X_test = self.X_test_default[feature_set]
            y_train = self.y_train_default[target]
            y_test = self.y_test_default[target]
        elif target == "is_fraud":
            X_train = self.X_train_fraud[feature_set]
            X_test = self.X_test_fraud[feature_set]
            y_train = self.y_train_fraud[target]
            y_test = self.y_test_fraud[target]
        
        return X_train, X_test, y_train, y_test

  
    def train(self, target, model, X_train, y_train):
        """Fit on pre-split, already-scaled training data passed in by the caller."""
        self.model.fit(X_train, y_train)

   
    def train_and_save(self, target, feature_set):
        """Full standalone training from processed CSVs."""
        X_train, X_test, y_train, y_test = self.load_data(target, feature_set)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s  = scaler.transform(X_test)

        self.model.fit(X_train_s, y_train)
        y_pred = self.model.predict(X_test_s)

        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        confusion = confusion_matrix(y_test, y_pred)

        joblib.dump(scaler, MODELS_DIR / "logistic_regression_scaler.pkl")
        joblib.dump(self.model, MODELS_DIR / f"logistic_{target}.pkl")

        return accuracy, precision, recall, f1, confusion



def main():
    model = LinearRegressionModel()
    # loading feature sets
    set_e   = model.feature_dir / "features_loan_amount_set_e.json"
    
    with open(set_e) as f:
        loan_features = json.load(f)

    print("SET_E  |  LOAN AMOUNT")
    acc, prec, rec, f1, conf = model.train_and_save("loan_amount", fraud_features)
    print(f"accuracy={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}")
    print(f"confusion:\n{conf}")

     
    set_d_fraud   = model.feature_dir / "features_loan_amount_set_d.json"
    
    with open(set_d_fraud) as f:
        fraud_features = json.load(f)
   
    print("\nSET_D  |  FRAUD")
    acc, prec, rec, f1, conf = model.train_and_save("loan_amount", fraud_features)
    print(f"accuracy={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}")
    print(f"confusion:\n{conf}")


if __name__ == "__main__":
    main()

    
    



    











