import sys
from pathlib import Path
from pydantic import typing
from typing import Any
import json
import joblib
sys.path.append(str(Path(__file__).resolve().parents[3]))

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from app.core.config import X_TRAIN_DEFAULT_PATH, X_TEST_DEFAULT_PATH, Y_TRAIN_DEFAULT_PATH, Y_TEST_DEFAULT_PATH, X_TRAIN_FRAUD_PATH, X_TEST_FRAUD_PATH, Y_TRAIN_FRAUD_PATH, Y_TEST_FRAUD_PATH
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

class DecisionTree:
    
    def __init__(self):
        self.model = DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
            class_weight="balanced"
            )

        # default
        self.X_train_default = pd.read_csv(PROCESSED_DATA_DIR / "X_train_default.csv")
        self.X_test_default  = pd.read_csv(PROCESSED_DATA_DIR / "X_test_default.csv")
        self.y_train_default = pd.read_csv(PROCESSED_DATA_DIR / "y_train_default.csv")
        self.y_test_default  = pd.read_csv(PROCESSED_DATA_DIR / "y_test_default.csv")

        # is_fraud
        self.X_train_fraud = pd.read_csv(PROCESSED_DATA_DIR / "X_train_is_fraud.csv")
        self.X_test_fraud  = pd.read_csv(PROCESSED_DATA_DIR / "X_test_is_fraud.csv")
        self.y_train_fraud = pd.read_csv(PROCESSED_DATA_DIR / "y_train_is_fraud.csv")
        self.y_test_fraud  = pd.read_csv(PROCESSED_DATA_DIR / "y_test_is_fraud.csv")

        #feature set
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
            y_train = self.y_train_default['default']
            y_test = self.y_test_default['default']
        elif target == "is_fraud":
            X_train = self.X_train_fraud[feature_set]
            X_test = self.X_test_fraud[feature_set]
            y_train = self.y_train_fraud['is_fraud']
            y_test = self.y_test_fraud['is_fraud']
        
        return X_train, X_test, y_train, y_test

    
    def train_(self,target,model,feature_set):
        model = model
        X_train, X_test, y_train, y_test = model.load_data(target,feature_set)
        self.model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall=recall_score(y_test, y_pred, zero_division=0)
        f1=f1_score(y_test, y_pred, zero_division=0)
        confusion=confusion_matrix(y_test, y_pred)
        return accuracy, precision ,recall,f1,confusion


def main():
    model = DecisionTree()

    with open(model.feature_dir / "features_is_fraud_set_e.json", "r") as f:
        fraud_features = json.load(f)

    with open(model.feature_dir / "features_default_set_e.json", "r") as f:
        default_features = json.load(f)
        
    print("SET E")
    
    is_fraud_accuracy, is_fraud_precision,is_fraud_recall,is_fraud_f1,is_fraud_confusion= model.train_("is_fraud",model,fraud_features)
    print("is_fraud_accuracy:", is_fraud_accuracy)
    print("is_fraud_precision:", is_fraud_precision)
    print("is_Recall:",is_fraud_recall)
    print("is_f1:",is_fraud_f1)
    print("is_confusion:",is_fraud_confusion)
     
    
    default_accuracy, default_precision,default_recall,default_f1,default_confusion= model.train_("default",model,default_features)
    print("default_accuracy:", default_accuracy)
    print("default_precision:", default_precision)
    print("default_recall:",default_recall)
    print("default_f1:",default_f1)
    print("default_confusion:",default_confusion)

    print("SET_D")

    with open(model.feature_dir / "features_is_fraud_set_d.json", "r") as f:
        fraud_features = json.load(f)

    with open(model.feature_dir / "features_default_set_d.json", "r") as f:
        default_features = json.load(f)

    
    is_fraud_accuracy, is_fraud_precision,is_fraud_recall,is_fraud_f1,is_fraud_confusion= model.train_("is_fraud",model,fraud_features)
    print("is_fraud_accuracy:", is_fraud_accuracy)
    print("is_fraud_precision:", is_fraud_precision)
    print("is_Recall:",is_fraud_recall)
    print("is_f1:",is_fraud_f1)
    print("is_confusion:",is_fraud_confusion)
     
    
    default_accuracy, default_precision,default_recall,default_f1,default_confusion= model.train_("default",model,default_features)
    print("default_accuracy:", default_accuracy)
    print("default_precision:", default_precision)
    print("default_recall:",default_recall)
    print("default_f1:",default_f1)
    print("default_confusion:",default_confusion)


    

    MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODELS_DIR / "decision_fraud.pkl")
    joblib.dump(model, MODELS_DIR / "decision_default.pkl")


if __name__ == "__main__":
    main()