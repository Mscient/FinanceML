import sys
from pathlib import Path

 
sys.path.append(str(Path(__file__).resolve().parents[3]))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from app.core.config import X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH

class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)

    def get_model(self):
        return self.model

    def load_data(self):
        X_train = pd.read_csv(X_TRAIN_PATH)
        X_test = pd.read_csv(X_TEST_PATH)
        y_train = pd.read_csv(Y_TRAIN_PATH)
        y_test = pd.read_csv(Y_TEST_PATH)
        return X_train, X_test, y_train, y_test


def is_fraud():
    model = LogisticRegressionModel()
    X_train, X_test, y_train, y_test = model.load_data()
    y_train = y_train['is_fraud']
    y_test = y_test['is_fraud']
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall=recall_score(y_test, y_pred, zero_division=0)
    f1=f1_score(y_test, y_pred, zero_division=0)
    confusion=confusion_matrix(y_test, y_pred)
    return accuracy, precision ,recall,f1,confusion


def default():
    model = LogisticRegressionModel()
    X_train, X_test, y_train, y_test = model.load_data()
    y_train = y_train['default']
    y_test = y_test['default']
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall=recall_score(y_test, y_pred, zero_division=0)
    f1=f1_score(y_test, y_pred, zero_division=0)
    confusion=confusion_matrix(y_test, y_pred)
    return accuracy, precision,recall,f1,confusion 


def main():
    is_fraud_accuracy, is_fraud_precision,is_fraud_recall,is_fraud_f1,is_fraud_confusion= is_fraud()
    print("is_fraud_accuracy:", is_fraud_accuracy)
    print("is_fraud_precision:", is_fraud_precision)
    print("is_Recall:",is_fraud_recall)
    print("is_f1:",is_fraud_f1)
    print("is_confusion:",is_fraud_confusion)
     
    default_accuracy, default_precision,default_recall,default_f1,default_confusion= default()
    print("default_accuracy:", default_accuracy)
    print("default_precision:", default_precision)
    print("default_recall:",default_recall)
    print("default_f1:",default_f1)
    print("default_confusion:",default_confusion)


if __name__ == "__main__":
    main()

    
    



    











