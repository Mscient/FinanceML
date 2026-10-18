import sys
from pathlib import Path
import json
import pandas as pd
import joblib
sys.path.append(str(Path(__file__).resolve().parents[3]))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR=str(Path(__file__).resolve().parents[2] / "models")
class Random:

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced_subsample',
            min_samples_leaf=1
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

    def load_data(self, target,feature_set):
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
    
    def feature(self, X_train):
        return pd.Series(self.model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

   

    def train(self, target, model,feature_set):
        X_train, X_test, y_train, y_test = self.load_data(target,feature_set)
        self.model.fit(X_train, y_train)
        dist1 = self.feature(X_train)
        if target == "default":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            joblib.dump(scaler, Path(MODELS_DIR) / "random_forest_scaler.pkl")

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        confusion = confusion_matrix(y_test, y_pred)
        return accuracy, precision, recall, f1, confusion, dist1



def main():
    model=Random()

    print("set_e")

    set_e_fraud = model.feature_dir / "features_is_fraud_set_e.json"
    set_e_default = model.feature_dir / "features_default_set_e.json"

    with open(set_e_fraud) as f:
        fraud_features = json.load(f)

    with open(set_e_default) as f:
        default_features = json.load(f)
    
    print("is_fraud")
    is_fraud_accuracy, is_fraud_precision,is_fraud_recall,is_fraud_f1,is_confusion,dist1=model.train("is_fraud",model,fraud_features)
    print("is_fraud_accuracy:", is_fraud_accuracy)
    print("is_fraud_precision:", is_fraud_precision)
    print("is_Recall:",is_fraud_recall)
    print("is_f1:",is_fraud_f1)
    print("is_confusion:",is_confusion)

     
    print("default")
    default_accuracy, default_precision,default_recall,default_f1,default_confusion,dist2=model.train("default",model,default_features)
    print("default_accuracy:", default_accuracy)
    print("default_precision:", default_precision)
    print("default_recall:",default_recall)
    print("default_f1:",default_f1)
    print("default_confusion:",default_confusion)


    set_d_fraud = model.feature_dir / "features_is_fraud_set_d.json"
    set_d_default = model.feature_dir / "features_default_set_d.json"

    with open(set_d_fraud) as f:
        fraud_features = json.load(f)

    with open(set_e_default) as f:
        default_features = json.load(f)
    
    


    print("SET_D")
    print("fraud")
    is_fraud_accuracy, is_fraud_precision,is_fraud_recall,is_fraud_f1,is_confusion,dist1=model.train("is_fraud",model,fraud_features)
    print("is_fraud_accuracy:", is_fraud_accuracy)
    print("is_fraud_precision:", is_fraud_precision)
    print("is_Recall:",is_fraud_recall)
    print("is_f1:",is_fraud_f1)
    print("is_confusion:",is_confusion)

     
    print("default")
    default_accuracy, default_precision,default_recall,default_f1,default_confusion,dist2=model.train("default",model,default_features)
    print("default_accuracy:", default_accuracy)
    print("default_precision:", default_precision)
    print("default_recall:",default_recall)
    print("default_f1:",default_f1)
    print("default_confusion:",default_confusion)


    MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model.model, MODELS_DIR / "random_forest_fraud.pkl")
    joblib.dump(model.model, MODELS_DIR / "random_forest_default.pkl")
    

if __name__ == "__main__":
    main()



    

    