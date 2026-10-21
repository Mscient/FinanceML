 
import sys
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parents[3]))

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
FEATURES_DIR  = PROCESSED_DIR / "selected_features"
MODELS_DIR    = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


from sklearn.pipeline import Pipeline


def train_svc(target: str, feature_set: list, pkl_name: str, scaler_name: str):
    if target == "is_fraud":
        X_train = pd.read_csv(PROCESSED_DIR / "X_train_is_fraud.csv")[feature_set]
        X_test  = pd.read_csv(PROCESSED_DIR / "X_test_is_fraud.csv")[feature_set]
        y_train = pd.read_csv(PROCESSED_DIR / "y_train_is_fraud.csv")["is_fraud"]
        y_test  = pd.read_csv(PROCESSED_DIR / "y_test_is_fraud.csv")["is_fraud"]
    else:
        X_train = pd.read_csv(PROCESSED_DIR / "X_train_default.csv")[feature_set]
        X_test  = pd.read_csv(PROCESSED_DIR / "X_test_default.csv")[feature_set]
        y_train = pd.read_csv(PROCESSED_DIR / "y_train_default.csv")["default"]
        y_test  = pd.read_csv(PROCESSED_DIR / "y_test_default.csv")["default"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", C=1.0, class_weight="balanced",
                    probability=True, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(f"\n[{target}] ({pkl_name})")
    print(f"  accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  f1       : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  confusion:\n{confusion_matrix(y_test, y_pred)}")

    joblib.dump(pipeline, MODELS_DIR / pkl_name)
    joblib.dump(pipeline.named_steps["scaler"], MODELS_DIR / scaler_name)
    print(f"  Saved: {pkl_name}, {scaler_name}")


def main():
    with open(FEATURES_DIR / "features_is_fraud_set_e.json") as f:
        fraud_features = json.load(f)
    with open(FEATURES_DIR / "features_default_set_e.json") as f:
        default_features = json.load(f)

    print("Training SVM — fraud (set_e) ...")
    train_svc("is_fraud", fraud_features, "svm_fraud.pkl", "svm_fraud_scaler.pkl")

    print("\nTraining SVM — default (set_e) ...")
    train_svc("default",  default_features, "svm_default.pkl", "svm_default_scaler.pkl")


if __name__ == "__main__":
    main()
