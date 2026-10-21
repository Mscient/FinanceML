
import json
from pathlib import Path

import joblib
import pandas as pd
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
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=1,
            n_jobs=-1,
        )
        self._model_files = {
            "default": "random_forest_default.pkl",
            "is_fraud": "random_forest_fraud.pkl",
        }

    # ------------------------------------------------------------------
    # Interface used by feature_set_validation (pre-split arrays/frames)
    # ------------------------------------------------------------------
    def train(self, target, model, X_train, y_train):
        """Fit on pre-split training data passed in by the caller."""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    # ------------------------------------------------------------------
    # Standalone training (reads CSVs, saves model to disk)
    # ------------------------------------------------------------------
    def train_and_save(self, target):
        """Full standalone training from processed CSVs."""
        data_cfg = {
            "default": {
                "features_file": "features_default_set_e.json",
            },
            "is_fraud": {
                "features_file": "features_is_fraud_set_e.json",
            },
        }
        config = data_cfg[target]

        with open(FEATURE_DIR / config["features_file"]) as f:
            features = json.load(f)

        X_train = pd.read_csv(PROCESSED_DATA_DIR / f"X_train_{target}.csv")[features]
        X_test  = pd.read_csv(PROCESSED_DATA_DIR / f"X_test_{target}.csv")[features]
        y_train = pd.read_csv(PROCESSED_DATA_DIR / f"y_train_{target}.csv")[target].squeeze()
        y_test  = pd.read_csv(PROCESSED_DATA_DIR / f"y_test_{target}.csv")[target].squeeze()

        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)

        accuracy  = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall    = recall_score(y_test, predictions, zero_division=0)
        f1        = f1_score(y_test, predictions, zero_division=0)
        confusion = confusion_matrix(y_test, predictions)

        model_file = self._model_files[target]
        joblib.dump(self.model, MODELS_DIR / model_file)

        print(f"\nTarget: {target}")
        print(f"Features: {len(features)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"Confusion matrix:\n{confusion}")
        print(f"Saved: {model_file}")

        return self.model


def main():
    trainer = Random()

    trainer.train_and_save("default")
    trainer.train_and_save("is_fraud")


if __name__ == "__main__":
    main()
