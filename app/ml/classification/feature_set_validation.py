import pandas as pd
import numpy as np
from pathlib import Path
from feature_selection import Select_feature
from train_random import Random
from train_logistic import LogisticRegressionModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class feature_validation:

    def __init__(self):
        self.X_train = pd.read_csv("C:/Users/prash/Desktop/ML/Finance/FinanceML/app/data/X_train.csv")
        self.X_test  = pd.read_csv("C:/Users/prash/Desktop/ML/Finance/FinanceML/app/data/X_test.csv")
        self.y_train = pd.read_csv("C:/Users/prash/Desktop/ML/Finance/FinanceML/app/data/y_train.csv")
        self.y_test  = pd.read_csv("C:/Users/prash/Desktop/ML/Finance/FinanceML/app/data/y_test.csv")

    def validation(self, feature_set, target):
        """Split X_train into train/val using only the given feature_set columns,
        and return the correct 1-D label Series for target."""
        X = self.X_train[feature_set]
        y = self.y_train[target]

        X_train_main, X_val, y_train_main, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        return X_train_main, y_train_main, X_val, y_val

    def train_model(self, model, train_x, train_y, test_x, test_y):
        """Train model and return (accuracy, precision, recall, f1, confusion)."""
        from train_logistic import LogisticRegressionModel as LR
        if isinstance(model, LR):
            scaler = StandardScaler()
            # Use .values to avoid sklearn feature-name mismatch
            train_x = scaler.fit_transform(train_x.values if hasattr(train_x, 'values') else train_x)
            test_x  = scaler.transform(test_x.values if hasattr(test_x, 'values') else test_x)

        model.train(train_x, train_y)
        pred      = model.predict(test_x)
        accuracy  = accuracy_score(test_y, pred)
        precision = precision_score(test_y, pred, zero_division=0)
        recall    = recall_score(test_y, pred, zero_division=0)
        f1        = f1_score(test_y, pred, zero_division=0)
        conf      = confusion_matrix(test_y, pred)
        return accuracy, precision, recall, f1, conf


def validate_feature_sets(fv, target, feature_sets):
    """Run both models over all feature sets for a single target.

    Parameters
    ----------
    fv           : feature_validation instance
    target       : str  — "is_fraud" or "default"
    feature_sets : dict — {"a": [...], "b": [...], ...}
    """
    print(f"\n{'='*50}")
    print(f"TARGET: {target}")
    print(f"{'='*50}")

    set_names = list(feature_sets.keys())

    # ── Build train/val splits for every feature set (target is locked here) ──
    splits = {}
    for name, cols in feature_sets.items():
        splits[name] = fv.validation(cols, target)

    # ── Logistic Regression ──────────────────────────────────────────────────
    print(f"\n--- Logistic Regression | target={target} ---")
    logistic = LogisticRegressionModel()
    for name in set_names:
        tr_x, tr_y, te_x, te_y = splits[name]
        acc, prec, rec, f1, conf = fv.train_model(logistic, tr_x, tr_y, te_x, te_y)
        print(f"  set_{name}  acc:{acc:.4f}  prec:{prec:.4f}  rec:{rec:.4f}  f1:{f1:.4f}")
        print(f"  conf:\n{conf}\n")

    # ── Random Forest ────────────────────────────────────────────────────────
    print(f"--- Random Forest       | target={target} ---")
    random = Random()
    for name in set_names:
        tr_x, tr_y, te_x, te_y = splits[name]
        acc, prec, rec, f1, conf = fv.train_model(random, tr_x, tr_y, te_x, te_y)
        print(f"  set_{name}  acc:{acc:.4f}  prec:{prec:.4f}  rec:{rec:.4f}  f1:{f1:.4f}")
        print(f"  conf:\n{conf}\n")


def main():
    fv       = feature_validation()
    selector = Select_feature()

    set_a = fv.X_train.columns.tolist()
    set_b1, set_c1, set_d1, set_e1 = selector.run("is_fraud")
    set_b2, set_c2, set_d2, set_e2 = selector.run("default")

    fraud_feature_sets   = {"a": set_a, "b": set_b1, "c": set_c1, "d": set_d1, "e": set_e1}
    default_feature_sets = {"a": set_a, "b": set_b2, "c": set_c2, "d": set_d2, "e": set_e2}

    validate_feature_sets(fv, "is_fraud", fraud_feature_sets)
    validate_feature_sets(fv, "default",  default_feature_sets)


if __name__ == "__main__":
    main()