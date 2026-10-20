import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURE_DIR = PROCESSED_DATA_DIR / "selected_features"

 
def _scale_y(y):
    std = y.std()
    return y / std, std


# ---------------------------------------------------------------------------
# Regression metrics  (no confusion matrix / predict_proba — this is regression)
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    # Mean Absolute Percentage Error
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mae, rmse, r2, mape


# ---------------------------------------------------------------------------
# Feature Validation class
# ---------------------------------------------------------------------------
class feature_validation:

    def __init__(self):
        self.X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train_loan_amount.csv")
        self.X_test  = pd.read_csv(PROCESSED_DATA_DIR / "X_test_loan_amount.csv")
        self.y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train_loan_amount.csv").squeeze()
        self.y_test  = pd.read_csv(PROCESSED_DATA_DIR / "y_test_loan_amount.csv").squeeze()

        # Set A = all available features
        self.set_a = self.X_train.columns.tolist()

    def validation_split(self, feature_set):
        """Hold-out split from training data for feature-set comparison.
        No stratify — loan_amount is continuous (regression).
        """
        X = self.X_train[feature_set]
        y = self.y_train
        return train_test_split(X, y, test_size=0.2, random_state=42)
 
    def train_model(self, model, X_train, y_train, X_val, y_val):
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)

        # Lasso / ElasticNet need y scaled to match alpha magnitude
        if isinstance(model, (Lasso, ElasticNet)):
            y_scaled, y_std = _scale_y(y_train)
            model.fit(X_train_s, y_scaled)
            y_pred_scaled = model.predict(X_val_s)
            y_pred = y_pred_scaled * y_std           
        else:
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_val_s)

        return regression_metrics(y_val.values, y_pred)

 
def validate_feature_sets(fv, model_name, model_factory, feature_sets):
    print(f"\n{'-'*55}")
    print(f"  {model_name}")
    print(f"{'-'*55}")
    print(f"  {'Set':<8}  {'MAE':>14}  {'RMSE':>14}  {'R2':>7}  {'MAPE':>8}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*7}  {'-'*8}")

    for name, cols in feature_sets.items():
        X_tr, X_val, y_tr, y_val = fv.validation_split(cols)
        model = model_factory()
        mae, rmse, r2, mape = fv.train_model(model, X_tr, y_tr, X_val, y_val)
        print(f"  set_{name:<5}  {mae:>14,.0f}  {rmse:>14,.0f}  {r2:>7.4f}  {mape:>7.2f}%")


 
def main():
    fv = feature_validation()

    # Load feature sets saved by feature_Set.py
    def load(path):
        with open(FEATURE_DIR / path) as f:
            return json.load(f)

    sets = {
        "a":         fv.set_a,
        "ridge_d":   load("features_loan_amount_ridge_set_d.json"),
        "ridge_e":   load("features_loan_amount_ridge_set_e.json"),
        "lasso_d":   load("features_loan_amount_lasso_set_d.json"),
        "lasso_e":   load("features_loan_amount_lasso_set_e.json"),
        "elastic_d": load("features_loan_amount_elastic_set_d.json"),
        "elastic_e": load("features_loan_amount_elastic_set_e.json"),
    }

    print("\n" + "="*55)
    print("  FEATURE SET VALIDATION  —  target: loan_amount")
    print("="*55)

    # Linear Regression
    validate_feature_sets(
        fv,
        "Linear Regression",
        lambda: LinearRegression(),
        sets,
    )

    # Ridge Regression
    validate_feature_sets(
        fv,
        "Ridge (alpha=1.0)",
        lambda: Ridge(alpha=1.0),
        sets,
    )

    # Lasso Regression
    validate_feature_sets(
        fv,
        "Lasso (alpha=1.0)",
        lambda: Lasso(alpha=1.0, max_iter=10000),
        sets,
    )

    # ElasticNet
    validate_feature_sets(
        fv,
        "ElasticNet (alpha=1.0, l1=0.5)",
        lambda: ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000),
        sets,
    )

    # Random Forest
    validate_feature_sets(
        fv,
        "Random Forest (n=200)",
        lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        sets,
    )


if __name__ == "__main__":
    main()