
import pandas as pd 
from pathlib import Path 
import json
import joblib
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
MODELS_DIR = str(Path(__file__).resolve().parents[1] / "models")
FEATURES_DIR = str(Path(__file__).resolve().parents[1] / "data" / "processed" / "selected_features")

class Predict:
    def __init__(self):
        self.model = joblib.load(MODELS_DIR + "/random_laon_amount.pkl")
    
        with open(FEATURES_DIR + "/features_loan_amount_lasso_set_e.json",encoding="utf-8") as f:
            self.features = json.load(f)

    def predict(self, input_df: pd.DataFrame):
        df = input_df.copy()
        if "spending_to_income" not in df.columns or df["spending_to_income"].isna().any():
            if "monthly_spending" in df.columns and "monthly_income" in df.columns:
                df["spending_to_income"] = df["monthly_spending"] / df["monthly_income"].replace(0, float("nan"))
                df["spending_to_income"] = df["spending_to_income"].fillna(0.0)
        x_test = df[self.features]
        return self.model.predict(x_test)   

    def predict_proba(self, input_df: pd.DataFrame):
        df = input_df.copy()
        if "spending_to_income" not in df.columns or df["spending_to_income"].isna().any():
            if "monthly_spending" in df.columns and "monthly_income" in df.columns:
                df["spending_to_income"] = df["monthly_spending"] / df["monthly_income"].replace(0, float("nan"))
                df["spending_to_income"] = df["spending_to_income"].fillna(0.0)
        x_test = df[self.features]
        return self.model.predict_proba(x_test)

def main(input_df: pd.DataFrame = None):
    if input_df is None:
        sample = {
            "monthly_income": 85000.0,
            "annual_income": 1020000.0,
            "monthly_spending": 35000.0,
            "num_transactions_per_month": 25.0,
            "avg_transaction_amount": 1400.0,
            "credit_score": 720.0,
            "savings_balance": 250000.0,
            "existing_debt": 100000.0,
            "loan_purpose_auto": 0.0,
            "loan_purpose_education": 0.0,
            "loan_purpose_personal": 1.0,
            "loan_purpose_home": 0.0,
        }
        input_df = pd.DataFrame([sample])

    predict = Predict()
    predictions = predict.predict(input_df)
    
    return {
        "predictions": predictions.tolist()
    }


if __name__ == "__main__":
    result = main()
    print("Test Prediction:", result)
