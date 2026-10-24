
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
if not (MODELS_DIR / "kmeans.pkl").exists():
    MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


FEATURES = [
    "monthly_income",
    "monthly_spending",
    "savings_balance",
    "existing_debt",
    "credit_util",
    "debt_to_income",
    "savings_to_income",
]


CLUSTER_PROFILES = {
    0: {
        "segment": "Stable, lower-debt",
        "description": (
            "Relatively low debt burden and lower "
            "credit utilization compared with the "
            "other clusters."
        ),
        "recommendations": [
            "Maintain consistent savings habits.",
            "Monitor debt and credit utilization.",
            "Maintain an emergency fund.",
        ],
    },
    1: {
        "segment": "High-income, high-savings",
        "description": (
            "Higher income and substantial savings "
            "compared with the other clusters."
        ),
        "recommendations": [
            "Maintain an appropriate emergency reserve.",
            "Review savings allocation and financial goals.",
            "Monitor credit utilization.",
        ],
    },
    2: {
        "segment": "Higher-debt, higher-utilization",
        "description": (
            "Higher debt burden relative to income and "
            "higher credit utilization compared with "
            "the other clusters."
        ),
        "recommendations": [
            "Review outstanding debt obligations.",
            "Explore ways to reduce credit utilization.",
            "Review monthly cash flow and savings.",
        ],
    },
}


class Predict:

    def __init__(self):
        self.model = joblib.load(MODELS_DIR / "kmeans.pkl")
        self.scaler = joblib.load(
            MODELS_DIR / "kmeans_scaler.pkl"
        )
        self.features = FEATURES

    def predict(self, input_df: pd.DataFrame):
        missing = [
            feature
            for feature in self.features
            if feature not in input_df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required features: {missing}"
            )

        X = input_df[self.features].copy()

        if X.isnull().any().any():
            raise ValueError("Input features cannot contain missing values.")

        if not X.map(lambda value: pd.notna(value) and
                     value != float("inf") and
                     value != float("-inf")).all().all():
            raise ValueError("Input features must be finite numbers.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def generate_response(
        self,
        input_df: pd.DataFrame,
        predictions,
    ):
        results = []

        for index, prediction in enumerate(predictions):
            cluster_id = int(prediction)
            profile = CLUSTER_PROFILES[cluster_id]

            financial_data = {
                feature: float(input_df.iloc[index][feature])
                for feature in self.features
            }

            results.append({
                "cluster_id": cluster_id,
                "segment": profile["segment"],
                "description": profile["description"],
                "financial_summary": financial_data,
                "recommendations": profile["recommendations"],
                "disclaimer": (
                    "This is customer segmentation, not a "
                    "creditworthiness or loan-default prediction."
                ),
            })

        return results


def main(input_df: pd.DataFrame = None):

    if input_df is None:
        sample = {
            "monthly_income": 85000.0,
            "monthly_spending": 35000.0,
            "savings_balance": 250000.0,
            "existing_debt": 100000.0,
            "credit_util": 0.35,
            "debt_to_income": 0.25,
            "savings_to_income": 0.40,
        }

        input_df = pd.DataFrame([sample])

    predictor = Predict()
    predictions = predictor.predict(input_df)

    results = predictor.generate_response(
        input_df,
        predictions,
    )

    return {"results": results}


if __name__ == "__main__":
    result = main()
    print("Cluster prediction:")
    print(result)
