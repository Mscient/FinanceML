
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
        self.model = joblib.load(
            MODELS_DIR + "/random_forest_default.pkl"
        )

        with open(FEATURES_DIR + "/features_default_set_e.json") as f:
            self.features = json.load(f)

    def predict(self, input_df):
        X = input_df[self.features]
        return self.model.predict(X)

    def predict_proba(self, input_df):
        X = input_df[self.features]
        return self.model.predict_proba(X)


def main(input_df: pd.DataFrame):
    predictor = Predict()
    predictions = predictor.predict(input_df)

    return {"predictions": predictions.tolist()}

 
    
if __name__ == "__main__":
    main()
