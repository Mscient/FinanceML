
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
        self.model  = joblib.load(MODELS_DIR + "/random_forest_default.pkl")
        self.scaler = joblib.load(MODELS_DIR + "/random_forest_scaler.pkl")  
    
        with open(FEATURES_DIR + "/features_default_set_e.json") as f:
            self.features = json.load(f)

    def predict(self, input_df):
        x_test   = input_df[self.features]
        X_scaled = pd.DataFrame(self.scaler.transform(x_test), columns=self.features)
        return self.model.predict(X_scaled)

    def predict_proba(self, input_df):
        x_test   = input_df[self.features]
        X_scaled = pd.DataFrame(self.scaler.transform(x_test), columns=self.features)
        return self.model.predict_proba(X_scaled)
    

def main():
    predict = Predict()
    input_df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "processed" / "X_test_default.csv")
    result = predict.predict(input_df)
    return result
    
    
if __name__ == "__main__":
    main()
