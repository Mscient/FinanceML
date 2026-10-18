
import pandas as pd 
from pathlib import Path 
import json
import joblib
import sys
sys.path.append(str(Path(__file__).resolve().parents[3]))
MODELS_DIR=str(Path(__file__).resolve().parents[2] / "models")

class Predict:

    def __init__(self):
        self.model = joblib.load(MODELS_DIR + "/random_forest_default.pkl")
        self.scaler = joblib.load(MODELS_DIR + "/random_forest_scaler.pkl")
    
        with open(MODELS_DIR + "/features_default_set_e.json") as f:
            self.features = json.load(f)

    def predict(self,input_df):
        x_test=input_df[self.features]
        X_scaled = self.scaler.transform(x_test)
        return self.model.predict(X_scaled)

    def predict_proba(self,input_df):
        x_test=input_df[self.features]
        X_scaled = self.scaler.transform(x_test)
        return self.model.predict_proba(X_scaled)
    

def main():
    predict=Predict()
    input_df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "raw" / "credit_card_transactions.csv")
    result=predict.predict(input_df)
    print(result)
    
    
if __name__ == "__main__":
    main()
