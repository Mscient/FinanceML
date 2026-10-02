#input

import pandas as pd
from pathlib import Path
import joblib
import numpy as np

X_train = pd.read_csv(Path(__file__).parent /"../data/X_train.csv")
X_test=pd.read_csv(Path(__file__).parent /"../data/X_test.csv")
y_train=pd.read_csv(Path(__file__).parent /"../data/y_train.csv")
y_test=pd.read_csv(Path(__file__).parent /"../data/y_test.csv")


#Model
from sklearn.linear_model import LinearRegression
Linear=LinearRegression()

#Traning 
Linear.fit(X_train,y_train)


#predictions
y_pred=Linear.predict(X_test)

#Evaluation Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")




#save model
model_path = Path(__file__).parent / "../models/linear_regression.pkl"
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(Linear, model_path)
