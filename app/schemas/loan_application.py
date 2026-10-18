
from pydantic import BaseModel
from typing import Any

class Loan(BaseModel):
    amount:int

class FraudPredictionRequest(BaseModel):
    spending_to_income:float
    txn_transaction_count:float
    txn_failed_attempts_24h_max:float
    txn_distance_from_usual_location_std:float
    txn_distance_from_usual_location_min:float
    txn_is_new_device_max:float
    txn_failed_attempts_24h_mean:float
    txn_transaction_amount_std:float
    txn_transaction_hour_max:float
    txn_transaction_hour_min:float
    txn_transactions_last_hour_std:float
    txn_merchant_risk_score_min:float
    txn_transaction_hour_std:float
    avg_transaction_amount:float
    txn_merchant_risk_score_max:float
    txn_distance_from_usual_location_max:float
    txn_transaction_amount_sum:float
    txn_is_international_mean:float
    txn_transaction_hour_sum:float
    txn_is_international_max:float
    txn_failed_attempts_24h_std:float
    txn_merchant_risk_score_sum:float
    txn_is_international_sum:float
    txn_distance_from_usual_location_sum:float
    txn_transaction_amount_max:float
    txn_is_new_device_sum:float
    txn_failed_attempts_24h_sum:float
    txn_merchant_risk_score_std:float
    txn_transaction_amount_mean:float
    txn_merchant_risk_score_mean:float
    txn_distance_from_usual_location_mean:float
    

class DefaultPredictionRequest(BaseModel):
    txn_merchant_risk_score_max:float
    txn_distance_from_usual_location_max:float
    txn_transaction_amount_sum:float
    txn_is_international_mean:float
    txn_transaction_hour_sum:float
    txn_is_international_max:float
    txn_failed_attempts_24h_std:float
    txn_merchant_risk_score_sum:float
    txn_is_international_sum:float
    txn_distance_from_usual_location_sum:float
    txn_transaction_amount_max:float
    txn_is_new_device_sum:float
    txn_failed_attempts_24h_sum:float
    txn_merchant_risk_score_std:float
    txn_transaction_amount_mean:float
    txn_merchant_risk_score_mean:float
    txn_distance_from_usual_location_mean:float

class FraudResponse(BaseModel):
    is_fraud:bool
    prediction:int

class DefaultResponse(BaseModel):
    default:bool
    prediction:int


class LoanApplication(BaseModel):
    loan:Loan
    fraud:FraudPredictionRequest
    default:DefaultPredictionRequest
    