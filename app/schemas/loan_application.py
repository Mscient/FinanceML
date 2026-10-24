from pydantic import BaseModel
from typing import Any

class Loan(BaseModel):
    amount:int

class  FraudPredictionRequest(BaseModel):
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
    monthly_spending: float
    loan_term_months: float
    loan_amount: float
    loan_purpose_personal: float
    num_transactions_per_month: float
    salary_to_savings: float
    txn_failed_attempts_24h_sum: float
    credit_score: float
    debt_to_income: float
    annual_income: float
    monthly_income: float
    salary_to_loan: float
    savings_balance: float
    credit_util: float
    years_employed: float
    loan_purpose_home: float
    avg_transaction_amount: float
    age: float
    existing_debt: float
    savings_to_income: float
    debt_to_saving_ratio: float
    monthly_emi: float

class Loan_amount(BaseModel):
    monthly_income: float
    annual_income: float
    monthly_spending: float
    num_transactions_per_month: float = 20.0
    avg_transaction_amount: float
    credit_score: float
    savings_balance: float = 0.0
    existing_debt: float = 0.0
    spending_to_income: float | None = None
    loan_purpose_auto: float = 0.0
    loan_purpose_education: float = 0.0
    loan_purpose_personal: float = 0.0
    loan_purpose_home: float = 0.0

class Cluster(BaseModel):
    monthly_income:float
    monthly_spending:float
    savings_balance:float
    existing_debt:float
    credit_util:float
    debt_to_income:float
    savings_to_income:float

#RESPONSE

class LoanResponse(BaseModel):
    amount:int


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

class ClusterResponse(BaseModel):
    pass