from fastapi import APIRouter,Depends
from app.schemas.loan_application import FraudPredictionRequest,DefaultPredictionRequest
from pydantic import BaseModel
import pandas as pd

 
from app.services.predict_default import main as predict_default_service
from app.services.predict_fraud import main as predict_fraud_service

router = APIRouter()


@router.post("/default")
def predict_default(loan_application:DefaultPredictionRequest):
    df = pd.DataFrame([loan_application.model_dump()])
    return  predict_default_service(df)

@router.post("/fraud")
def predict_fraud(loan_application:FraudPredictionRequest):
    df = pd.DataFrame([loan_application.model_dump()])
    return  predict_fraud_service(df)
 

