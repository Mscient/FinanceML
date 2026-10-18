from fastapi import APIRouter,Depends
from app.schemas.loan_application import LoanApplication,FraudResponse,DefaultResponse
from pydantic import BaseModel
import pandas as pd
from app.services.predict_default import main as predict_default
from app.services.predict_fraud import main as predict_fraud

router = APIRouter()



@router.post("/default")
async def predict_default(loan_application:LoanApplication):
    return predict_default.main(loan_application.default)

@router.post("/fraud")
async def predict_fraud(loan_application:LoanApplication):
    return predict_fraud.main(loan_application.fraud)
 
    
