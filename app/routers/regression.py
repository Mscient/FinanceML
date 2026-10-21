from fastapi import APIRouter,Depends
from app.schemas.loan_application import LoanApplication,Loan_amount,LoanResponse
from pydantic import BaseModel
import pandas as pd

 
from app.services.predict_amount import main as predict_default_service
 

router = APIRouter()



@router.post("/loan_amount")
@router.post("/laon_amount")
def predict_loan_amount(loan_application: Loan_amount):
    df = pd.DataFrame([loan_application.model_dump()])
    return predict_default_service(df)
 
 
