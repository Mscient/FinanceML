from fastapi import APIRouter
from app.schemas.loan_application import Loan_amount
import pandas as pd

from app.services.predict_amount import main as predict_amount_service

router = APIRouter()


@router.post("/loan_amount")
def predict_loan_amount(loan_application: Loan_amount):
    df = pd.DataFrame([loan_application.model_dump()])
    return predict_amount_service(df)
