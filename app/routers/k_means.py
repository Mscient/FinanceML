from fastapi import APIRouter,Depends
from app.schemas.loan_application import Cluster,ClusterResponse
from pydantic import BaseModel
import pandas as pd

 
from app.services.cluster import main as predict_default_service
 

router = APIRouter()


@router.post("/cluster")
def predict_cluster(loan_application: Cluster):
    df = pd.DataFrame([loan_application.model_dump()])
    return predict_default_service(df)
 
 
