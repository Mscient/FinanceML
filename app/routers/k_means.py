from fastapi import APIRouter
from app.schemas.loan_application import Cluster, ClusterResponse
import pandas as pd

from app.services.cluster import main as predict_cluster_service

router = APIRouter()


@router.post("/cluster", response_model=ClusterResponse)
def predict_cluster(loan_application: Cluster):
    df = pd.DataFrame([loan_application.model_dump()])
    return predict_cluster_service(df)
