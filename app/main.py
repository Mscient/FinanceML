from fastapi import FastAPI,APIRouter
from app.routers.classificatiom import router as classification_router
from app.routers.regression import router as regression_router



app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


app.include_router(
    classification_router,
    prefix="/api/predict",
    tags=["Classification"]
)


app.include_router(
    regression_router,
    prefix="/api",
    tags=["Regression"]
)


