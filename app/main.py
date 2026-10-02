from fastapi import FastAPI,APIRouter
 

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}




router=APIRouter()

router.include()
router.include()

