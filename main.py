from fastapi import FastAPI, UploadFile, File
from pipeline import train_pipeline
from predictor import make_prediction

app = FastAPI()

@app.get("/")
def home():
    return {"message": "AutoML Backend Running"}

@app.post("/train")
async def train(file: UploadFile = File(...), target_column: str = "target"):
    result = train_pipeline(file, target_column)
    return result

@app.post("/predict")
async def predict(data: dict):
    return make_prediction(data)