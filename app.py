import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from http import HTTPStatus

from dotenv import load_dotenv, find_dotenv

from app_schema import CreditApplication
from modelling.preprocess.data_preprocessor import Preprocessor
from modelling.train.model_trainer import RfModelTrainer

load_dotenv(find_dotenv())

MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "")
DATA_PATH = os.getenv("DATA_PATH", "")

# Load the pre-saved model
with open(MODEL_SAVE_PATH, "rb") as model_file:
    model = pickle.load(model_file)

app = FastAPI()

@app.get("/")
@app.get("/home")
def home():
    return {
        "status": HTTPStatus.OK,
        "message": "Welcome to the classification application"
    }

preprocessor = Preprocessor(data_path=DATA_PATH)

@app.get("/train")
def train():
    try:
        data = preprocessor.read_data()
        X_scaled, y = preprocessor.preprocess(data=data)
        trainer = RfModelTrainer(X=X_scaled, y=y)
        trainer.run_model()
        return {
            "status": HTTPStatus.OK,
            "message": "Model trained successfully"
        }
    except Exception as e:
        return {
            "status": HTTPStatus.BAD_REQUEST,
            "message": "Model training failed",
            "data": str(e),
        }

@app.post("/predict")
def predict(input_data: CreditApplication):
    try:
        # Convert the received input into a dictionary
        input_dict = input_data.dict()

        features = preprocessor.predict_preprocess(input_data=input_dict)

        # Perform the prediction using the preloaded model
        prediction = model.predict(features)
        print("prediction: ", prediction)

        # Assuming the model returns a classification (e.g., 0 for no default, 1 for default)
        return {
            "status": HTTPStatus.OK,
            "prediction": int(prediction[0])
        }

    except Exception as e:
        # In case of an error, raise an HTTP Exception with a detailed error message
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
