# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.compose import ColumnTransformer
from starter.ml.data import process_data
from starter.ml.model import inference

app = FastAPI()

class inference_api(BaseModel):
    age: int
    workclass: str = Field(alias="workclass")
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")
    salary: str

@app.get("/")
async def root():
    return {"Welcome to Model Inference API"}

@app.post("/inference")
async def inference_api(request: inference_api):
    model = joblib.load("model/model.joblib")
    encoder = joblib.load("model/encoder.joblib")
    lb = joblib.load("model/lb.joblib")
    data = pd.DataFrame([request.dict(by_alias=True)])
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_test, y_test, encoder, lb = process_data(
        data, categorical_feature=cat_features, label="salary", encoder=encoder, lb=lb, training=False
    )

    preds = inference(model, X_test)

    return {"prediction": preds.tolist()}

    

    

   





