"""FastAPI app serving the abalone age prediction model."""

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from src.web_service.lib.inference import run_inference
from src.web_service.lib.models import Abalone, Prediction
from src.web_service.utils import load_latest_artifacts

app = FastAPI(
    title="Abalone age prediction",
    description="Predict the age of an abalone from its physical measurements.",
)


def get_artifacts() -> tuple[DictVectorizer, LinearRegression]:
    """Provide the latest encoder and model, or a 503 if none is trained yet."""
    try:
        return load_latest_artifacts()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail="Model not trained yet") from e


Artifacts = Annotated[tuple[DictVectorizer, LinearRegression], Depends(get_artifacts)]


@app.get("/")
def home() -> dict:
    """Health check."""
    return {"health_check": "App up and running!"}


@app.post("/predict", response_model=Prediction, status_code=201)
def predict(payload: Abalone, artifacts: Artifacts) -> Prediction:
    """Predict the number of rings and the age of an abalone."""
    dv, model = artifacts
    return run_inference(payload, dv, model)
