"""FastAPI app serving the abalone age prediction model."""

from fastapi import FastAPI

from src.web_service.lib.inference import run_inference
from src.web_service.lib.models import Abalone, Prediction
from src.web_service.utils import LOCAL_OBJECTS_DIR, load_object

app = FastAPI(
    title="Abalone age prediction",
    description="Predict the age of an abalone from its physical measurements.",
)


@app.get("/")
def home() -> dict:
    """Health check."""
    return {"health_check": "App up and running!"}


@app.post("/predict", response_model=Prediction, status_code=201)
def predict(payload: Abalone) -> Prediction:
    """Predict the number of rings and the age of an abalone.

    The encoder and model are tiny, so they are loaded on each request: a
    retrained model is served as soon as the training flow saves it.
    """
    dv = load_object(LOCAL_OBJECTS_DIR / "dv.pkl")
    model = load_object(LOCAL_OBJECTS_DIR / "model.pkl")
    return run_inference(payload, dv, model)
