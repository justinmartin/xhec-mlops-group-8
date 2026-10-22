"""Run the trained model on API inputs."""

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from src.modelling.preprocessing import extract_x_y
from src.web_service.lib.models import Abalone, Prediction

# Age (in years) = Rings + 1.5, see the dataset description
RINGS_TO_AGE_OFFSET = 1.5


def run_inference(
    abalone: Abalone, dv: DictVectorizer, model: LinearRegression
) -> Prediction:
    """Predict the number of rings and the age of an abalone.

    Features are built with the training preprocessing, so both stay in sync.
    """
    df = pd.DataFrame([abalone.model_dump(by_alias=True)])
    x, _, _ = extract_x_y.fn(df, dv=dv, with_target=False)
    rings = float(model.predict(x)[0])
    return Prediction(rings=rings, age=rings + RINGS_TO_AGE_OFFSET)
