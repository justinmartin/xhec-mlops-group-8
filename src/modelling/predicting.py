"""Make predictions with a trained model."""

import numpy as np
from prefect import task
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression


@task
def predict(x: csr_matrix, model: LinearRegression) -> np.ndarray:
    """Predict the number of rings for each abalone in x."""
    return model.predict(x)
