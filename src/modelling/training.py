"""Train and evaluate the model."""

import numpy as np
from prefect import task
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


@task
def train_model(x: csr_matrix, y: np.ndarray) -> LinearRegression:
    """Fit a linear regression on the training data."""
    model = LinearRegression()
    model.fit(x, y)
    return model


@task
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the root mean squared error (in number of rings)."""
    return root_mean_squared_error(y_true, y_pred)
