"""Load the abalone data and turn it into model-ready features."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer

TARGET_COL = "Rings"
CATEGORICAL_COLS = ["Sex"]
NUMERICAL_COLS = [
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
]


def load_data(path: str) -> pd.DataFrame:
    """Read the abalone dataset from a CSV file."""
    return pd.read_csv(path)


def filter_outliers(
    df: pd.DataFrame, min_height: float = 0, max_height: float = 0.4
) -> pd.DataFrame:
    """Remove the Height outliers found in the EDA (Height = 0 and Height > 0.4)."""
    return df[(df["Height"] > min_height) & (df["Height"] <= max_height)]


def extract_x_y(
    df: pd.DataFrame, dv: DictVectorizer | None = None, with_target: bool = True
) -> tuple[csr_matrix, np.ndarray | None, DictVectorizer]:
    """Build the feature matrix (and target) from the dataframe.

    The DictVectorizer one-hot encodes `Sex` (a string) and keeps the numerical
    columns as they are. Fit it on the training set only (dv=None), then reuse
    the fitted one for the test set and at prediction time.
    """
    dicts = df[CATEGORICAL_COLS + NUMERICAL_COLS].to_dict(orient="records")
    if dv is None:
        dv = DictVectorizer()
        x = dv.fit_transform(dicts)
    else:
        x = dv.transform(dicts)

    y = df[TARGET_COL].values if with_target else None
    return x, y, dv
