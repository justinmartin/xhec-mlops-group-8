"""Load the abalone data and turn it into model-ready features."""

import numpy as np
import pandas as pd
from prefect import task
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

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


@task
def load_data(path: str) -> pd.DataFrame:
    """Read the abalone dataset from a CSV file."""
    return pd.read_csv(path)


@task
def filter_outliers(
    df: pd.DataFrame, min_height: float = 0, max_height: float = 0.4
) -> pd.DataFrame:
    """Remove the Height outliers found in the EDA (Height = 0 and Height > 0.4)."""
    return df[(df["Height"] > min_height) & (df["Height"] <= max_height)]


@task
def split_data(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into a train set and a test set (reproducible split)."""
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=42)
    return df_train, df_test


@task
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
