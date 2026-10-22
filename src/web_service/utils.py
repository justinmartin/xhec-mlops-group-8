"""Load the encoder and model saved by the training flow."""

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

LOCAL_OBJECTS_DIR = Path(__file__).resolve().parent / "local_objects"
DV_PATH = LOCAL_OBJECTS_DIR / "dv.pkl"
MODEL_PATH = LOCAL_OBJECTS_DIR / "model.pkl"


def load_object(path: Path) -> Any:
    """Load a pickled object (model, encoder...) from disk."""
    with path.open("rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _load_artifacts(
    version: tuple[int, int],
) -> tuple[DictVectorizer, LinearRegression]:
    """Load the encoder and model; `version` is only used as the cache key."""
    return load_object(DV_PATH), load_object(MODEL_PATH)


def load_latest_artifacts() -> tuple[DictVectorizer, LinearRegression]:
    """Return the encoder and model, reloading them when a retraining overwrote them.

    Raises:
        FileNotFoundError: If the model has not been trained yet.
    """
    version = (DV_PATH.stat().st_mtime_ns, MODEL_PATH.stat().st_mtime_ns)
    return _load_artifacts(version)
