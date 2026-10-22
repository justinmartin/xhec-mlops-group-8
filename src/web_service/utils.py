"""Load the encoder and model saved by the training flow."""

import pickle
from pathlib import Path
from typing import Any

LOCAL_OBJECTS_DIR = Path(__file__).resolve().parent / "local_objects"


def load_object(path: Path) -> Any:
    """Load a pickled object (model, encoder...) from disk."""
    with path.open("rb") as f:
        return pickle.load(f)
