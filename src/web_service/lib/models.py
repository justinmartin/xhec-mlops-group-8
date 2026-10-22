"""Pydantic models for the prediction API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Abalone(BaseModel):
    """Physical measurements of one abalone, in the dataset's units.

    Each field is serialized under its dataset column name (e.g. `Whole weight`)
    so the training preprocessing can be reused as is.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "sex": "M",
                    "length": 0.455,
                    "diameter": 0.365,
                    "height": 0.095,
                    "whole_weight": 0.514,
                    "shucked_weight": 0.2245,
                    "viscera_weight": 0.101,
                    "shell_weight": 0.15,
                }
            ]
        }
    )

    sex: Literal["M", "F", "I"] = Field(
        serialization_alias="Sex", description="M (male), F (female) or I (infant)"
    )
    length: float = Field(gt=0, serialization_alias="Length")
    diameter: float = Field(gt=0, serialization_alias="Diameter")
    height: float = Field(gt=0, serialization_alias="Height")
    whole_weight: float = Field(gt=0, serialization_alias="Whole weight")
    shucked_weight: float = Field(gt=0, serialization_alias="Shucked weight")
    viscera_weight: float = Field(gt=0, serialization_alias="Viscera weight")
    shell_weight: float = Field(gt=0, serialization_alias="Shell weight")


class Prediction(BaseModel):
    """Predicted number of rings and the corresponding age (in years)."""

    rings: float
    age: float
