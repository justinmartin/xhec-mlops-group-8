"""Training flow: read the data, preprocess it, train a model and save it."""

import argparse
from pathlib import Path

from predicting import predict
from prefect import flow
from preprocessing import extract_x_y, filter_outliers, load_data, split_data
from training import evaluate_model, train_model
from utils import pickle_object

# src/modelling/main.py -> src/web_service/local_objects
LOCAL_OBJECTS_DIR = (
    Path(__file__).resolve().parents[1] / "web_service" / "local_objects"
)


@flow(name="Train model", log_prints=True)
def main(trainset_path: str) -> None:
    """Train a model using the data at the given path and save the model (pickle)."""
    # Read data
    df = filter_outliers(load_data(trainset_path))
    df_train, df_test = split_data(df)

    # Preprocess data (the encoder is fitted on the training set only)
    x_train, y_train, dv = extract_x_y(df_train)
    x_test, y_test, _ = extract_x_y(df_test, dv=dv)

    # Train model
    model = train_model(x_train, y_train)
    rmse = evaluate_model(y_test, predict(x_test, model))
    print(f"Test RMSE: {rmse:.2f} rings")

    # Pickle encoder and model for the web service
    pickle_object(dv, LOCAL_OBJECTS_DIR / "dv.pkl")
    pickle_object(model, LOCAL_OBJECTS_DIR / "model.pkl")
    print(f"Model and encoder saved to {LOCAL_OBJECTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using the data at the given path."
    )
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()
    main(args.trainset_path)
