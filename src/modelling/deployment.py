"""Deploy the training flow so that the model is retrained regularly."""

from pathlib import Path

from main import main

# src/modelling/deployment.py -> data/abalone.csv
TRAINSET_PATH = Path(__file__).resolve().parents[2] / "data" / "abalone.csv"

if __name__ == "__main__":
    main.serve(
        name="abalone-retraining",
        cron="0 0 * * 0",
        parameters={"trainset_path": str(TRAINSET_PATH)},
    )
