<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/justinmartin/xhec-mlops-group-8/blob/main/.pre-commit-config.yaml)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![Prefect](https://img.shields.io/badge/orchestration-Prefect-070E10?logo=prefect&logoColor=white)](https://www.prefect.io)
</div>

## Team

| Name | GitHub |
|------|--------|
| Eleanor Liard | [@ela-du-91](https://github.com/ela-du-91) |
| Maxime Despreaux | [@MaximeDespreaux](https://github.com/MaximeDespreaux) |
| Zixiao Guo | [@ZixiaoGuo-China](https://github.com/ZixiaoGuo-China) |
| Justin Martin | [@justinmartin](https://github.com/justinmartin) |

## Project Overview

Determining the age of an abalone (a sea snail) normally means cutting its shell, staining it and counting the rings under a microscope. This project predicts the number of rings, and therefore the age (`age = rings + 1.5` years), from easy-to-take physical measurements.

It covers the full MLOps loop:
- **Training pipeline**: a Prefect flow that loads the data, preprocesses it, trains a linear regression and saves the model and encoder.
- **Periodic retraining**: a Prefect deployment that reruns the flow every Friday at 6pm UTC.
- **Prediction API**: a FastAPI service with Pydantic input validation, which picks up a retrained model without restarting.
- **Docker**: one container runs the API, the Prefect server and the retraining deployment.

**Dataset**: [Abalone dataset on Kaggle](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset). Download `abalone.csv` into `data/abalone.csv` (the `data/` folder is not versioned).

## Project Structure

```
├── bin/run_services.sh       # Container entrypoint: Prefect server + retraining deployment + API
├── Dockerfile.app            # Image for the API and the Prefect services
├── Makefile                  # Shortcuts for install, Docker, training, API and lint
├── notebooks/                # EDA and modelling experiments (tracked with MLflow)
└── src/
    ├── modelling/            # Prefect training flow (main.py) and retraining deployment (deployment.py)
    └── web_service/
        ├── main.py           # FastAPI app
        ├── utils.py          # Loads the pickled encoder and model
        ├── lib/models.py     # Pydantic request / response models
        ├── lib/inference.py  # Runs the model on a request
        └── local_objects/    # Trained encoder (dv.pkl) and model (model.pkl)
```

## Setup

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/getting-started/installation/) and, to run the container, [Docker](https://docs.docker.com/get-docker/).

```bash
make install   # uv sync + pre-commit hooks
make docker    # API on http://localhost:8000/docs, Prefect UI on http://localhost:4200
```

That's all you need to try the project. The other targets, each detailed below:

| Command | What it does |
|---|---|
| `make prefect` | Start a local Prefect server and UI on :4200 |
| `make train` | Train the model once on `data/abalone.csv` |
| `make deploy` | Serve the weekly retraining deployment |
| `make api` | Run the API locally on :8000 |
| `make lint` | Run the same checks as the CI |

No `make` (e.g. on Windows)? Every target is a one-liner from the `Makefile` that you can copy, or use the commands in the sections below.

## Training Pipeline with Prefect

The training code in `src/modelling` is a Prefect **flow** (`main.py`) made of **tasks** (load, preprocess, train, evaluate, save).

1. **Start the Prefect server and UI** (first terminal, keep it open):
   ```bash
   uv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
   uv run prefect server start
   ```

2. **Train the model once** (second terminal):
   ```bash
   cd src/modelling
   uv run python main.py ../../data/abalone.csv
   ```
   The model and the encoder are saved in `src/web_service/local_objects/`.

3. **Deploy the flow for regular retraining** (every Friday at 6pm UTC, keep it running):
   ```bash
   cd src/modelling
   uv run python deployment.py
   ```
   To trigger a run now: in the UI (*Deployments* → *abalone-retraining* → *Run*), or with
   `uv run prefect deployment run 'Train model/abalone-retraining'`.

4. **Visualize the flow runs**: open http://localhost:4200 to see each run, its tasks and logs.

## Prediction API

### Run it with Docker (recommended)

From the repository root, with `data/abalone.csv` downloaded:

```bash
docker build -f Dockerfile.app -t abalone-app .
docker run -p 0.0.0.0:8000:8001 -p 0.0.0.0:4200:4201 -v "$(pwd)/data:/app/data" abalone-app
```

On Windows PowerShell, use `-v "${PWD}/data:/app/data"`. Stop any local Prefect server first, since it also uses port 4200.

The container starts:
- the **API** on http://localhost:8000 (interactive docs at http://localhost:8000/docs),
- the **Prefect UI** on http://localhost:4200,
- the **retraining deployment**, which retrains on the mounted `data/abalone.csv` every Friday at 6pm UTC. The API serves the new model as soon as it is saved. Trigger a run from the UI to try it.

The image ships with the model committed in `local_objects/`, so the API works even without the data. Only retraining needs it.

### Run it locally

```bash
uv run uvicorn src.web_service.main:app --reload
```

### Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sex": "M", "length": 0.455, "diameter": 0.365, "height": 0.095,
       "whole_weight": 0.514, "shucked_weight": 0.2245,
       "viscera_weight": 0.101, "shell_weight": 0.15}'
```

```json
{"rings": 8.86, "age": 10.36}
```

Measurements use the dataset's units. `sex` must be `M`, `F` or `I` (infant), and every measurement must be positive. Invalid inputs return a `422` that explains what is wrong.

## Notebooks

The EDA and modelling experiments are in `notebooks/` (open them with the project's `.venv` kernel). Experiments are tracked with MLflow; to browse them:

```bash
cd notebooks
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open http://localhost:5000.

## Development

### Managing Dependencies

Dependencies are declared in `pyproject.toml` and managed with uv:
```bash
uv add <package>
uv add --dev <package>
```
uv updates `pyproject.toml` and `uv.lock` automatically. Always commit `uv.lock`, and run `uv sync` after pulling changes.

### Code Quality

Pre-commit hooks run automatically on every `git commit`:
- **Ruff**: linting (with auto-fix) and formatting
- **File checks**: trailing whitespace, end of file, valid YAML / TOML / JSON, no large files
- **uv-lock**: checks that `uv.lock` is in sync with `pyproject.toml`

If a hook modifies files, the commit is stopped: `git add` the changes and commit again.

The same checks run in **GitHub Actions CI** on every push and pull request, so run `uv run pre-commit run --all-files` before pushing.
