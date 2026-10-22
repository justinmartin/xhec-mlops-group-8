.PHONY: install docker prefect train deploy api lint

install:  # Install the dependencies and the pre-commit hooks
	uv sync
	uv run pre-commit install

docker:  # Build and run the container: API on :8000, Prefect UI on :4200, weekly retraining
	docker build -f Dockerfile.app -t abalone-app .
	docker run --rm -p 0.0.0.0:8000:8001 -p 0.0.0.0:4200:4201 -v "$(CURDIR)/data:/app/data" abalone-app

prefect:  # Start a local Prefect server and UI on :4200
	uv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
	uv run prefect server start

train:  # Train the model once on data/abalone.csv (needs `make prefect` running)
	cd src/modelling && uv run python main.py ../../data/abalone.csv

deploy:  # Serve the weekly retraining deployment (needs `make prefect` running)
	cd src/modelling && uv run python deployment.py

api:  # Run the API locally on :8000
	uv run uvicorn src.web_service.main:app --reload

lint:  # Run the same checks as the CI
	uv run pre-commit run --all-files
