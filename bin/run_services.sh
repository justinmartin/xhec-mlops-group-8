#!/bin/bash
# Start the Prefect server, the weekly retraining deployment and the prediction API.
set -e

prefect server start --host 0.0.0.0 --port 4201 &

# The deployment can only be registered once the Prefect API is up
until python -c "import urllib.request; urllib.request.urlopen('${PREFECT_API_URL}/health')" 2>/dev/null; do
    sleep 1
done
(cd src/modelling && python deployment.py) &

exec uvicorn src.web_service.main:app --host 0.0.0.0 --port 8001
