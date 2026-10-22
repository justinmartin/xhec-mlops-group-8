FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /bin/uv

WORKDIR /app

# Install the dependencies first so this layer is cached across code changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src ./src
COPY bin ./bin

# Inside the container the API listens on 8001 and Prefect on 4201;
# the Prefect UI is reached from the host on port 4200 (see README)
ENV PATH="/app/.venv/bin:$PATH" \
    PREFECT_API_URL=http://127.0.0.1:4201/api \
    PREFECT_UI_API_URL=http://localhost:4200/api

EXPOSE 8001 4201

CMD ["bash", "bin/run_services.sh"]
