# ---- Stage 1: grab the uv binary ----
FROM ghcr.io/astral-sh/uv:latest AS uv

# ---- Stage 2: build the runtime image ----
FROM python:3.12-slim AS runtime

# Copy uv from the first stage
COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first (cache-friendly layer ordering)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Copy the rest of the source code
COPY . .

# Install the project itself (editable / full)
RUN uv sync --frozen

# MLflow: default tracking URI (override via docker-compose environment)
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

ENTRYPOINT ["uv", "run", "accelerate", "launch", "-m", "src.runner"]

CMD ["--allrun"]
