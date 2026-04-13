# deployable-classifier [![python](https://img.shields.io/badge/Python-3.12.1-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

A modular, PyTorch-based CNN classifier for CIFAR-10, packaged into a production-style ML stack with experiment tracking, artifact storage, a RESTful API, and observability.

## Stack

| Service | Purpose |
|---|---|
| PostgreSQL | MLflow metadata backend |
| MinIO | S3-compatible artifact store (model weights) |
| MLflow | Experiment tracking and model registry |
| FastAPI | Model serving API (`/predict`, `/predict_batched`) |
| Prometheus | Metrics scraping from API `/metrics` endpoint |
| Grafana | Dashboard for request rate, latency, error rate |

## Quickstart

1. Install dependencies:
```
uv sync
```

2. Start the infrastructure (PostgreSQL, MinIO, MLflow):
```
docker compose up database minio minio-buckets mlflow
```

3. Train the model:
```
docker compose run ml-pipeline --allrun
```
Or run locally:
```
uv run accelerate launch -m src.runner --allrun
```

After training, note the MLflow run ID from the UI at `http://localhost:5000` and set it in `docker-compose.yml` under `MLFLOW_RUN_ID`.

4. Start the full stack:
```
docker compose up
```

Services will be available at:
- MLflow UI: `http://localhost:5000`
- MinIO console: `http://localhost:9001`
- API: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## Running inference

```
python -m src.api_infer_serial --config=src/configs/config.py
python -m src.api_infer_batched --config=src/configs/config.py
```

## Training flags

```
uv run accelerate launch -m src.runner --train          # train only
uv run accelerate launch -m src.runner --eval --run_id=<id>  # eval only
uv run accelerate launch -m src.runner --allrun         # train + eval
```

## TODO

- [ ] Fix `processing_task` scoping bug in `api.py` startup handler
- [ ] Replace deprecated `@app.on_event("startup")` with lifespan context manager
- [ ] Add MLflow Model Registry integration
- [ ] Add pytest test suite
- [ ] Add Docker healthchecks for proper service dependency ordering
