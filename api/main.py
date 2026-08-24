"""FastAPI wrapper serving predictions from the MLflow-registered champion and
challenger models (see src/evaluate.py for how they get promoted)."""

import os
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from api.schemas import HealthResponse, LabelScore, ModelStatus, PredictRequest, PredictResponse
from src.preprocessing import clean_text  # same cleaning used at train time — no train/serve skew

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
# Unset for local self-hosted MLflow; "databricks-uc" when pointed at Databricks
# (see docker-compose.databricks.yml) — same toggle as src/evaluate.py.
MLFLOW_REGISTRY_URI = os.environ.get("MLFLOW_REGISTRY_URI")
REGISTERED_MODEL_NAME = os.environ.get("NAYSAYER_REGISTERED_MODEL_NAME", "issue-label-classifier")
TOP_K = 3
ALIASES = ("champion", "challenger")

_state: dict = dict.fromkeys(ALIASES)


def _load_alias(alias: str) -> dict | None:
    """Loads a model + its label vocabulary from the MLflow registry by alias.
    label_classes isn't part of the sklearn model itself — it's logged alongside it
    as a separate run artifact at train time (see evaluate.log_to_mlflow). Uses
    version.source (not the legacy models:/name/version URI) — MLflow 3.x's
    "Logged Models" store log_model() artifacts under a model-id path that the
    legacy URI form doesn't resolve to."""
    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
    except MlflowException:
        return None

    pipeline = mlflow.sklearn.load_model(version.source)
    label_classes = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/label_classes.json")["label_classes"]
    return {"pipeline": pipeline, "label_classes": label_classes, "version": version.version}


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if MLFLOW_REGISTRY_URI:
        mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
    for alias in ALIASES:
        _state[alias] = _load_alias(alias)
    yield


app = FastAPI(title="Naysayer Inference API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    def status(alias: str) -> ModelStatus:
        entry = _state[alias]
        return ModelStatus(loaded=entry is not None, version=entry["version"] if entry else None)

    return HealthResponse(status="ok", champion=status("champion"), challenger=status("challenger"))


def _predict(alias: str, request: PredictRequest) -> PredictResponse:
    entry = _state.get(alias)
    if entry is None:
        raise HTTPException(status_code=503, detail=f"No {alias} model loaded yet")

    text = clean_text(f"{request.title} {request.body}")
    scores = entry["pipeline"].predict_proba([text])[0]
    top = sorted(zip(entry["label_classes"], scores), key=lambda pair: -pair[1])[:TOP_K]
    return PredictResponse(
        alias=alias,
        model_version=entry["version"],
        labels=[LabelScore(label=label, score=float(score)) for label, score in top],
    )


@app.post("/predict/champion", response_model=PredictResponse)
def predict_champion(request: PredictRequest) -> PredictResponse:
    return _predict("champion", request)


@app.post("/predict/challenger", response_model=PredictResponse)
def predict_challenger(request: PredictRequest) -> PredictResponse:
    return _predict("challenger", request)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Alias for /predict/champion — kept so existing calls to /predict still work."""
    return _predict("champion", request)
