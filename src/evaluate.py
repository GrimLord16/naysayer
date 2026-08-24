"""Shared data loading, evaluation, MLflow logging, and champion/challenger
promotion logic — used by train_baseline.py and train_champion.py, which differ
only in which model pipeline they build (see each file's build_pipeline())."""

import json
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from ingestion import REPO_NAME, REPO_OWNER

# Overridable so the same code runs unchanged locally (Docker bind mounts) and on
# Databricks (Unity Catalog Volumes, e.g. NAYSAYER_DATA_DIR=/Volumes/workspace/naysayer/data)
DATA_DIR = Path(os.environ.get("NAYSAYER_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
MODELS_DIR = Path(os.environ.get("NAYSAYER_MODELS_DIR", str(Path(__file__).resolve().parent.parent / "models")))
HOLDOUT_FRACTION = float(os.environ.get("HOLDOUT_FRACTION", "0.15"))

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_REGISTRY_URI = os.environ.get("MLFLOW_REGISTRY_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "issue-label-classifier")
REGISTERED_MODEL_NAME = os.environ.get("NAYSAYER_REGISTERED_MODEL_NAME", "issue-label-classifier")

PROMOTION_METRIC = "micro_f1"
PROMOTION_MARGIN = float(os.environ.get("PROMOTION_MARGIN", "0.005"))


def load_processed(filename: str = "processed_issues.json") -> list[dict]:
    return json.loads((DATA_DIR / filename).read_text())


def time_split(records: list[dict], holdout_fraction: float) -> tuple[list[dict], list[dict]]:
    """Train on older issues, validate on newer ones — mimics production and exposes
    drift. A fraction of the sorted records, not a fixed day count: a fixed window can
    silently shrink to near-nothing if the data doesn't span as much real time as
    assumed (e.g. a capped, recent-issues-only pull)."""
    ordered = sorted(records, key=lambda r: r["created_at"])
    split_idx = int(len(ordered) * (1 - holdout_fraction))
    return ordered[:split_idx], ordered[split_idx:]


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 3) -> float:
    k = min(k, y_score.shape[1])
    top_k = np.argsort(-y_score, axis=1)[:, :k]
    hits = np.take_along_axis(y_true, top_k, axis=1)
    return float(hits.sum(axis=1).mean() / k)


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 3) -> float:
    k = min(k, y_score.shape[1])
    top_k = np.argsort(-y_score, axis=1)[:, :k]
    hits = np.take_along_axis(y_true, top_k, axis=1).sum(axis=1)
    totals = np.maximum(y_true.sum(axis=1), 1)
    return float((hits / totals).mean())


def train_and_evaluate(build_pipeline, records: list[dict], holdout_fraction: float) -> dict:
    """`build_pipeline()` returns an unfit sklearn Pipeline — the one thing that
    differs between train_baseline.py and train_champion.py. Everything else
    (splitting, fitting, evaluation) is identical so their metrics are comparable."""
    train_records, holdout_records = time_split(records, holdout_fraction)

    label_set = sorted({label for r in records for label in r["labels"]})
    mlb = MultiLabelBinarizer(classes=label_set)

    pipeline = build_pipeline()

    y_train = mlb.fit_transform([r["labels"] for r in train_records])
    train_texts = [r["text"] for r in train_records]
    pipeline.fit(train_texts, y_train)

    y_holdout = mlb.transform([r["labels"] for r in holdout_records])
    if holdout_records:
        y_score = pipeline.predict_proba([r["text"] for r in holdout_records])
    else:
        y_score = np.zeros((0, len(label_set)))

    y_pred = (y_score >= 0.5).astype(int)
    metrics = {
        "micro_f1": float(f1_score(y_holdout, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_holdout, y_pred, average="macro", zero_division=0)),
        "precision_at_3": precision_at_k(y_holdout, y_score, k=3),
        "recall_at_3": recall_at_k(y_holdout, y_score, k=3),
        "train_size": len(train_records),
        "holdout_size": len(holdout_records),
    }
    # A few raw training inputs (+ their real predictions), kept for log_to_mlflow's
    # signature/input_example — Unity Catalog's model registry requires a signature
    # (input/output schema) on every registered version; self-hosted MLflow doesn't
    # need it but isn't hurt by it. Must be a numpy object array, not a bare Python
    # list — mlflow's signature auto-inference has a bug on plain list[str] input
    # (AttributeError: 'int' object has no attribute 'lower'), confirmed while
    # testing against Databricks; an explicit array + infer_signature() sidesteps it.
    sample_input = np.array(train_texts[:5], dtype=object)
    sample_output = pipeline.predict_proba(list(sample_input))
    return {
        "pipeline": pipeline,
        "label_classes": label_set,
        "metrics": metrics,
        "sample_input": sample_input,
        "sample_output": sample_output,
    }


def save_model(result: dict, filename: str) -> Path:
    """Local copy for quick inspection/debugging — the API no longer reads from
    here, it loads champion/challenger straight from the MLflow registry."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": result["pipeline"], "label_classes": result["label_classes"]}, MODELS_DIR / filename)
    metrics_filename = filename.replace(".joblib", "_metrics.json")
    (MODELS_DIR / metrics_filename).write_text(json.dumps(result["metrics"], indent=2))
    print(f"Saved model to {MODELS_DIR / filename}")
    print(f"Metrics: {result['metrics']}")
    return MODELS_DIR / filename


def log_to_mlflow(result: dict, holdout_fraction: float, model_variant: str) -> str:
    """Logs params/metrics/model to MLflow and registers a new model version,
    tagged `challenger` (README §5.4: "every training run is logged to MLflow ...
    and registered as the challenger"). `model_variant` (e.g. "baseline",
    "champion_candidate") is logged as a param so runs from each script are
    distinguishable in the MLflow UI. Returns the new version number."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if MLFLOW_REGISTRY_URI:
        mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "repo": f"{REPO_OWNER}/{REPO_NAME}",
                "holdout_fraction": holdout_fraction,
                "num_labels": len(result["label_classes"]),
                "model_variant": model_variant,
            }
        )
        mlflow.log_metrics(result["metrics"])
        mlflow.log_dict({"label_classes": result["label_classes"]}, "label_classes.json")
        signature = infer_signature(result["sample_input"], result["sample_output"])
        model_info = mlflow.sklearn.log_model(
            result["pipeline"],
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=signature,
            input_example=result["sample_input"],
        )

    version = model_info.registered_model_version
    MlflowClient().set_registered_model_alias(REGISTERED_MODEL_NAME, "challenger", version)
    print(f"Logged to MLflow: run {model_info.run_id}, registered as challenger v{version}")
    return version


def promote_if_better(new_version: str, metric: str = PROMOTION_METRIC, margin: float = PROMOTION_MARGIN) -> None:
    """Compares the new challenger's holdout metric against the current champion's
    and promotes it if it wins by `margin`, or if there's no champion yet. Runs
    identically regardless of which script produced the new version — whichever
    model is actually better wins, the promotion logic doesn't favor either script."""
    client = MlflowClient()
    new_run_id = client.get_model_version(REGISTERED_MODEL_NAME, new_version).run_id
    new_score = client.get_run(new_run_id).data.metrics[metric]

    try:
        champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
        champion_score = client.get_run(champion.run_id).data.metrics.get(metric, float("-inf"))
    except MlflowException:
        champion, champion_score = None, float("-inf")

    if new_score >= champion_score + margin or champion is None:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", new_version)
        prev = f"v{champion.version} ({champion_score:.4f})" if champion else "none yet"
        print(f"Promoted v{new_version} ({metric}={new_score:.4f}) to champion, beating {prev}")
    else:
        print(
            f"v{new_version} ({metric}={new_score:.4f}) stays challenger — "
            f"didn't beat champion v{champion.version} ({champion_score:.4f}) by margin {margin}"
        )
