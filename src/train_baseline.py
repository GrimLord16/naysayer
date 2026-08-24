"""Baseline label classifier: word-level TF-IDF + OneVsRest Logistic Regression.
Runs in parallel with train_champion.py in the Airflow DAG; whichever produces the
better micro-F1 gets promoted to `champion` (see evaluate.promote_if_better). This
same unchanged script also drives the Databricks-hosted MLflow/Unity Catalog registry
for Assignment 4 — see docker-compose.databricks.yml — by pointing evaluate.py's
MLFLOW_TRACKING_URI/MLFLOW_REGISTRY_URI env vars at Databricks instead of local
self-hosted MLflow; no code here changes between the two."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

import evaluate as common


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            # stop_words="english" strips noise words (the, a, is, ...) that carry
            # no label signal but previously ate vocabulary budget and diluted TF-IDF
            # weights on the words that actually matter.
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, stop_words="english")),
            ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))),
        ]
    )


if __name__ == "__main__":
    records = common.load_processed()
    result = common.train_and_evaluate(build_pipeline, records, common.HOLDOUT_FRACTION)
    common.save_model(result, "model_baseline.joblib")

    new_version = common.log_to_mlflow(result, common.HOLDOUT_FRACTION, model_variant="baseline")
    common.promote_if_better(new_version)
