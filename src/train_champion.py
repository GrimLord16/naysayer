"""Candidate label classifier: word-level + character-level TF-IDF (via FeatureUnion)
+ OneVsRest Logistic Regression. Char n-grams catch code-like tokens, typos, and
sub-word patterns that word-level features miss — common in GitHub issue text — so
this is a real architectural improvement over train_baseline.py, not a parameter
tweak. Runs in parallel with train_baseline.py in the Airflow DAG; whichever produces
the better micro-F1 gets promoted to `champion` (see evaluate.promote_if_better) —
this script doesn't force its own promotion, it just aims to actually be better.
Also drives the Databricks-hosted MLflow/Unity Catalog registry for Assignment 4
unchanged — see train_baseline.py's docstring."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

import evaluate as common


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("word_tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, stop_words="english")),
                        ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=20_000)),
                    ]
                ),
            ),
            ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))),
        ]
    )


if __name__ == "__main__":
    records = common.load_processed()
    result = common.train_and_evaluate(build_pipeline, records, common.HOLDOUT_FRACTION)
    common.save_model(result, "model_champion.joblib")

    new_version = common.log_to_mlflow(result, common.HOLDOUT_FRACTION, model_variant="champion_candidate")
    common.promote_if_better(new_version)
