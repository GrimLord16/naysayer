"""Daily pipeline: fetch GitHub issues, preprocess, train two model variants in
parallel (train_baseline.py, train_champion.py), upload to storage. The two training
tasks are graph-parallel (neither depends on the other) — under Airflow's
SequentialExecutor (see docker-compose.yml) they still run one at a time in
practice, so there's no race on the champion/challenger registry writes."""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {"owner": "naysayer", "retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="issue_pipeline",
    description="Fetch issues -> preprocess -> [train_baseline, train_champion] -> upload to storage",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
) as dag:
    ingest = BashOperator(task_id="ingestion", bash_command="python /opt/airflow/src/ingestion.py")
    preprocess = BashOperator(task_id="preprocessing", bash_command="python /opt/airflow/src/preprocessing.py")
    train_baseline = BashOperator(
        task_id="train_baseline", bash_command="python /opt/airflow/src/train_baseline.py"
    )
    train_champion = BashOperator(
        task_id="train_champion", bash_command="python /opt/airflow/src/train_champion.py"
    )
    upload = BashOperator(task_id="upload_to_storage", bash_command="python /opt/airflow/src/upload_to_storage.py")

    ingest >> preprocess >> [train_baseline, train_champion] >> upload
