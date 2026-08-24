FROM apache/airflow:2.9.3-python3.11

COPY requirements.txt /opt/airflow/requirements.txt
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt \
    # mlflow needs cryptography>=43, but this base image ships pyOpenSSL 24.1.0
    # (needs cryptography<43) — an unresolvable pin conflict. pyOpenSSL is only an
    # optional TLS backend for urllib3/botocore (used transitively by boto3, for
    # src/upload_to_storage.py); when it's cleanly absent rather than present-but-
    # broken, they fall back to Python's stdlib ssl module without issue.
    && pip uninstall -y pyopenssl

COPY src /opt/airflow/src

ENV PYTHONPATH=/opt/airflow
