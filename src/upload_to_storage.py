"""Uploads local data/model artifacts to S3-compatible object storage.

Works unchanged against MinIO (local, free) or real AWS S3 — set S3_ENDPOINT_URL to
point at MinIO; leave it unset to hit real AWS S3 with boto3's default credential chain.
"""

import os
from pathlib import Path

import boto3

S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL") or None  # unset = real AWS S3
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
S3_BUCKET = os.environ.get("S3_BUCKET", "naysayer")

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FILES = [
    BASE_DIR / "data" / "raw_issues.json",
    BASE_DIR / "data" / "processed_issues.json",
    BASE_DIR / "models" / "model.joblib",
    BASE_DIR / "models" / "metrics.json",
]


def get_client():
    kwargs = {}
    if S3_ENDPOINT_URL:
        kwargs["endpoint_url"] = S3_ENDPOINT_URL
    if S3_ACCESS_KEY and S3_SECRET_KEY:
        kwargs["aws_access_key_id"] = S3_ACCESS_KEY
        kwargs["aws_secret_access_key"] = S3_SECRET_KEY
    return boto3.client("s3", **kwargs)


def ensure_bucket(client, bucket: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        client.create_bucket(Bucket=bucket)
        print(f"Created bucket {bucket}")


def upload_all(client=None, bucket: str = S3_BUCKET) -> None:
    """Uploads each local file to the same relative path under the bucket
    (data/raw_issues.json -> s3://bucket/data/raw_issues.json, etc.) — a 1:1 mirror
    of the local layout, simple to reason about."""
    client = client or get_client()
    ensure_bucket(client, bucket)

    for path in UPLOAD_FILES:
        if not path.exists():
            print(f"Skipping {path} (not found)")
            continue
        key = str(path.relative_to(BASE_DIR))
        client.upload_file(str(path), bucket, key)
        print(f"Uploaded {path} -> s3://{bucket}/{key}")


if __name__ == "__main__":
    upload_all()
