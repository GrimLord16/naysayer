FROM python:3.11-slim

# boto3 is needed for the server itself to talk to MinIO (S3 API) when proxying
# artifact reads/writes (`--serve-artifacts`) — not for the training client, which
# only ever talks to this server over plain HTTP.
RUN pip install --no-cache-dir mlflow boto3

EXPOSE 5000
