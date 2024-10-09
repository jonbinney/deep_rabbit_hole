# Dockerfile for MLFlow with support for Google Cloud Storage
# Slight modification from https://github.com/ploomber/doc/tree/main/examples/docker/mlflow
FROM python:3.11-slim

RUN pip install mlflow --no-cache-dir
RUN pip install google-cloud-storage



ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "80", "--app-name", "basic-auth"]
