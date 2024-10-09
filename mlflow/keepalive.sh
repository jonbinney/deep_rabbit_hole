#!/bin/bash
# Silly script that queries MLFlow projects, to keep it busy and avoid Ploomber from shutting it down
# curl -X GET \
#   https://soft-pond-5082.ploomberapp.io/api/2.0/mlflow/experiments/search \
#   -d '{"max_results": "100"}' \
#   -H 'Authorization: Basic YWRtaW46VzNBcmVMYXp5IQ==' \
#   -H 'Content-Type: application/json'

# Setting up this CURL command in Google Cloud Scheduler
gcloud scheduler jobs create http mlflow_query_projects \
  --project="deep-learning-focus" \
  --location="us-central1" \
  --schedule="*/10 * * * *" \
  --time-zone="America/New_York" \
  --uri="https://soft-pond-5082.ploomberapp.io/api/2.0/mlflow/experiments/search?max_results=100" \
  --http-method="GET" \
  --headers="Authorization=Basic YWRtaW46VzNBcmVMYXp5IQ==,Content-Type=application/json" \