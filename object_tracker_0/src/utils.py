# This file contains utility functions that are used in the training and testing process.

import torch
import time
import mlflow
import os

def preprocess_data(data):
    # Preprocess the data here
    pass

def calculate_accuracy(predictions, labels):
    # Calculate the accuracy of the model predictions
    pass

def save_model(model, filepath):
    # Save the trained model to a file
    torch.save(model.state_dict(), filepath)

def load_model(filepath):
    # Load a trained model from a file
    model = Model()
    model.load_state_dict(torch.load(filepath))
    return model

def my_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

class Timer:
    def __init__(self):
        self.total_time = 0
        self.num_executions = 0

    def start(self):
        self.start_time = time.time()

    def stop(self, num_executions=1):
        end_time = time.time()
        self.total_time += (end_time - self.start_time)
        self.num_executions += num_executions

    def get_average_time(self):
        return self.total_time / self.num_executions if self.num_executions > 0 else 0

    def get_total_time(self):
        return self.total_time

    def get_num_executions(self):
        return self.num_executions

    def __str__(self) -> str:
        return f"Total: {self.total_time:.2f}s ({self.num_executions} x {self.get_average_time():.2f}s)"

def start_experiment(experiment_name: str):
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")
    print(f"Using MLFlow tracking URI: {mlflow_tracking_uri}")
    mlflow.set_tracking_uri(uri=mlflow_tracking_uri)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location="gs://deep-rabbit-hole/mlflow")
    mlflow.set_experiment(experiment_name)
    mlflow.enable_system_metrics_logging()