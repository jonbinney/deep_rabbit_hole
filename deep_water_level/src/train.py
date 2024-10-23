# Script that trains a network model for deep water level detection
# TODO:
# - Create the network model using CNN
# - Load the data
# - Train the model
# - Set-up
import argparse
import mlflow
import torch
import torch.nn as nn
from model import BasicCnnRegression
from data import get_data_loader
from utils import start_experiment

def create_model():
    # Create the model
    model = BasicCnnRegression()
    return model

def load_data():
    # TODO(adamantivm) Normalization? Resizing?
    data = None
    return data

def do_training(
        dataset_dir: str,
        annotations_file: str,
        ):
    # Set-up environment
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.set_default_device(device)
    print(f"Using device: {device}")

    # TODO(adamantivm) Load the data
    data_loader = get_data_loader(dataset_dir + '/annotations/' + annotations_file, dataset_dir + '/images')

    # Train the model
    model = create_model()
    model.to(device)
    print(f"Model summary: {model}")

    learning_rate = 0.001
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    n_epochs = 10

    for epoch in range(n_epochs):
        for i, (inputs, labels) in enumerate(data_loader):
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {loss.item():.4f}')

        mlflow.log_metric('loss', loss.item(), step=epoch)

    # TODO: Log Model. It's a bit trickier than this, it requires the signature to be inferred or defined properly
    #mlflow.pytorch.log_model(model, "model")

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model on the Deep Water Level dataset')
    parser.add_argument('--dataset_dir', type=str, default='datasets/water_2024_10_19_set1', help='Path to the dataset directory')
    parser.add_argument('--annotations_file', type=str, default='manual_annotations.json', help='File name of the JSON file containing annotations')
    args = parser.parse_args()

    start_experiment("Deep Water Level Training")

    with mlflow.start_run():
        mlflow.log_params(vars(args))
        mlflow.log_param("hardware/gpu", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
        model = do_training(**vars(args))