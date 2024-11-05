# Script that trains a network model for deep water level detection
import argparse
import mlflow
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torchvision
from model import BasicCnnRegression
from data import get_data_loader, get_data_loaders
from utils import start_experiment

def create_model(image_size):
    # Create the model
    model = BasicCnnRegression(image_size)
    return model

def load_data():
    # TODO(adamantivm) Normalization? Resizing?
    data = None
    return data

def do_training(
        train_dataset_dir: str,
        test_dataset_dir: str,
        annotations_file: str,
        n_epochs: int,
        learning_rate: float,
        crop_box: list,
        log_transformed_images,
        ):
    # Set-up environment
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.set_default_device(device)
    print(f"Using device: {device}")

    # Load the data
    if train_dataset_dir is None or test_dataset_dir == train_dataset_dir:
        # Split the train dataset in test and train datasets
        (train_data, test_data) = get_data_loaders(
            train_dataset_dir + '/images',
            train_dataset_dir + '/annotations/' + annotations_file,
            crop_box=crop_box,
        )

    else:
        train_data = get_data_loader(
            train_dataset_dir + '/images',
            train_dataset_dir + '/annotations/' + annotations_file,
            crop_box=crop_box,
        )
        test_data = get_data_loader(
            test_dataset_dir + '/images',
            test_dataset_dir + '/annotations/' + annotations_file,
            shuffle=False,
            crop_box=crop_box,
        )

    if log_transformed_images:
        log_dir = Path('/tmp/deep_water_level')
        log_dir.mkdir(parents=True, exist_ok=True)
        for image_i in range(len(train_data.dataset)):
            image_filename = log_dir / f"train_data_{image_i}.png"
            image = train_data.dataset[image_i][0]
            torchvision.utils.save_image(image, image_filename)
        for image_i in range(len(test_data.dataset)):
            image_filename = log_dir / f"test_data_{image_i}.png"
            image = test_data.dataset[image_i][0]
            torchvision.utils.save_image(image, image_filename)

    # Grab the first training image to find the resolution, which determines the size of the model
    first_image = train_data.dataset[0][0]

    # Train the model
    model = create_model(first_image.shape)
    model.to(device)
    print(f"Model summary: {model}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        # Train for the epoch
        model.train()
        for i, (inputs, labels, filenames) in enumerate(train_data):
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}', end='')
        mlflow.log_metric('loss', loss.item(), step=epoch)

        # Test for this epoch
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (inputs, labels, filenames) in enumerate(test_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

        test_loss /= len(test_data)

        print(f', Test loss: {test_loss:.4f}')
        mlflow.log_metric('test_loss', test_loss, step=epoch)

    # Save model to disk, locally
    torch.save(model.state_dict(), 'model.pth')

    # TODO: Log Model. It's a bit trickier than this, it requires the signature to be inferred or defined properly
    #mlflow.pytorch.log_model(model, "model")

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model on the Deep Water Level dataset')
    parser.add_argument('--train_dataset_dir', type=str, default='datasets/water_2024_10_19_set1', help='Path to the train dataset directory')
    parser.add_argument('--test_dataset_dir', type=str, default='datasets/water_test_set3', help='Path to the test dataset directory')
    parser.add_argument('--annotations_file', type=str, default='manual_annotations.json', help='File name of the JSON file containing annotations within a dataset')
    parser.add_argument('--n_epochs', type=int, default=40, help='Number of epochs to train the model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training the model')
    parser.add_argument('--crop_box', nargs=4, type=int, default=None, help='Box with which to crop images, of form: top left height width')
    parser.add_argument('--log_transformed_images', type=bool, default=False, help='Log transformed images using mlflow')
    args = parser.parse_args()

    start_experiment("Deep Water Level Training")

    with mlflow.start_run():
        mlflow.log_params(vars(args))
        mlflow.log_param("hardware/gpu", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
        model = do_training(**vars(args))