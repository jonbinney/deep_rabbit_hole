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

def create_model(**kwargs):
    # Create the model
    model = BasicCnnRegression(**kwargs)
    return (model, kwargs)

def do_training(
        # Dataset parameters
        train_dataset_dir: str,
        test_dataset_dir: str,
        annotations_file: str,
        # Training parameters
        n_epochs: int = 40,
        learning_rate: float = 1e-3,
        normalize_output: bool = False,
        crop_box: list = None,
        # Model parameters
        dropout_p: float = None,
        n_conv_layers: int = 2,
        channel_multiplier: float = 2.0,
        conv_kernel_size: int = 4,
        conv_stride: int = 2,
        conv_padding: int = 1,
        max_pool_kernel_size: int = 2,
        max_pool_stride: int = 1,
        # Configuration parameters
        log_transformed_images: bool = False,
        report_fn: callable = lambda *args, **kwargs: None
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
            normalize_output=normalize_output
        )
        test_data = get_data_loader(
            test_dataset_dir + '/images',
            test_dataset_dir + '/annotations/' + annotations_file,
            shuffle=False,
            crop_box=crop_box,
            normalize_output=normalize_output
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
    (model, model_args) = create_model(
        image_size=first_image.shape,
        dropout_p=dropout_p,
        n_conv_layers=n_conv_layers,
        channel_multiplier=channel_multiplier,
        conv_kernel_size=conv_kernel_size,
        conv_stride=conv_stride,
        conv_padding=conv_padding,
        max_pool_kernel_size=max_pool_kernel_size,
        max_pool_stride=max_pool_stride)
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
 
        train_loss = loss.item()

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

        # TODO: Abstract this into a reporting function
        print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {train_loss:.4f}, Test loss: {test_loss:.4f}')
        mlflow.log_metric('loss', train_loss, step=epoch)
        mlflow.log_metric('test_loss', test_loss, step=epoch)
        report_fn({
            'epoch': epoch,
            'loss': train_loss,
            'test_loss': test_loss
        })


    # Save model to disk, locally
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_args': model_args,
    }, 'model.pth')

    # TODO: Log Model. It's a bit trickier than this, it requires the signature to be inferred or defined properly
    #mlflow.pytorch.log_model(model, "model")

    return { 'loss': train_loss, 'test_loss': test_loss }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model on the Deep Water Level dataset')
    parser.add_argument('--train_dataset_dir', type=str, default='datasets/water_train_set4', help='Path to the train dataset directory')
    parser.add_argument('--test_dataset_dir', type=str, default='datasets/water_test_set5', help='Path to the test dataset directory')
    parser.add_argument('--annotations_file', type=str, default='filtered.csv', help='File name of the JSON file containing annotations within a dataset')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training the model')
    parser.add_argument('--crop_box', nargs=4, type=int, default=[130,275,140,140], help='Box with which to crop images, of form: top left height width')
    parser.add_argument('--dropout_p', type=float, default=0, help='Dropout probability to apply, from 0 to 1. None or 0.0 means disabled.')
    parser.add_argument('--log_transformed_images', type=bool, default=False, help='Log transformed images using mlflow')
    parser.add_argument('--normalize_output', type=bool, default=False, help='Normalize depth value to [-1, 1] range')
    parser.add_argument('--n_conv_layers', type=int, default=3, help='Number of convolutional layers in the model (2 or 3)')
    parser.add_argument('--channel_multiplier', type=float, default=2.0, help='Multiplier for the number of channels in each convolutional layer')
    parser.add_argument('--conv_kernel_size', type=int, default=7, help='Convolutional kernel size, for all layers')
    args = parser.parse_args()

    start_experiment("Deep Water Level Training")

    with mlflow.start_run():
        mlflow.log_params(vars(args))
        mlflow.log_param("hardware/gpu", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
        model = do_training(**vars(args))