# Script that trains a network model for deep water level detection
# TODO:
# - Create the network model using CNN
# - Load the data
# - Train the model
# - Set-up
import argparse
import torch
import torch.nn as nn
from model import BasicCnnRegression
from data import get_data_loader

def create_model():
    # Create the model
    model = BasicCnnRegression()
    return model

def load_data():
    # TODO(adamantivm) Normalization? Resizing?
    data = None
    return data

def do_training(annotations_file: str, images_dir: str):
    # Set-up environment
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.set_default_device(device)
    print(f"Using device: {device}")

    # TODO(adamantivm) Load the data
    data_loader = get_data_loader(annotations_file, images_dir)

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

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {loss.item():.4f}')

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model on the Deep Water Level dataset')
    parser.add_argument('--annotations_file', type=str, default='datasets/water_2024_10_19_set1/annotations/manual_annotations.json', help='Path to a JSON file containing annotations')
    parser.add_argument('--images_dir', type=str, default='datasets/water_2024_10_19_set1/images', help='Path to a directory containing images')
    args = parser.parse_args()

    model = do_training(args.annotations_file, args.images_dir)