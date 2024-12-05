import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import utils
import torch.nn as nn
import cv2 as cv
from torchvision.transforms import v2
from torchvision import transforms
from infer import load_model
from model import BasicCnnRegression
import argparse


def get_conv_layers(model):
    model_children = list(model.children())
    counter = 0
    conv_layers = []
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for x in model_children[i]:
                for child in x.children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        conv_layers.append(child)

    print(f"Found {counter} convolutional layers:")

    for layer in conv_layers:
        print(f" - {layer} => {layer.weight.shape}")

    return conv_layers


def visualize_tensor(tensor, title, ch=0, allkernels=False, nrow=8, padding=1):
    n, c, w, h = tensor.shape

    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(title, figsize=(nrow, rows))
    plt.axis("off")
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


def visualize_feature_map(conv_layers, filename):
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()
    transform = transforms.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),  # convert to float32 and normalize to 0, 1
        ]
    )

    img = np.array(img)
    img = transform(img)
    # unsqueeze to add a batch dimension
    img = img.unsqueeze(0)

    results = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        results.append(conv_layers[i](results[-1]))
    # make a copy of the `results`
    outputs = results

    for num_layer in range(len(outputs)):
        plt.figure(f"Conv layer {num_layer}", figsize=(15, 10))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        for i, filter in enumerate(layer_viz):
            if i == 64:  # we will visualize up to 8x8 blocks from each layer
                break
            plt.subplot(4, 4, i + 1)
            plt.imshow(filter, cmap="gray")
            plt.axis("off")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Water Level")
    parser.add_argument("-m", "--model_path", type=str, default="model.pth", help="Path to the model file")
    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        default="datasets/water_2024_10_19_set1/images/pilenew/images/2024-10-09/0-1728486001.jpg",
        help="Path to a sampe image file",
    )
    args = parser.parse_args()

    model, _ = load_model(args.model_path)
    conv_layers = get_conv_layers(model)

    for layer in conv_layers:
        visualize_tensor(layer.weight, f"Kernels for {layer}", allkernels=True)

    visualize_feature_map(conv_layers, args.file_path)

    plt.show()
