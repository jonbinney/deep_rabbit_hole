from typing import Tuple
import torch.nn as nn

class BasicCnnRegression(nn.Module):
    """
    Basic model which includes two CNN layers and a single linear layer.
    """
    def __init__(self, image_size: Tuple[int, int, int] = (3, 810, 510)):
        super().__init__()

        self.image_size = image_size
        self.linear_line_size = (image_size[1] // 4 ) * (image_size[2] // 4) * 12

        self.conv1 = nn.Conv2d(in_channels=image_size[0], out_channels=6, kernel_size=4, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=4, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fcn1 = nn.Linear(in_features=self.linear_line_size, out_features=120)
        self.fcn2 = nn.Linear(in_features=120, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fcn1(x)
        x = nn.functional.relu(x)
        x = self.fcn2(x)
        return x

