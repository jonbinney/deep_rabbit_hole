from typing import Tuple
import torch.nn as nn
import torch

class BasicCnnRegression(nn.Module):
    """
    Basic model which includes two CNN layers and a single linear layer.
    """
    def __init__(
            self,
            image_size: Tuple[int, int, int] = (3, 810, 510),
            dropout_p: float = None,
        ):
        super().__init__()

        self.image_size = image_size
        self.dropout_p = dropout_p

        self.conv1 = nn.Conv2d(in_channels=image_size[0], out_channels=6, kernel_size=4, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=4, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.flatten = nn.Flatten()

        # Calculate linear size assuming fixed image size
        self.linear_size = self.calculate_linear_size(image_size)

        self.fcn1 = nn.Linear(in_features=self.linear_size, out_features=120)
        if dropout_p is not None and dropout_p > 0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.fcn2 = nn.Linear(in_features=120, out_features=1)
    
    def calculate_linear_size(self, image_size):
        dummy_input = torch.randn(1, *image_size)
        output = self.aux_conv_forward(dummy_input)
        return output.shape[1:][0]

    
    def aux_conv_forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        return x

    def forward(self, x):
        x = self.aux_conv_forward(x)
        x = self.fcn1(x)
        x = nn.functional.relu(x)
        if self.dropout_p is not None and self.dropout_p > 0:
            x = self.dropout(x)
        x = self.fcn2(x)
        return x

