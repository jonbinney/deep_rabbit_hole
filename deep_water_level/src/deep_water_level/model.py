from typing import Literal, Tuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

ModelNames = Literal["BasicCnnRegression", "BasicCnnRegressionWaterLine", "ResNet50Pretrained", "DeepLabV3"]


class BasicCnnRegression(nn.Module):
    """
    Basic model which includes two CNN layers and a single linear layer.
    """

    def __init__(
        self,
        image_size: Tuple[int, int, int] = (3, 810, 510),
        dropout_p: float = None,
        n_conv_layers: int = 2,  # Number of convolutional layers
        channel_multiplier: float = 2.0,  # On each conv layer, number of channels is increased by this factor
        conv_kernel_size: int = 4,
        conv_stride: int = 2,
        conv_padding: int = 1,
        max_pool_kernel_size: int = 2,
        max_pool_stride: int = 1,
    ):
        super().__init__()

        if n_conv_layers not in [2, 3, 4, 5]:
            raise ValueError("conv_layers must an integer between 2 and 5")

        self.n_conv_layers = n_conv_layers
        self.image_size = image_size
        self.dropout_p = dropout_p
        self.channel_multiplier = channel_multiplier
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.max_pool_kernel_size = max_pool_kernel_size
        self.max_pool_stride = max_pool_stride

        self.conv_list = nn.ModuleList()
        next_channels = image_size[0]
        for _ in range(self.n_conv_layers):
            (conv, pool, next_channels) = self.make_cnn_layer(next_channels)
            self.conv_list.append(conv)
            self.conv_list.append(pool)

        self.flatten = nn.Flatten()

        # Calculate linear size assuming fixed image size
        self.linear_size = self.calculate_linear_size(image_size)

        self.fcn1 = nn.Linear(in_features=self.linear_size, out_features=120)
        if dropout_p is not None and dropout_p > 0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.fcn2 = nn.Linear(in_features=120, out_features=1)

    def make_cnn_layer(self, in_channels):
        out_channels = int(in_channels * self.channel_multiplier)
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.conv_kernel_size,
            stride=self.conv_stride,
            padding=self.conv_padding,
        )
        pool = nn.MaxPool2d(kernel_size=self.max_pool_kernel_size, stride=self.max_pool_stride)
        return conv, pool, out_channels

    def calculate_linear_size(self, image_size):
        dummy_input = torch.randn(1, *image_size)
        output = self.aux_conv_forward(dummy_input)
        return output.shape[1:][0]

    def aux_conv_forward(self, x):
        for i in range(self.n_conv_layers):
            x = self.conv_list[i * 2](x)
            x = nn.functional.relu(x)
            x = self.conv_list[i * 2 + 1](x)
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


class BasicCnnRegressionWaterLine(nn.Module):
    """
    Basic model which includes two CNN layers and a single linear layer, and the output is the coordinates for the water line.
    """

    def __init__(
        self,
        image_size: Tuple[int, int, int] = (3, 810, 510),
        dropout_p: float = None,
        n_conv_layers: int = 2,  # Number of convolutional layers
        channel_multiplier: float = 2.0,  # On each conv layer, number of channels is increased by this factor
        conv_kernel_size: int = 4,
        conv_stride: int = 2,
        conv_padding: int = 1,
        max_pool_kernel_size: int = 2,
        max_pool_stride: int = 1,
    ):
        super().__init__()

        if n_conv_layers not in [2, 3, 4, 5]:
            raise ValueError("conv_layers must an integer between 2 and 5")

        self.n_conv_layers = n_conv_layers
        self.image_size = image_size
        self.dropout_p = dropout_p
        self.channel_multiplier = channel_multiplier
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.max_pool_kernel_size = max_pool_kernel_size
        self.max_pool_stride = max_pool_stride

        self.conv_list = nn.ModuleList()
        next_channels = image_size[0]
        for _ in range(self.n_conv_layers):
            (conv, pool, next_channels) = self.make_cnn_layer(next_channels)
            self.conv_list.append(conv)
            self.conv_list.append(pool)

        self.flatten = nn.Flatten()

        # Calculate linear size assuming fixed image size
        self.linear_size = self.calculate_linear_size(image_size)

        self.fcn1 = nn.Linear(in_features=self.linear_size, out_features=120)
        if dropout_p is not None and dropout_p > 0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.fcn2 = nn.Linear(in_features=120, out_features=5)

    def make_cnn_layer(self, in_channels):
        out_channels = int(in_channels * self.channel_multiplier)
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.conv_kernel_size,
            stride=self.conv_stride,
            padding=self.conv_padding,
        )
        pool = nn.MaxPool2d(kernel_size=self.max_pool_kernel_size, stride=self.max_pool_stride)
        return conv, pool, out_channels

    def calculate_linear_size(self, image_size):
        dummy_input = torch.randn(1, *image_size)
        output = self.aux_conv_forward(dummy_input)
        return output.shape[1:][0]

    def aux_conv_forward(self, x):
        for i in range(self.n_conv_layers):
            x = self.conv_list[i * 2](x)
            x = nn.functional.relu(x)
            x = self.conv_list[i * 2 + 1](x)
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


class DeepLabModel(nn.Module):
    def __init__(self):
        super(DeepLabModel, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        self.model.classifier = DeepLabHead(2048, num_classes=2)

    def forward(self, x):
        return self.model(x)["out"]
