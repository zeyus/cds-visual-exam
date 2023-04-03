"""CNN model for CIFAR10 dataset."""

import torch
import torch.nn as nn


class Cifar10CNN(nn.Module):
    """CNN model for CIFAR10 dataset."""

    def __init__(self):
        """Init."""
        super(Cifar10CNN, self).__init__()

        self.model = nn.Sequential(
            self.conv_relu_pool(3, 64),
            self.conv_relu_pool(64, 128),
            self.conv_relu_pool(128, 256),
            nn.Flatten(),
            nn.Dropout(0.1),
            self.linear_relu(256 * 4 * 4, 1024),
            nn.Dropout(0.1),
            self.linear_relu(1024, 512),
            nn.Dropout(0.1),
            self.linear_relu(512, 10),
        )

    def conv_relu_pool(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1) -> nn.Sequential:
        """Add conv, relu, pool layers."""

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def linear_relu(
            self,
            in_features: int,
            out_features: int) -> nn.Sequential:
        """Add linear, relu layers."""

        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

        return self.model(x)

# The same as above but implemented with scikit-learn instead of PyTorch.

