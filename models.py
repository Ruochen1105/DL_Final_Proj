from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.s_dim = 256

    def forward(self, states, actions):
        """
        Args:
            states: [B, T, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        return torch.randn((self.bs, self.n_steps, self.s_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output


class Encoder(nn.Module):
    def __init__(self, input_shape, s_dim, cnn_dim):
        super().__init__()

        # Calculate linear layer dimentions
        C, H, W = input_shape
        H, W = (H - 1) // 2 + 1, (W - 1) // 2 + 1
        H, W = (H - 1) // 2 + 1, (W - 1) // 2 + 1
        fc_input_dim = H * W * cnn_dim * 2

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=cnn_dim, kernel_size=3, stride=2,
                      padding=1),
            nn.BatchNorm2d(num_features=cnn_dim)
            nn.ReLU(),
            nn.Conv2d(in_channels=cnn_dim, out_channels=cnn_dim * 2, kernel_size=5, stride=2,
                      padding=2),
            nn.BatchNorm2d(num_features=cnn_dim * 2)
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(fc_input_dim, s_dim)

    def forward(self, x):
        """
        Forward pass for the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, s_dim).
        """
        B, T, C, H, W = x.size()

        x = x.view(B * T, C, H, W)

        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)

        x = x.view(B, T, -1)

        return x


class Predictor(nn.Module):
    def __init__(self, s_dim, u_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(s_dim + u_dim, s_dim),
            nn.BatchNorm2d(num_features=s_dim)
            nn.ReLU(),
        )

    def forward(self, state, action):
        """
        Forward pass for the predictor.

        Args:
            state (torch.Tensor): State representation of shape (B, s_dim).
            action (torch.Tensor): Action vector of shape (B, u_dim).

        Returns:
            torch.Tensor: Predicted next state of shape (B, s_dim).
        """
        x = torch.cat([state, action], dim=1)

        x = self.fc(x)

        return x
