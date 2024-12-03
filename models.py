from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dataset import create_wall_dataloader


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
            states: [B, 1, Ch, H, W]
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


class JEPA(nn.Module):
    def __init__(self, s_dim, u_dim, cnn_dim):
        """
        Joint Embedding Predictive Architecture (JEPA).

        Args:
            s_dim (int): Space dimensionality for the encoded states.
            u_dim (int): Dimensionality of the action vector.
            cnn_dim (int): Base dimensionality for the CNN.
        """
        super().__init__()
        self.encoder = Encoder(s_dim, cnn_dim)
        self.predictor = Predictor(s_dim, u_dim)

    def forward(self, states, actions):
        """
        Forward pass for the JEPA model.

        Args:
            states (torch.Tensor): Sequence of states of shape (B, T, C, H, W).
            actions (torch.Tensor): Sequence of actions of shape (B, T-1, u_dim).

        Returns:
            torch.Tensor: Predicted next states of shape (B, T-1, s_dim).
        """
        # Encode the states into representations
        states = self.encoder(states)  # Shape: (B, T, s_dim)

        # Use states and actions to predict the next states
        B, T, _ = actions.size()
        predicted_states = []
        for t in range(T - 1):
            # Use state at time t and action at time t to predict state at t+1
            predicted_state = self.predictor(
                states[:, t], actions[:, t])
            predicted_states.append(predicted_state)

        # Concatenate all predicted states along the temporal dimension
        predicted_states = torch.stack(
            predicted_states, dim=1)  # Shape: (B, T-1, s_dim)

        return predicted_states


class Encoder(nn.Module):
    """
    Encoder module for JEPA (Joint Embedding Predictive Architecture).

    The Encoder transforms a sequence of input states (e.g., images or feature maps)
    into a representation in a reduced-dimensional space. It uses a
    Convolutional Neural Network (CNN) followed by a fully connected layer.

    Args:
        input_shape (tuple): Shape of a single input state (C, H, W), where:
            - C: Number of channels in the input.
            - H: Height of the input image.
            - W: Width of the input image.
        s_dim (int): Dimensionality of the space for the encoded state.
        cnn_dim (int): Base number of channels used in the CNN.

    Attributes:
        cnn (torch.nn.Sequential): Convolutional layers for feature extraction.
        flatten (torch.nn.Flatten): Layer to flatten the CNN output before the fully connected layer.
        fc (torch.nn.Linear): Fully connected layer mapping the flattened CNN output to the space.

    Forward Pass:
        The input is expected to be a batch of sequences of states, with shape
        (B, T, C, H, W). Each state is processed through the CNN and fully connected
        layers, and the output is a sequence of representations with shape
        (B, T, s_dim).

    Methods:
        forward(x):
            Processes the input states through the Encoder and outputs representations.
            Args:
                x (torch.Tensor): Input tensor of shape (B, T, C, H, W), where:
                    - B: Batch size.
                    - T: Sequence length.
                    - C: Number of channels.
                    - H: Height of each input image.
                    - W: Width of each input image.
            Returns:
                torch.Tensor: Encoded representations of shape (B, T, s_dim).
    """

    def __init__(self, s_dim, cnn_dim):
        super().__init__()

        data_path = "/scratch/DL24FA/train"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader = create_wall_dataloader(
            data_path=f"{data_path}",
            probing=False,
            device=device,
            train=True,
        )

        for batch in train_loader:
            # Extract the shape of a single state
            C, H, W = batch.states.shape[2:]
            input_shape = (C, H, W)
            break

        # Calculate dimensions for the fully connected layer
        C, H, W = input_shape
        H, W = (H - 1) // 2 + 1, (W - 1) // 2 + 1
        H, W = (H - 1) // 2 + 1, (W - 1) // 2 + 1
        fc_input_dim = H * W * cnn_dim * 2

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=cnn_dim,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=cnn_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=cnn_dim, out_channels=cnn_dim *
                      2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=cnn_dim * 2),
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

        # Process each frame in the batch
        x = x.reshape(B * T, C, H, W)
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = x.reshape(B, T, -1)

        return x


class Predictor(nn.Module):
    """
    Predictor module for JEPA (Joint Embedding Predictive Architecture).

    The Predictor takes a state and a corresponding action as input, and predicts
    the next state. It combines the current state and action into a single feature
    vector and processes it through a fully connected layer with a non-linear activation.

    Args:
        s_dim (int): Dimensionality of the state vector.
        u_dim (int): Dimensionality of the action vector.

    Attributes:
        fc (torch.nn.Sequential): Fully connected layer for predicting the next state.

    Forward Pass:
        The input is expected to be a batch of states and actions, with shapes
        (B, s_dim) and (B, u_dim), respectively. The output is the predicted next state
        with shape (B, s_dim).

    Methods:
        forward(state, action):
            Processes the input state and action to predict the next state.
            Args:
                state (torch.Tensor): Current state of shape (B, s_dim), where:
                    - B: Batch size.
                    - s_dim: Dimensionality of the state vector.
                action (torch.Tensor): Current action vector of shape (B, u_dim), where:
                    - B: Batch size.
                    - u_dim: Dimensionality of the action vector.
            Returns:
                torch.Tensor: Predicted next state of shape (B, s_dim).
    """

    def __init__(self, s_dim, u_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(s_dim + u_dim, s_dim),
            nn.BatchNorm1d(num_features=s_dim),
            nn.ReLU()
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
