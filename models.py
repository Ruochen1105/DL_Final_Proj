from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


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
            During training:
                states: [B, T, Ch, H, W]
            During inference:
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
    def __init__(self, s_dim, cnn_dim):
        """
        Joint Embedding Predictive Architecture (JEPA).

        Args:
            s_dim (int): Space dimensionality for the encoded states.
            cnn_dim (int): Base dimensionality for the CNN.
        """
        super().__init__()
        self.encoder = Encoder(s_dim, cnn_dim)
        self.predictor = Predictor(s_dim)
        self.repr_dim = s_dim

    def forward(self, states, actions):
        """
        Forward pass for the JEPA model.

        Args:
            states (torch.Tensor): Sequence of states of shape (B, T, C, H, W).
            actions (torch.Tensor): Sequence of actions of shape (B, T-1, 2).

        Returns:
            torch.Tensor: Predicted next states of shape (B, T, s_dim).
        """
        # Encode the states into representations
        states = self.encoder(states)  # shape: (B, T, s_dim)
        if states.shape[1] == 1:  # inferencing
            # Use states and actions to predict the next states
            predicted_states = [states]
            for t in range(actions.shape[1]):
                predicted_state = self.predictor(
                    predicted_states[-1], actions[:, t])
                predicted_states.append(predicted_state)
            predicted_states = torch.cat(
                predicted_states, dim=1)
        else:  # training
            predicted_states = self.predictor(states[:, :-1], actions)
            initial_state = states[:, 0].unsqueeze(1)
            predicted_states = torch.cat(
                (initial_state, predicted_states), dim=1)

        return predicted_states


class Encoder(nn.Module):
    """
    Encoder module for JEPA.

    Args:
        s_dim (int): Dimensionality of the space for the encoded state.
        cnn_dim (int): Base number of channels used in the CNN.

    Forward Pass:
        The input is expected to be a batch of sequences of states, with shape
        (B, T, C, H, W). Each state is processed through the CNN and fully connected
        layers, and the output is a sequence of representations with shape
        (B, T, s_dim).

    Returns:
        torch.Tensor: Encoded representations of shape (B, T, s_dim).
    """

    def __init__(self, s_dim, cnn_dim):
        super().__init__()
        C, H, W = 2, 65, 65

        # Calculate dimensions for the fully connected layer
        H, W = (H - 1) // 2 + 1, (W - 1) // 2 + 1
        H, W = (H - 1) // 2 + 1, (W - 1) // 2 + 1
        fc_input_dim = H * W * cnn_dim * 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=cnn_dim,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(cnn_dim),
            nn.ReLU()
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cnn_dim,
                      cnn_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_dim // 4,
                      cnn_dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=cnn_dim, out_channels=cnn_dim * 2,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(cnn_dim * 2),
            nn.ReLU()
        )

        self.dropout = nn.Dropout2d(p=0.2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(fc_input_dim, s_dim)

    def forward(self, x):
        B, T, C, H, W = x.size()

        # Process each frame in the batch
        x = x.reshape(B * T, C, H, W)

        x = self.conv1(x)
        channel_weights = self.channel_attention(x)
        x = x * channel_weights
        x = F.relu(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = F.relu(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = x.reshape(B, T, -1)

        return x


class Predictor(nn.Module):
    """
    Attention-based Predictor module for JEPA.

    The Predictor takes a state and a corresponding action as input and predicts
    the next state using an attention mechanism.

    Args:
        s_dim (int): Dimensionality of the state vector.
        a_dim (int): Dimensionality of the action vector.

    Attributes:
        query (torch.nn.Linear): Linear layer to generate query vectors from state.
        key (torch.nn.Linear): Linear layer to generate key vectors from action.
        value (torch.nn.Linear): Linear layer to generate value vectors from action.
        output (torch.nn.Linear): Linear layer to project the attention output to the predicted state.

    Forward Pass:
        The input is expected to be a batch of states and actions, with shapes
        (B, s_dim) and (B, a_dim), respectively. The output is the predicted next state
        with shape (B, s_dim).

    Methods:
        forward(state, action):
            Processes the input state and action using attention to predict the next state.
    """

    def __init__(self, s_dim, a_dim=2):
        super().__init__()

        # Attention mechanism layers
        self.query = nn.Linear(s_dim, s_dim)
        self.key = nn.Linear(a_dim, s_dim)
        self.value = nn.Linear(a_dim, s_dim)

        # Output projection
        self.output = nn.Linear(s_dim, s_dim)

    def forward(self, state, action):
        B, T, s_dim = state.shape
        state = state.reshape(B * T, s_dim)
        action = action.reshape(B * T, -1)

        Q = self.query(state)  # Shape: (B*T, s_dim)
        K = self.key(action)  # Shape: (B*T, s_dim)
        V = self.value(action)  # Shape: (B*T, s_dim)

        attention_scores = torch.softmax(
            Q @ K.T / (s_dim ** 0.5), dim=-1)  # Shape: (B*T, B*T)

        attention_output = attention_scores @ V  # Shape: (B*T, s_dim)

        predicted_state = self.output(attention_output)  # Shape: (B*T, s_dim)

        predicted_state = nn.ReLU(predicted_state)

        predicted_state = predicted_state.reshape(B, T, s_dim)

        return predicted_state
