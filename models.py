from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)

class JEPAEncoder(nn.Module):
    def __init__(self, input_channels=2, repr_dim=256):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc = None
        self.repr_dim = repr_dim

    def forward(self, x):
        x = self.conv_net(x)
        if self.fc is None:
            flatten_size = x.size(1) * x.size(2) * x.size(3)
            self.fc = nn.Linear(flatten_size, self.repr_dim).to(x.device)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class JEPAPredictor(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2):
        super().__init__()
        self.predictor = nn.GRU(input_size=repr_dim + action_dim, hidden_size=repr_dim, batch_first=True)

    def forward(self, encoded_state, actions):
        # Concatenate encoded state and actions
        inputs = torch.cat([encoded_state[:, :-1], actions], dim=-1)
        predicted, _ = self.predictor(inputs)
        return predicted

class JEPAWorldModel(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2, input_channels=2):
        super().__init__()
        self.encoder = JEPAEncoder(input_channels, repr_dim)
        self.predictor = JEPAPredictor(repr_dim, action_dim)
        self.repr_dim = repr_dim

    def forward(self, states, actions):
        batch_size, seq_len, _, _, _ = states.size()
        # Encode all states
        encoded_states = torch.stack([self.encoder(states[:, t]) for t in range(seq_len)], dim=1)
        # Predict future embeddings
        predicted_states = self.predictor(encoded_states, actions)
        return predicted_states

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
