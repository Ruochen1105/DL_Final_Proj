
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

class WallSample:
    def __init__(self, states, actions, locations):
        self.states = states
        self.actions = actions
        self.locations = locations

class WallDataset(Dataset):
    def __init__(self, data_path, probing=False, device="cuda"):
        self.device = device
        self.states = np.load(f"{data_path}/states.npy", mmap_mode="r")
        self.actions = np.load(f"{data_path}/actions.npy")

        if probing:
            self.locations = np.load(f"{data_path}/locations.npy")
        else:
            self.locations = None

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        states = torch.from_numpy(self.states[idx]).float().to(self.device)
        actions = torch.from_numpy(self.actions[idx]).float().to(self.device)
        locations = torch.from_numpy(self.locations[idx]).float().to(self.device) if self.locations is not None else torch.empty(0).to(self.device)
        return WallSample(states=states, actions=actions, locations=locations)

def create_wall_dataloader(data_path, probing=False, device="cuda", batch_size=64, train=True):
    dataset = WallDataset(data_path=data_path, probing=probing, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=True, pin_memory=False)
    return loader
