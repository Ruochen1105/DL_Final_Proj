
import torch
from torch import nn
from tqdm import tqdm
from normalizer import Normalizer
from models import vicreg_loss

class ProbingEvaluator:
    def __init__(self, device, model, probe_train_ds, probe_val_ds, quick_debug=False):
        self.device = device
        self.model = model.eval()
        self.ds = probe_train_ds
        self.val_ds = probe_val_ds
        self.normalizer = Normalizer()
        self.quick_debug = quick_debug

    def train_pred_prober(self):
        repr_dim = self.model.repr_dim
        prober = nn.Sequential(
            nn.Linear(repr_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        ).to(self.device)

        optimizer = torch.optim.Adam(prober.parameters(), lr=0.0002)
        epochs = 20

        for epoch in tqdm(range(epochs), desc="Training Prober"):
            for batch in tqdm(self.ds, desc="Train Step"):
                pred_encs = self.model(states=batch.states, actions=batch.actions).detach()
                target = self.normalizer.normalize_location(batch.locations.to(self.device))

                pred_locs = torch.stack([prober(enc) for enc in pred_encs], dim=1)
                loss = nn.MSELoss()(pred_locs, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.quick_debug:
                    break

        return prober

    @torch.no_grad()
    def evaluate_all(self, prober):
        avg_losses = {}
        for prefix, val_ds in self.val_ds.items():
            avg_losses[prefix] = self.evaluate_pred_prober(prober, val_ds, prefix)
        return avg_losses

    @torch.no_grad()
    def evaluate_pred_prober(self, prober, val_ds, prefix=""):
        probing_losses = []
        for batch in tqdm(val_ds, desc=f"Eval {prefix}"):
            pred_encs = self.model(states=batch.states, actions=batch.actions).detach()
            target = self.normalizer.normalize_location(batch.locations.to(self.device))

            pred_locs = torch.stack([prober(enc) for enc in pred_encs], dim=1)
            loss = nn.MSELoss()(pred_locs, target)
            probing_losses.append(loss.cpu())

        avg_loss = torch.stack(probing_losses).mean().item()
        return avg_loss
