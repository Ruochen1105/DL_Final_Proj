from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
import glob
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from models import JEPAWorldModel

def jepa_loss(predicted_states, target_states, reg_weight=1e-3):
    mse_loss = F.mse_loss(predicted_states, target_states)
    reg_loss = (predicted_states.pow(2).mean() + target_states.pow(2).mean()) * reg_weight
    return mse_loss + reg_loss

def train_jepa_model(model, train_loader, optimizer, num_epochs=10, device="cuda"):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            states = batch.states  # Shape: [batch_size, seq_length, channels, height, width]
            actions = batch.actions  # Shape: [batch_size, seq_length-1, 2]
            optimizer.zero_grad()

            # Predicted states from the JEPA model
            predicted_states = model(states, actions)

            # Encode target future states (skip the first state since actions start at 0)
            target_states = torch.stack(
                [model.encoder(states[:, t]) for t in range(1, states.size(1))], dim=1
            )

            # Compute loss
            loss = jepa_loss(predicted_states, target_states)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")
    print("Training Complete")

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/scratch/DL24FA"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        batch_size=64,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    return probe_train_ds, probe_val_ds


def load_model():
    """Load or initialize the model."""
    model = JEPAWorldModel(repr_dim=256, action_dim=2, input_channels=2)
    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")

if __name__ == "__main__":
    device = get_device()
    probe_train_ds, probe_val_ds = load_data(device)
    train_loader = probe_train_ds
    #train_loader = DataLoader(probe_train_ds, batch_size=64, shuffle=True)
    model = load_model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_jepa_model(model, train_loader, optimizer, num_epochs=20, device=device)

    evaluate_model(device, model, probe_train_ds, probe_val_ds)

