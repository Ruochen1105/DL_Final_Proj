import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import create_wall_dataloader
from models import JEPA


def train_model(model, train_loader, optimizer, scheduler, epochs, device, save_path="./checkpoint", patience=10):
    """
    Train the JEPA model using an energy-based approach.

    Args:
        model (JEPA): The JEPA model to train.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        epochs (int): Number of training epochs.
        device (torch.device): Device to run the training on (e.g., 'cuda' or 'cpu').
        save_path (str): Path to the checkpoint.
        patience (int): Number of epochs before early stopping.

    Returns:
        list: List of average training losses for each epoch.
    """
    os.makedirs(save_path, exist_ok=True)

    loss_fn = nn.MSELoss()
    best_loss = float("inf")
    patience_counter = 0
    losses = []

    model.train()

    for epoch in tqdm(range(epochs), desc=f"training", total=epochs):
        epoch_loss = 0.0

        for batch in tqdm(train_loader, total=len(train_loader)):
            states, actions = batch.states, batch.actions
            states, actions = states.to(device), actions.to(device)

            next_states_true = states[:, 1:]  # Shape: (B, T-1, C, H, W)
            current_states = states[:, :-1]  # Shape: (B, T-1, C, H, W)

            next_states_true = model.encoder(next_states_true)

            predicted_next_states = model(
                current_states, actions)  # Shape: (B, T-1, s_dim)

            print(predicted_next_states.shape)
            print(next_states_true.shape)

            loss = loss_fn(predicted_next_states, next_states_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        losses.append(avg_epoch_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

        scheduler.step(avg_epoch_loss)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0

            # Save the model checkpoint
            checkpoint_path = os.path.join(save_path, f"best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model improved. Saved checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return losses


if __name__ == "__main__":
    s_dim = 256
    cnn_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = "/scratch/DL24FA/train"

    train_loader = create_wall_dataloader(
        data_path=f"{data_path}",
        probing=False,
        device=device,
        train=True,
    )

    # Model, optimizer, and device
    model = JEPA(s_dim, cnn_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    # Train the model
    epochs = 1
    train_model(model, train_loader, optimizer, scheduler, epochs, device)
