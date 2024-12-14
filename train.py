import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import create_wall_dataloader
from models import JEPA


def barlow_twins_loss(z_a, z_b, lambda_=1):
    """
    Based on arxiv.org/abs/2103.03230
    """
    B, T, repr_dim = z_a.shape
    z_a = z_a.view(B * T, -1)
    z_b = z_b.view(B * T, -1)

    # Normalize representations along the batch dimension
    z_a_norm = (z_a - z_a.mean(dim=0)) / z_a.std(dim=0, unbiased=False)
    z_b_norm = (z_b - z_b.mean(dim=0)) / z_b.std(dim=0, unbiased=False)

    # Compute cross-correlation matrix
    c = torch.mm(z_a_norm.T, z_b_norm) / z_a.size(0)

    # Compute loss
    c_diff = (c.to("cuda") - torch.eye(z_a.size(1)).to("cuda")).pow(2)

    # Multiply off-diagonal elements of c_diff by lambda
    c_diff[~torch.eye(z_a.size(1), dtype=bool)] *= lambda_

    # Sum all elements of c_diff
    loss = c_diff.sum()

    return loss


def train_model(model, train_loader, optimizer, scheduler, epochs, device, save_path="./", patience=10, checkpoint=None):
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

    if checkpoint:
        print("Continue from checkpoint")
        model.load_state_dict(torch.load(
            checkpoint, weights_only=False))

    model.to(device)

    best_loss = float("inf")
    patience_counter = 0
    losses = []

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in tqdm(train_loader, total=len(train_loader), smoothing=50/len(train_loader)):
            states, actions = batch.states, batch.actions
            states, actions = states.to(device), actions.to(device)

            predicted_next_states = model(
                states, actions)  # Shape: (B, T, s_dim)

            next_states_true = model.encoder(states)

            loss = barlow_twins_loss(
                predicted_next_states, next_states_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        losses.append(avg_epoch_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

        # scheduler.step(avg_epoch_loss)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0

            # Save the model checkpoint
            checkpoint_path = os.path.join(save_path, f"model_weights.pth")
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
    optimizer = optim.Adam(model.parameters(), lr=5e-2)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    # Train the model
    epochs = 200
    train_model(model=model, train_loader=train_loader, optimizer=optimizer, scheduler=scheduler,
                epochs=epochs, device=device, checkpoint="best_model.pth")
