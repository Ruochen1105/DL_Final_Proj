import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import create_wall_dataloader
from models import JEPA


def barlow_twins_loss(predicted, target, lambda_corr=5e-1):
    """
    Computes the Barlow Twins-inspired loss for decorrelation.

    Args:
        predicted: Predicted next states, shape (B, T, s_dim).
        target: True next states, shape (B, T, s_dim).
        lambda_corr: Weight for the cross-correlation loss term.

    Returns:
        Cross-Correlation loss.
    """
    # Normalize predicted and target embeddings
    B, T, s_dim = predicted.size()
    predicted = predicted.view(B * T, s_dim)
    target = target.view(B * T, s_dim)

    predicted_norm = F.normalize(predicted, dim=0)
    target_norm = F.normalize(target, dim=0)

    # Cross-Correlation Matrix
    corr_matrix = torch.mm(predicted_norm.T, target_norm) / (B * T)

    # Cross-Correlation Loss
    on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
    off_diag = off_diag = cross_corr.fill_diagonal_(0).pow_(2).sum()
    cross_corr_loss = on_diag + lambda_corr * off_diag

    return cross_corr_loss


def train_model(model, train_loader, optimizer, scheduler, epochs, device, save_path="./", patience=5):
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

            loss = barlow_twins_loss(predicted_next_states, next_states_true)

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
    cnn_dim = 32
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
    epochs = 100
    train_model(model, train_loader, optimizer, scheduler, epochs, device)
