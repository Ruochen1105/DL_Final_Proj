import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import create_wall_dataloader
from models import JEPA


def vicreg_loss(z1, z2, lambda1=10, lambda2=25.0, lambda3=1.0, eps=1e-4):
    invariance_loss = F.mse_loss(z1, z2)

    z1_flat = z1.view(-1, z1.shape[-1])  # Shape: (B * T, s_dim)
    z2_flat = z2.view(-1, z2.shape[-1])  # Shape: (B * T, s_dim)

    def variance_regularizer(z):
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(F.relu(1 - std))   # Penalize std < 1

    variance_loss = variance_regularizer(
        z1_flat) + variance_regularizer(z2_flat)

    def covariance_regularizer(z):
        batch_size, embedding_dim = z.shape
        z_centered = z - z.mean(dim=0, keepdim=True)
        covariance_matrix = (z_centered.T @ z_centered) / (batch_size - 1)
        off_diagonal = covariance_matrix - \
            torch.diag(torch.diag(covariance_matrix))
        return (off_diagonal ** 2).sum() / embedding_dim

    covariance_loss = covariance_regularizer(
        z1_flat) + covariance_regularizer(z2_flat)

    loss = lambda1 * invariance_loss + lambda2 * \
        variance_loss + lambda3 * covariance_loss
    return loss


def train_model(model, train_loader, optimizer, scheduler, epochs, device, save_path="./", patience=50):
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
    tau = 0.9

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in tqdm(train_loader, total=len(train_loader), smoothing=50/len(train_loader)):
            states, actions = batch.states, batch.actions
            states, actions = states.to(device), actions.to(device)

            predicted_next_states = model(
                states, actions)  # Shape: (B, T, s_dim)

            next_states_true = model.target_encoder(states)

            loss = vicreg_loss(predicted_next_states, next_states_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for target_param, online_param in zip(model.target_encoder.parameters(), model.encoder.parameters()):
                    target_param.data = tau * target_param.data + \
                        (1 - tau) * online_param.data

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
    epochs = 200
    train_model(model, train_loader, optimizer, scheduler, epochs, device)
