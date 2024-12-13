import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import create_wall_dataloader
from models import JEPA


def barlow_twins_loss(predicted, target, lambda_=1e-2):
    """
    Computes the Barlow Twins loss.

    Args:
        predicted (torch.Tensor): Embeddings from the model.
        target (torch.Tensor): Embeddings from the augmented view.
        lambda_ (float): Hyperparameter for off-diagonal weight.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Normalize embeddings along the last dimension (representation dimension)
    predicted = F.normalize(predicted, dim=-1)
    target = F.normalize(target, dim=-1)

    # Reshape tensors to combine batch and temporal dimensions
    batch_size, temporal_dim, repr_dim = predicted.size()
    predicted = predicted.view(batch_size * temporal_dim, repr_dim)
    target = target.view(batch_size * temporal_dim, repr_dim)

    # Compute the cross-correlation matrix
    cross_correlation = torch.mm(
        predicted.T, target) / (batch_size * temporal_dim)

    # Create the identity matrix for comparison
    identity = torch.eye(repr_dim, device=predicted.device)

    # Compute the invariance term (diagonal elements)
    invariance_loss = torch.sum(F.relu(1 - torch.diag(cross_correlation)) ** 2)

    # Compute the redundancy reduction term (off-diagonal elements)
    off_diagonal_mask = ~identity.bool()
    off_diagonal_loss = torch.sum(cross_correlation[off_diagonal_mask] ** 2)

    loss = invariance_loss + lambda_ * off_diagonal_loss

    return loss


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
    tau = 0.999

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in tqdm(train_loader, total=len(train_loader), smoothing=50/len(train_loader)):
            states, actions = batch.states, batch.actions
            states, actions = states.to(device), actions.to(device)

            predicted_next_states = model(
                states, actions)  # Shape: (B, T, s_dim)

            next_states_true = model.target_encoder(states)

            loss = barlow_twins_loss(predicted_next_states, next_states_true)

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

        scheduler.step(avg_epoch_loss)

        if avg_epoch_loss < best_loss - 1e-4:
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
    s_dim = 128
    cnn_dim = 16
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
