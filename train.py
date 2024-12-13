import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import create_wall_dataloader
from models import JEPA

from torchvision import transforms
from torch.nn.functional import mse_loss

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
    # Define augmentation pipeline for SimCLR-like views
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    os.makedirs(save_path, exist_ok=True)

    model.to(device)

    loss_fn = nn.MSELoss()
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

            loss = loss_fn(predicted_next_states, next_states_true)

            # Generate two augmented views
            view_1 = augmentation(states)
            view_2 = augmentation(states)

            # Encode both views
            encoded_view_1 = model.encoder(view_1)
            encoded_view_2 = model.encoder(view_2)

            # Predict future states
            pred_view_1 = model.predictor(encoded_view_1[:, :-1].detach(), actions)
            pred_view_2 = model.predictor(encoded_view_2[:, :-1].detach(), actions)

            # Compute contrastive loss between views
            contrastive_loss = mse_loss(pred_view_1, pred_view_2)

            # Add the contrastive loss to the original loss
            loss += contrastive_loss * 0.5  # Weight the loss term

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
    epochs = 200
    train_model(model, train_loader, optimizer, scheduler, epochs, device)
