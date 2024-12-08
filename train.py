
import torch
import torch.optim as optim
from models import JEPAWorldModel
from dataset import create_wall_dataloader

def train_model(epochs=50, lr=0.0002, save_path="jepa_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    data_path = "/scratch/DL24FA/train"
    train_ds = create_wall_dataloader(data_path, probing=False, device=device)

    # Initialize Model
    model = JEPAWorldModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch in train_ds:
            states = batch.states
            actions = batch.actions

            # Print the input shape
            print(f"Input shape of states: {states.shape}")
            print(f"Input shape of actions: {actions.shape}")

            # Compute Loss
            loss = model.compute_loss(states, actions)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_ds)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

    # Save the Model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")

if __name__ == "__main__":
    train_model()
