
import torch
from torch import nn
import torch.nn.functional as F

# Encoder for visual observations
class Encoder(nn.Module):
    def __init__(self, input_channels=2, latent_dim=256):
        super().__init__()

        # Define convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),              # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),             # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),            # 8x8 -> 4x4
            nn.ReLU()
        )

        # Calculate dynamic output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 64, 64)
            conv_output = self.conv(dummy_input)
            conv_output_size = conv_output.view(1, -1).shape[1]
            print(f"Computed CNN output size: {conv_output_size}")

        # Fully connected layer
        self.fc = nn.Linear(conv_output_size, latent_dim)

    def forward(self, x):
        x = self.conv(x)         # B x C_out x H_out x W_out
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)        # B x latent_dim

# Embedding actions into the latent space
class ActionEmbedder(nn.Module):
    def __init__(self, action_dim=2, embed_dim=256):
        super().__init__()
        self.fc = nn.Linear(action_dim, embed_dim)

    def forward(self, actions):
        return self.fc(actions)

# Predictor head for contrastive loss
class PredictorHead(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x):
        return self.head(x)

# Full JEPA model with Transformer
class JEPAWorldModel(nn.Module):
    def __init__(self, latent_dim=256, action_dim=2, n_layers=4, n_heads=8):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.target_encoder = Encoder(latent_dim=latent_dim)
        self.action_embedder = ActionEmbedder(action_dim, latent_dim)
        self.predictor_head = PredictorHead(latent_dim)

        self.transformer = nn.Transformer(
            d_model=latent_dim,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )

        self.repr_dim = latent_dim

    def forward(self, states, actions):
        B, T, C, H, W = states.shape
        encoded_states = self.encoder(states[:, 0])
        action_embeds = self.action_embedder(actions)

        memory = encoded_states.unsqueeze(1)
        outputs = [memory]

        for t in range(1, T):
            action_input = action_embeds[:, t-1].unsqueeze(1)
            pred_state = self.transformer(src=memory, tgt=action_input)[:, -1, :]
            pred_state = self.predictor_head(pred_state)

            memory = torch.cat([memory, pred_state.unsqueeze(1)], dim=1)
            outputs.append(pred_state.unsqueeze(1))

        return torch.cat(outputs, dim=1)

    def compute_loss(self, states, actions):
        # Extract initial observation for the encoder
        initial_states = states[:, 0]  # Shape: [64, 2, 64, 64]

        predicted_states = self.forward(states, actions)

        with torch.no_grad():
            target_states = self.target_encoder(initial_states)

        loss = vicreg_loss(predicted_states[:, 0], target_states)
        return loss

# VICReg Loss for collapse prevention
def vicreg_loss(pred, target, lambda_v=25.0, mu=25.0, nu=1.0):
    mse_loss = F.mse_loss(pred, target)
    pred_std = torch.sqrt(pred.var(dim=0) + 1e-4)
    target_std = torch.sqrt(target.var(dim=0) + 1e-4)

    variance_loss = (F.relu(1 - pred_std).mean() + F.relu(1 - target_std).mean())

    pred_centered = pred - pred.mean(dim=0)
    target_centered = target - target.mean(dim=0)

    pred_cov = (pred_centered.T @ pred_centered) / (pred.shape[0] - 1)
    target_cov = (target_centered.T @ target_centered) / (target.shape[0] - 1)

    pred_cov_loss = (pred_cov - torch.eye(pred.shape[1]).cuda()).pow(2).sum()
    target_cov_loss = (target_cov - torch.eye(target.shape[1]).cuda()).pow(2).sum()

    cov_loss = pred_cov_loss + target_cov_loss

    total_loss = lambda_v * mse_loss + mu * variance_loss + nu * cov_loss
    return total_loss

# Momentum Updater for the target encoder
class MomentumUpdater:
    @staticmethod
    def update(target_net, online_net, beta=0.99):
        for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data = beta * target_param.data + (1 - beta) * online_param.data
