import torch
from torch import nn


class LGL_VAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # Encoder maps continuous vector -> hidden
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU()
        )

        # Latent parameters
        self.to_mu = nn.Linear(in_features=hidden_dim, out_features=latent_dim)
        self.to_logvar = nn.Linear(in_features=hidden_dim, out_features=latent_dim)

        # Decoder maps latent -> reconstructed continuous vector
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=input_dim)
        )
        
    def forward(self, input_vector):
        # input_vector: [batch, input_dim] continuous values
        x = input_vector.view(input_vector.shape[0], -1).float()

        # Encode
        hidden = self.encoder(x)

        mu = self.to_mu(hidden)      # [batch, latent_dim]
        logvar = self.to_logvar(hidden)  # [batch, latent_dim]

        # Reparameterization trick
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

        # Decode
        reconstruction = self.decoder(z)  # [batch, input_dim]

        return reconstruction, mu, logvar

