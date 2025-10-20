import torch
from torch import nn
import torch.nn.functional as F


class LGL_VAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, dropout_rate=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        # Encoder: Progressive compression from input_dim -> latent_dim
        # Calculate intermediate dimensions for smooth compression
        compression_steps = 5
        dim_step = (input_dim - latent_dim) // compression_steps
        
        encoder_layers = []
        current_dim = input_dim
        
        for i in range(compression_steps):
            next_dim = max(current_dim - dim_step, latent_dim)
            encoder_layers.extend([
                nn.Linear(in_features=current_dim, out_features=next_dim),
                nn.BatchNorm1d(next_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            current_dim = next_dim
        
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent parameters
        self.to_mu = nn.Linear(in_features=current_dim, out_features=latent_dim)
        self.to_logvar = nn.Linear(in_features=current_dim, out_features=latent_dim)

        # Decoder: Progressive decompression from latent_dim -> input_dim
        decompression_steps = 5
        dim_step = (input_dim - latent_dim) // decompression_steps
        
        decoder_layers = []
        current_dim = latent_dim
        
        for i in range(decompression_steps):
            next_dim = min(current_dim + dim_step, input_dim)
            decoder_layers.extend([
                nn.Linear(in_features=current_dim, out_features=next_dim),
                nn.BatchNorm1d(next_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            current_dim = next_dim
        
        # Final layer to ensure exact input_dim output
        if current_dim != input_dim:
            decoder_layers.append(nn.Linear(in_features=current_dim, out_features=input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Skip connections for preserving structural details
        # Store intermediate encoder outputs for skip connections
        self.skip_connections = []
        self.skip_dims = []
        
        # Calculate skip connection dimensions
        temp_dim = input_dim
        for i in range(compression_steps):
            temp_dim = max(temp_dim - dim_step, latent_dim)
            if temp_dim > latent_dim:  # Only skip if dimension is larger than latent
                self.skip_dims.append(temp_dim)
        
        # Create skip connection layers
        self.skip_layers = nn.ModuleList()
        for skip_dim in self.skip_dims:
            self.skip_layers.append(nn.Linear(skip_dim, latent_dim))
        
    def forward(self, input_vector):
        # input_vector: [batch, input_dim] continuous values
        x = input_vector.view(input_vector.shape[0], -1).float()

        # Encode with skip connections
        hidden = x
        skip_outputs = []
        
        # Manual forward pass to capture intermediate outputs
        for i, layer in enumerate(self.encoder):
            hidden = layer(hidden)
            # Capture outputs after each linear layer for skip connections
            if isinstance(layer, nn.Linear) and len(skip_outputs) < len(self.skip_layers):
                skip_outputs.append(hidden)

        mu = self.to_mu(hidden)      # [batch, latent_dim]
        logvar = self.to_logvar(hidden)  # [batch, latent_dim]

        # Reparameterization trick
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

        # Add skip connection information to latent representation
        if len(skip_outputs) > 0:
            skip_info = torch.zeros_like(z)
            for i, skip_out in enumerate(skip_outputs):
                if i < len(self.skip_layers):
                    skip_info += self.skip_layers[i](skip_out)
            z = z + 0.1 * skip_info  # Weighted combination

        # Decode
        reconstruction = self.decoder(z)  # [batch, input_dim]

        return reconstruction, mu, logvar, z

