import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from VAE.Simple_VAE import LGL_VAE
from Utils.Utils import Utils
from DataLoader.Chromosome21PDBDataset import Chromosome21PDBDataset
import torch.optim as optim
import random
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Path to PDB file
PDB_PATH = "./Data/chromosome21_aligned.pdb"
# Path to save trained model
MODEL_SAVE_PATH = "./outputs/ChromatinVAE/trained_vae_model_Large.pt"

# Model hyperparameters
HIDDEN_DIM = 1024
LATENT_DIM = 32

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 5e-4  # Reduced learning rate for stability
NUM_EPOCHS = 1000
BETA = 1  # Reduced KL weight for stability
L2_REGULARIZATION = 0.0001

# Initialize Utils
utils = Utils()
print(f"Using device: {utils.device}")


def vae_loss(reconstructed, target, mu, logvar):
    """VAE loss for continuous coordinate data."""
    # Reconstruction loss (MSE for continuous coordinates)
    reconstruction_loss = nn.MSELoss()
    recon_loss = reconstruction_loss(reconstructed, target)
    
    # KL divergence loss with numerical stability
    # Clamp logvar to prevent numerical issues
    logvar = torch.clamp(logvar, min=-10, max=10)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.mean(kl_loss)
    
    # Total loss
    total_loss = recon_loss + BETA * kl_loss
    
    return total_loss, recon_loss, kl_loss

# Load PDB dataset
print(f"Loading PDB dataset from {PDB_PATH}")
dataset = Chromosome21PDBDataset(
    pdb_path=PDB_PATH,
    record_types=("ATOM",),
    center=True,  # Center coordinates
    scale=1.0  # Don't scale - let's see raw coordinate ranges first
)

print(f"Dataset loaded: {len(dataset)} models, {dataset.vector_length} coordinates per model")

# Create model
vae = LGL_VAE(hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, input_dim=dataset.vector_length)
vae = vae.to(utils.device)

print(f"Model parameters: {sum(p.numel() for p in vae.parameters()):,}")

# Wrap model for multi-GPU if available
vae = utils.wrap_model_for_multi_gpu(vae)

# Check for previously saved model and load if exists
if Path(MODEL_SAVE_PATH).exists():
    vae.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=utils.device))
    print(f"Loaded previously trained model from {MODEL_SAVE_PATH}")
else:
    print("No previous model found, starting training from scratch")

# Split dataset into train/validation (90%/10%)
total_size = len(dataset)
train_size = int(0.9 * total_size)
val_size = total_size - train_size

torch.manual_seed(0)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    collate_fn=Chromosome21PDBDataset.collate_batch
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    collate_fn=Chromosome21PDBDataset.collate_batch
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Initialize optimizer
optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION)

# Training loop
vae.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    
    # Training phase with loss tracking in progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = batch.to(utils.device)
        
        # Forward pass
        optimizer.zero_grad()
        reconstructed, mu, logvar = vae(batch)
        
        # Calculate loss
        loss, recon_loss, kl_loss = vae_loss(reconstructed, batch, mu, logvar)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        
    
    # Calculate average losses for this epoch
    avg_loss = total_loss / len(train_loader)
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_kl_loss = total_kl_loss / len(train_loader)
    
    # Validation phase
    vae.eval()
    val_total_loss = 0.0
    val_total_recon_loss = 0.0
    val_total_kl_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            batch = batch.to(utils.device)
            reconstructed, mu, logvar = vae(batch)
            loss, recon_loss, kl_loss = vae_loss(reconstructed, batch, mu, logvar)
            
            val_total_loss += loss.item()
            val_total_recon_loss += recon_loss.item()
            val_total_kl_loss += kl_loss.item()
    
    # Calculate average validation losses
    avg_val_loss = val_total_loss / len(val_loader)
    avg_val_recon_loss = val_total_recon_loss / len(val_loader)
    avg_val_kl_loss = val_total_kl_loss / len(val_loader)
    
    # Set model back to training mode
    vae.train()
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f} | Val Loss: {avg_val_loss:.4f}, Recon: {avg_val_recon_loss:.4f}, KL: {avg_val_kl_loss:.4f}")

    # Save the trained model every 100 epochs
    if (epoch + 1) % 100 == 0:
        torch.save(vae.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")

# Save final model
torch.save(vae.state_dict(), MODEL_SAVE_PATH)
print(f"Final model saved to {MODEL_SAVE_PATH}")



