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
MODEL_SAVE_PATH = "./outputs/Large_Latent_Dim/trained_vae_model_RMSD_64_Latent_Dim.pt"

# Model hyperparameters
HIDDEN_DIM = 4096
LATENT_DIM = 64

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-4  # Reduced learning rate for stability
NUM_EPOCHS = 100
L2_REGULARIZATION = 0.0001
BETA = 0.0001  # KL divergence weight

# Initialize Utils
utils = Utils()
print(f"Using device: {utils.device}")


def compute_rmsd(reconstructed: torch.Tensor, target: torch.Tensor, num_atoms: int) -> torch.Tensor:
    """
    Compute per-sample RMSD between reconstructed and target flattened coordinate vectors.

    Args:
        reconstructed: [batch_size, 3 * num_atoms] tensor
        target:        [batch_size, 3 * num_atoms] tensor
        num_atoms:     number of atoms per structure

    Returns:
        rmsd: [batch_size] tensor of RMSD values
    """
    batch_size = reconstructed.shape[0]

    # Reshape to [batch_size, num_atoms, 3]
    recon_coords = reconstructed.view(batch_size, num_atoms, 3)
    target_coords = target.view(batch_size, num_atoms, 3)

    # Mean squared error over all coordinates, then square root
    mse_per_sample = torch.mean((recon_coords - target_coords) ** 2, dim=(1, 2))
    rmsd = torch.sqrt(mse_per_sample + 1e-12)
    return rmsd


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute batch-mean KL divergence between q(z|x)=N(mu,exp(logvar)) and p(z)=N(0,I)."""
    # Clamp for numerical stability
    logvar = torch.clamp(logvar, min=-10, max=10)
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return torch.mean(kl_per_sample)


def vae_loss_rmsd_kl(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    num_atoms: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Total loss = mean RMSD + BETA * KL
    Returns (total_loss, mean_rmsd, kl_loss)
    """
    rmsd = compute_rmsd(reconstructed, target, num_atoms)
    mean_rmsd = torch.mean(rmsd)
    kl_loss = kl_divergence(mu, logvar)
    total = mean_rmsd + BETA * kl_loss
    return total, mean_rmsd, kl_loss


# Load PDB dataset
print(f"Loading PDB dataset from {PDB_PATH}")
dataset = Chromosome21PDBDataset(
    pdb_path=PDB_PATH,
    record_types=("ATOM",),
    center=True,  # Center coordinates
    scale=1.0  # Don't scale - keep raw coordinate ranges
)

print(f"Dataset loaded: {len(dataset)} models, {dataset.vector_length} coordinates per model")
print(f"Number of atoms per model: {dataset.atoms_per_model}")

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
    total_rmsd = 0.0
    total_kl = 0.0

    # Training phase with loss tracking in progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = batch.to(utils.device)

        # Forward pass
        optimizer.zero_grad()
        reconstructed, mu, logvar = vae(batch)

        # Calculate loss with KL regularization
        loss, mean_rmsd, kl_loss = vae_loss_rmsd_kl(
            reconstructed, batch, mu, logvar, dataset.atoms_per_model
        )

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        total_rmsd += mean_rmsd.item()
        total_kl += kl_loss.item()

    # Calculate average training loss for this epoch
    avg_loss = total_loss / len(train_loader)
    avg_rmsd = total_rmsd / len(train_loader)
    avg_kl = total_kl / len(train_loader)

    # Validation phase
    vae.eval()
    val_total_loss = 0.0
    val_total_rmsd = 0.0
    val_total_kl = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(utils.device)
            reconstructed, mu, logvar = vae(batch)
            loss, mean_rmsd, kl_loss = vae_loss_rmsd_kl(
                reconstructed, batch, mu, logvar, dataset.atoms_per_model
            )
            val_total_loss += loss.item()
            val_total_rmsd += mean_rmsd.item()
            val_total_kl += kl_loss.item()

    # Calculate average validation loss
    avg_val_loss = val_total_loss / len(val_loader)
    avg_val_rmsd = val_total_rmsd / len(val_loader)
    avg_val_kl = val_total_kl / len(val_loader)

    # Set model back to training mode
    vae.train()

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} - "
        f"Train Total: {avg_loss:.6f}, RMSD: {avg_rmsd:.6f}, KL: {avg_kl:.6f} | "
        f"Val Total: {avg_val_loss:.6f}, RMSD: {avg_val_rmsd:.6f}, KL: {avg_val_kl:.6f}"
    )

    # Save the trained model periodically
    if (epoch + 1) % 25 == 0:
        torch.save(vae.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")

# Save final model
torch.save(vae.state_dict(), MODEL_SAVE_PATH)
print(f"Final model saved to {MODEL_SAVE_PATH}")


