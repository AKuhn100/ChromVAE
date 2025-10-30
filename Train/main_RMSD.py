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
MODEL_SAVE_PATH = "./outputs/trained_vae_model_RMSD.pt"

# Model hyperparameters
HIDDEN_DIM = 4096
LATENT_DIM = 64

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-4  # Reduced learning rate for stability
NUM_EPOCHS = 1000
L2_REGULARIZATION = 0.0001

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


def rmsd_loss(reconstructed: torch.Tensor, target: torch.Tensor, num_atoms: int) -> torch.Tensor:
    """Return mean RMSD over the batch (no KL term)."""
    rmsd = compute_rmsd(reconstructed, target, num_atoms)
    return torch.mean(rmsd)


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

    # Training phase with loss tracking in progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = batch.to(utils.device)

        # Forward pass
        optimizer.zero_grad()
        reconstructed, mu, logvar = vae(batch)

        # Calculate RMSD loss (no KL or auxiliary terms)
        loss = rmsd_loss(reconstructed, batch, dataset.atoms_per_model)

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    # Calculate average training loss for this epoch
    avg_loss = total_loss / len(train_loader)

    # Validation phase
    vae.eval()
    val_total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(utils.device)
            reconstructed, mu, logvar = vae(batch)
            loss = rmsd_loss(reconstructed, batch, dataset.atoms_per_model)
            val_total_loss += loss.item()

    # Calculate average validation loss
    avg_val_loss = val_total_loss / len(val_loader)

    # Set model back to training mode
    vae.train()

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train RMSD: {avg_loss:.6f} | Val RMSD: {avg_val_loss:.6f}")

    # Save the trained model periodically
    if (epoch + 1) % 25 == 0:
        torch.save(vae.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")

# Save final model
torch.save(vae.state_dict(), MODEL_SAVE_PATH)
print(f"Final model saved to {MODEL_SAVE_PATH}")


