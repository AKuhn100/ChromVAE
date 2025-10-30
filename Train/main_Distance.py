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
MODEL_SAVE_PATH = "./outputs/trained_vae_model_Short.pt"

# Model hyperparameters
HIDDEN_DIM = 4096
LATENT_DIM = 64

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-4  # Reduced learning rate for stability
NUM_EPOCHS = 100
BETA = 0.01  # Reduced KL weight for stability
L2_REGULARIZATION = 0.0001
RADIUS_OF_GYRATION_WEIGHT = 1.0  # Weight for radius of gyration loss
ANGLE_WEIGHT = 1.0  # Weight for angle loss

# Initialize Utils
utils = Utils()
print(f"Using device: {utils.device}")


def compute_pairwise_distances(flattened_coords, num_atoms):
    """
    Compute pairwise distances from flattened coordinate vectors (highly optimized version).
    
    Args:
        flattened_coords: [batch_size, 3 * num_atoms] tensor of flattened coordinates
        num_atoms: number of atoms per structure
    
    Returns:
        pairwise_distances: [batch_size, num_atoms * (num_atoms - 1) // 2] tensor of pairwise distances
    """
    batch_size = flattened_coords.shape[0]
    
    # Reshape to [batch_size, num_atoms, 3]
    coords = flattened_coords.view(batch_size, num_atoms, 3)
    
    # Most efficient: use torch.pdist for each batch item
    # This avoids computing the full distance matrix
    pairwise_distances = []
    for i in range(batch_size):
        # torch.pdist computes upper triangular distances directly
        batch_distances = torch.pdist(coords[i], p=2)  # [num_pairs]
        pairwise_distances.append(batch_distances)
    
    pairwise_distances = torch.stack(pairwise_distances, dim=0)  # [batch_size, num_pairs]
    
    return pairwise_distances


def compute_radius_of_gyration(flattened_coords, num_atoms):
    """
    Compute radius of gyration from flattened coordinate vectors (optimized version).
    
    Args:
        flattened_coords: [batch_size, 3 * num_atoms] tensor of flattened coordinates
        num_atoms: number of atoms per structure
    
    Returns:
        radius_of_gyration: [batch_size] tensor of radius of gyration values
    """
    batch_size = flattened_coords.shape[0]
    
    # Reshape to [batch_size, num_atoms, 3]
    coords = flattened_coords.view(batch_size, num_atoms, 3)
    
    # Optimized radius of gyration calculation
    # Compute center of mass
    center_of_mass = torch.mean(coords, dim=1, keepdim=True)  # [batch_size, 1, 3]
    
    # Compute squared distances from center of mass (vectorized)
    coords_centered = coords - center_of_mass  # [batch_size, num_atoms, 3]
    squared_distances = torch.sum(coords_centered ** 2, dim=2)  # [batch_size, num_atoms]
    
    # Compute radius of gyration (vectorized)
    radius_of_gyration = torch.sqrt(torch.mean(squared_distances, dim=1))  # [batch_size]
    
    return radius_of_gyration


def compute_angles(flattened_coords, num_atoms):
    """
    Compute angles between consecutive triplets of beads from flattened coordinate vectors.
    
    For beads i, i+1, i+2, computes the angle at bead i+1.
    
    Args:
        flattened_coords: [batch_size, 3 * num_atoms] tensor of flattened coordinates
        num_atoms: number of atoms per structure
    
    Returns:
        angles: [batch_size, num_atoms - 2] tensor of angles in radians
    """
    batch_size = flattened_coords.shape[0]
    
    # Reshape to [batch_size, num_atoms, 3]
    coords = flattened_coords.view(batch_size, num_atoms, 3)
    
    # For each consecutive triplet (i, i+1, i+2), compute angle at i+1
    # We need at least 3 atoms to compute angles
    if num_atoms < 3:
        # Return empty tensor if not enough atoms
        return torch.zeros(batch_size, max(0, num_atoms - 2), device=flattened_coords.device)
    
    # Get vectors for angle computation
    # v1: from bead i+1 to bead i
    # v2: from bead i+1 to bead i+2
    v1 = coords[:, :-2, :] - coords[:, 1:-1, :]  # [batch_size, num_atoms-2, 3]
    v2 = coords[:, 2:, :] - coords[:, 1:-1, :]   # [batch_size, num_atoms-2, 3]
    
    # Compute dot product
    dot_product = torch.sum(v1 * v2, dim=2)  # [batch_size, num_atoms-2]
    
    # Compute magnitudes
    v1_magnitude = torch.norm(v1, dim=2)  # [batch_size, num_atoms-2]
    v2_magnitude = torch.norm(v2, dim=2)  # [batch_size, num_atoms-2]
    
    # Avoid division by zero
    magnitude_product = v1_magnitude * v2_magnitude
    magnitude_product = torch.clamp(magnitude_product, min=1e-8)
    
    # Compute cosine of angle
    cos_angle = dot_product / magnitude_product
    
    # Clamp to avoid numerical issues with arccos
    cos_angle = torch.clamp(cos_angle, min=-1.0 + 1e-6, max=1.0 - 1e-6)
    
    # Compute angles in radians
    angles = torch.acos(cos_angle)  # [batch_size, num_atoms-2]
    
    return angles


def vae_loss(reconstructed, target, mu, logvar, num_atoms):
    """VAE loss using pairwise distances, radius of gyration, and angles."""
    # Compute pairwise distances for both reconstructed and target
    recon_distances = compute_pairwise_distances(reconstructed, num_atoms)
    target_distances = compute_pairwise_distances(target, num_atoms)
    
    # Reconstruction loss (MSE on pairwise distances)
    reconstruction_loss = nn.MSELoss()
    recon_loss = reconstruction_loss(recon_distances, target_distances)
    
    # Radius of gyration loss
    recon_rog = compute_radius_of_gyration(reconstructed, num_atoms)
    target_rog = compute_radius_of_gyration(target, num_atoms)
    rog_loss = nn.MSELoss()(recon_rog, target_rog)
    
    # Angle loss
    recon_angles = compute_angles(reconstructed, num_atoms)
    target_angles = compute_angles(target, num_atoms)
    angle_loss = nn.MSELoss()(recon_angles, target_angles)
    
    # KL divergence loss with numerical stability
    # Clamp logvar to prevent numerical issues
    logvar = torch.clamp(logvar, min=-10, max=10)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.mean(kl_loss)
    
    # Total loss
    total_loss = recon_loss + BETA * kl_loss + RADIUS_OF_GYRATION_WEIGHT * rog_loss + ANGLE_WEIGHT * angle_loss
    
    return total_loss, recon_loss, kl_loss, rog_loss, angle_loss

# Load PDB dataset
print(f"Loading PDB dataset from {PDB_PATH}")
dataset = Chromosome21PDBDataset(
    pdb_path=PDB_PATH,
    record_types=("ATOM",),
    center=True,  # Center coordinates
    scale=1.0  # Don't scale - let's see raw coordinate ranges first
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
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_rog_loss = 0.0
    total_angle_loss = 0.0
    
    # Training phase with loss tracking in progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = batch.to(utils.device)
        
        # Forward pass
        optimizer.zero_grad()
        reconstructed, mu, logvar = vae(batch)
        
        # Calculate loss
        loss, recon_loss, kl_loss, rog_loss, angle_loss = vae_loss(reconstructed, batch, mu, logvar, dataset.atoms_per_model)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_rog_loss += rog_loss.item()
        total_angle_loss += angle_loss.item()
        
    
    # Calculate average losses for this epoch
    avg_loss = total_loss / len(train_loader)
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_kl_loss = total_kl_loss / len(train_loader)
    avg_rog_loss = total_rog_loss / len(train_loader)
    avg_angle_loss = total_angle_loss / len(train_loader)
    
    # Validation phase
    vae.eval()
    val_total_loss = 0.0
    val_total_recon_loss = 0.0
    val_total_kl_loss = 0.0
    val_total_rog_loss = 0.0
    val_total_angle_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            batch = batch.to(utils.device)
            reconstructed, mu, logvar = vae(batch)
            loss, recon_loss, kl_loss, rog_loss, angle_loss = vae_loss(reconstructed, batch, mu, logvar, dataset.atoms_per_model)
            
            val_total_loss += loss.item()
            val_total_recon_loss += recon_loss.item()
            val_total_kl_loss += kl_loss.item()
            val_total_rog_loss += rog_loss.item()
            val_total_angle_loss += angle_loss.item()
    
    # Calculate average validation losses
    avg_val_loss = val_total_loss / len(val_loader)
    avg_val_recon_loss = val_total_recon_loss / len(val_loader)
    avg_val_kl_loss = val_total_kl_loss / len(val_loader)
    avg_val_rog_loss = val_total_rog_loss / len(val_loader)
    avg_val_angle_loss = val_total_angle_loss / len(val_loader)
    
    # Set model back to training mode
    vae.train()
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}, RoG: {avg_rog_loss:.4f}, Angle: {avg_angle_loss:.4f} | Val Loss: {avg_val_loss:.4f}, Recon: {avg_val_recon_loss:.4f}, KL: {avg_val_kl_loss:.4f}, RoG: {avg_val_rog_loss:.4f}, Angle: {avg_val_angle_loss:.4f}")

    # Save the trained model every 100 epochs
    if (epoch + 1) % 25 == 0:
        torch.save(vae.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")

# Save final model
torch.save(vae.state_dict(), MODEL_SAVE_PATH)
print(f"Final model saved to {MODEL_SAVE_PATH}")



