import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from VAE.Stacked_VAE import LGL_VAE
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
MODEL_SAVE_PATH = "./outputs/trained_vae_model_stacked.pt"

# Model hyperparameters
HIDDEN_DIM = 4096
LATENT_DIM = 2

# Training hyperparameters
BATCH_SIZE = 128  # Increased for more stable gradients
LEARNING_RATE = 1e-4  # Will be warmed up from 1e-6
NUM_EPOCHS = 1000
BETA_START = 0.0001  # Start very low to prevent posterior collapse
BETA_END = 0.01  # Gradually increase to encourage latent usage
L2_REGULARIZATION = 0.00001  # Reduced regularization
FREE_BITS = 0.5  # Minimum KL per dimension

# Initialize Utils
utils = Utils()
print(f"Using device: {utils.device}")


def compute_pairwise_distances(flattened_coords, num_atoms):
    """
    Compute pairwise distances from flattened coordinate vectors.
    
    Args:
        flattened_coords: [batch_size, 3 * num_atoms] tensor of flattened coordinates
        num_atoms: number of atoms per structure
    
    Returns:
        pairwise_distances: [batch_size, num_atoms * (num_atoms - 1) // 2] tensor of pairwise distances
    """
    batch_size = flattened_coords.shape[0]
    
    # Reshape to [batch_size, num_atoms, 3]
    coords = flattened_coords.view(batch_size, num_atoms, 3)
    
    # Compute pairwise distances using broadcasting
    # coords: [batch_size, num_atoms, 3]
    # coords.unsqueeze(2): [batch_size, num_atoms, 1, 3]
    # coords.unsqueeze(1): [batch_size, 1, num_atoms, 3]
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [batch_size, num_atoms, num_atoms, 3]
    distances = torch.norm(diff, dim=3)  # [batch_size, num_atoms, num_atoms]
    
    # Extract upper triangular part (excluding diagonal) to get unique pairwise distances
    # Create mask for upper triangular matrix (excluding diagonal)
    mask = torch.triu(torch.ones(num_atoms, num_atoms, device=flattened_coords.device), diagonal=1).bool()
    
    # Apply mask to each batch item
    pairwise_distances = []
    for i in range(batch_size):
        batch_distances = distances[i][mask]  # [num_pairs]
        pairwise_distances.append(batch_distances)
    
    pairwise_distances = torch.stack(pairwise_distances, dim=0)  # [batch_size, num_pairs]
    
    return pairwise_distances


def get_beta(epoch, max_epochs, mode='monotonic'):
    """Beta annealing schedule to prevent posterior collapse."""
    if mode == 'monotonic':
        # Linear increase from BETA_START to BETA_END
        progress = epoch / max_epochs
        return BETA_START + (BETA_END - BETA_START) * progress
    elif mode == 'cyclical':
        # Cyclical annealing - can help escape local minima
        cycle_length = max_epochs // 4
        cycle_progress = (epoch % cycle_length) / cycle_length
        return BETA_START + (BETA_END - BETA_START) * cycle_progress
    else:
        return BETA_START


def compute_weighted_distance_loss(recon_distances, target_distances, num_atoms):
    """Compute distance loss with higher weight for local structure."""
    # Create distance-based weights (closer distances get higher weight)
    # This helps preserve local chromatin folding patterns
    distance_weights = torch.exp(-target_distances / 10.0)  # Exponential decay
    distance_weights = distance_weights / distance_weights.mean()  # Normalize
    
    # Weighted MSE loss
    weighted_loss = distance_weights * (recon_distances - target_distances) ** 2
    return weighted_loss.mean()


def compute_contact_map_loss(recon_distances, target_distances, contact_threshold=8.0):
    """Compute binary contact map preservation loss."""
    # Create binary contact maps
    recon_contacts = (recon_distances < contact_threshold).float()
    target_contacts = (target_distances < contact_threshold).float()
    
    # Binary cross-entropy loss for contact preservation
    contact_loss = nn.BCELoss()
    return contact_loss(recon_contacts, target_contacts)


def vae_loss(reconstructed, target, mu, logvar, num_atoms, beta, epoch, max_epochs):
    """VAE loss using pairwise distances with beta annealing, free bits, and auxiliary losses."""
    # Compute pairwise distances for both reconstructed and target
    recon_distances = compute_pairwise_distances(reconstructed, num_atoms)
    target_distances = compute_pairwise_distances(target, num_atoms)
    
    # Main reconstruction loss (MSE on pairwise distances)
    reconstruction_loss = nn.MSELoss()
    recon_loss = reconstruction_loss(recon_distances, target_distances)
    
    # Auxiliary losses for better structure preservation
    # 1. Weighted distance loss (emphasizes local structure)
    local_loss = compute_weighted_distance_loss(recon_distances, target_distances, num_atoms)
    
    # 2. Contact map preservation loss
    contact_loss = compute_contact_map_loss(recon_distances, target_distances)
    
    # Combined reconstruction loss
    total_recon_loss = recon_loss + 0.1 * local_loss + 0.05 * contact_loss
    
    # KL divergence loss with numerical stability
    # Clamp logvar to prevent numerical issues
    logvar = torch.clamp(logvar, min=-10, max=10)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.mean(kl_loss)
    
    # Free bits: prevent complete posterior collapse
    # Allow each latent dimension to have minimum KL divergence
    kl_loss = torch.max(kl_loss, torch.tensor(FREE_BITS, device=kl_loss.device))
    
    # Get current beta value
    current_beta = get_beta(epoch, max_epochs)
    
    # Total loss
    total_loss = total_recon_loss + current_beta * kl_loss
    
    return total_loss, recon_loss, kl_loss, current_beta, local_loss, contact_loss

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

# Initialize optimizer with warmup
optimizer = optim.Adam(vae.parameters(), lr=1e-6, weight_decay=L2_REGULARIZATION)  # Start low for warmup
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# Training loop
vae.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_local_loss = 0.0
    total_contact_loss = 0.0
    
    # Track latent space statistics
    latent_mu_stats = []
    latent_logvar_stats = []
    latent_z_stats = []
    
    # Training phase with loss tracking in progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = batch.to(utils.device)
        
        # Forward pass
        optimizer.zero_grad()
        reconstructed, mu, logvar, z = vae(batch)
        
        # Calculate loss with beta annealing and auxiliary losses
        loss, recon_loss, kl_loss, current_beta, local_loss, contact_loss = vae_loss(reconstructed, batch, mu, logvar, dataset.atoms_per_model, None, epoch, NUM_EPOCHS)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_local_loss += local_loss.item()
        total_contact_loss += contact_loss.item()
        
        # Track latent space statistics
        latent_mu_stats.append(mu.detach().cpu())
        latent_logvar_stats.append(logvar.detach().cpu())
        latent_z_stats.append(z.detach().cpu())
        
        # Learning rate warmup for first 10 epochs
        if epoch < 10:
            warmup_factor = (epoch + 1) / 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-6 + (LEARNING_RATE - 1e-6) * warmup_factor
        
    
    # Calculate average losses for this epoch
    avg_loss = total_loss / len(train_loader)
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_kl_loss = total_kl_loss / len(train_loader)
    avg_local_loss = total_local_loss / len(train_loader)
    avg_contact_loss = total_contact_loss / len(train_loader)
    
    # Calculate latent space statistics
    all_mu = torch.cat(latent_mu_stats, dim=0)
    all_logvar = torch.cat(latent_logvar_stats, dim=0)
    all_z = torch.cat(latent_z_stats, dim=0)
    
    # Per-dimension statistics
    mu_std = all_mu.std(dim=0)  # Standard deviation of mu across batch
    logvar_mean = all_logvar.mean(dim=0)  # Average logvar per dimension
    z_std = all_z.std(dim=0)  # Standard deviation of sampled z
    
    # KL per dimension
    kl_per_dim = -0.5 * (1 + logvar_mean - all_mu.mean(dim=0).pow(2) - logvar_mean.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=0)  # Ensure non-negative
    
    # Validation phase
    vae.eval()
    val_total_loss = 0.0
    val_total_recon_loss = 0.0
    val_total_kl_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            batch = batch.to(utils.device)
            reconstructed, mu, logvar, z = vae(batch)
            loss, recon_loss, kl_loss, _, local_loss, contact_loss = vae_loss(reconstructed, batch, mu, logvar, dataset.atoms_per_model, None, epoch, NUM_EPOCHS)
            
            val_total_loss += loss.item()
            val_total_recon_loss += recon_loss.item()
            val_total_kl_loss += kl_loss.item()
    
    # Calculate average validation losses
    avg_val_loss = val_total_loss / len(val_loader)
    avg_val_recon_loss = val_total_recon_loss / len(val_loader)
    avg_val_kl_loss = val_total_kl_loss / len(val_loader)
    
    # Set model back to training mode
    vae.train()
    
    # Update learning rate scheduler (after warmup)
    if epoch >= 10:
        scheduler.step()
    
    current_lr = optimizer.param_groups[0]['lr']
    current_beta = get_beta(epoch, NUM_EPOCHS)
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}, Local: {avg_local_loss:.4f}, Contact: {avg_contact_loss:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}, Recon: {avg_val_recon_loss:.4f}, KL: {avg_val_kl_loss:.4f} | LR: {current_lr:.2e}, Beta: {current_beta:.4f}")
    print(f"  Latent Stats - Mu std: {mu_std.mean().item():.4f}, Logvar mean: {logvar_mean.mean().item():.4f}, Z std: {z_std.mean().item():.4f}")
    print(f"  KL per dim: {kl_per_dim.tolist()}")

    # Save the trained model every 100 epochs
    if (epoch + 1) % 100 == 0:
        torch.save(vae.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")

# Save final model
torch.save(vae.state_dict(), MODEL_SAVE_PATH)
print(f"Final model saved to {MODEL_SAVE_PATH}")



