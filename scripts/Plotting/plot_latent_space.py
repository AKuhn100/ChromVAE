#!/usr/bin/env python3
"""
Script to load the trained VAE model and encode the training set into latent space,
then plot the latent representations as a scatter plot.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from torch.utils.data import DataLoader, random_split

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from VAE.Simple_VAE import LGL_VAE
from DataLoader.Chromosome21PDBDataset import Chromosome21PDBDataset
from Utils.Utils import Utils


def encode_dataset(model, dataloader, device):
    """
    Encode the entire dataset into latent space using the VAE encoder.
    
    Args:
        model: Trained VAE model
        dataloader: DataLoader containing the dataset
        device: Device to run inference on
    
    Returns:
        numpy array of shape (n_samples, latent_dim) containing latent representations
    """
    model.eval()
    latent_representations = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = batch.to(device)
            
            # Get latent representation (mu) from encoder
            # We use the mean of the latent distribution, not the sampled z
            x = batch.view(batch.shape[0], -1).float()
            hidden = model.encoder(x)
            mu = model.to_mu(hidden)
            
            # Store latent representations
            latent_representations.append(mu.cpu().numpy())
    
    # Concatenate all batches
    return np.concatenate(latent_representations, axis=0)


def main():
    # Configuration
    model_path = "/scratch/amk19/ChromVAE/ChromVAE/outputs/trained_vae_model_XL.pt"
    dataset_path = "/scratch/amk19/ChromVAE/ChromVAE/Data/chromosome21_aligned.pdb"
    output_dir = "/scratch/amk19/ChromVAE/ChromVAE/outputs/Generated_Samples_XL"
    
    # Model hyperparameters (must match training configuration)
    HIDDEN_DIM = 4096
    LATENT_DIM = 2
    BATCH_SIZE = 16
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize Utils
    utils = Utils()
    print(f"Using device: {utils.device}")
    
    # Load dataset with same preprocessing as training
    print("Loading PDB dataset...")
    dataset = Chromosome21PDBDataset(
        pdb_path=dataset_path,
        record_types=("ATOM",),
        center=True,  # Center coordinates (same as training)
        scale=1.0     # Don't scale (same as training)
    )
    
    print(f"Dataset loaded: {len(dataset)} models, {dataset.vector_length} coordinates per model")
    
    # Create model with same architecture as training
    print("Creating model...")
    model = LGL_VAE(hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, input_dim=dataset.vector_length)
    model = model.to(utils.device)
    
    # Load trained model weights
    print(f"Loading trained model from {model_path}...")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=utils.device)
    model.load_state_dict(checkpoint)
    print("Model loaded successfully!")
    
    # Split dataset into train/validation (same split as training)
    print("Splitting dataset...")
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    
    torch.manual_seed(0)  # Same seed as training
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,  # Don't shuffle for consistent ordering
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
    
    # Encode training data into latent space
    print("Encoding training data into latent space...")
    train_latent = encode_dataset(model, train_loader, utils.device)
    print(f"Training latent representations shape: {train_latent.shape}")
    
    # Encode validation data into latent space
    print("Encoding validation data into latent space...")
    val_latent = encode_dataset(model, val_loader, utils.device)
    print(f"Validation latent representations shape: {val_latent.shape}")
    
    # Create scatter plot
    print("Creating scatter plot...")
    plt.figure(figsize=(12, 8))
    
    # Plot training data
    plt.scatter(train_latent[:, 0], train_latent[:, 1], 
                alpha=0.6, s=20, label=f'Training ({len(train_latent)} samples)', 
                color='blue', edgecolors='none')
    
    # Plot validation data
    plt.scatter(val_latent[:, 0], val_latent[:, 1], 
                alpha=0.6, s=20, label=f'Validation ({len(val_latent)} samples)', 
                color='red', edgecolors='none')
    
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('VAE Latent Space Visualization\nTraining vs Validation Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add some statistics
    train_std = np.std(train_latent, axis=0)
    val_std = np.std(val_latent, axis=0)
    train_mean = np.mean(train_latent, axis=0)
    val_mean = np.mean(val_latent, axis=0)
    
    plt.text(0.02, 0.98, f'Train mean: ({train_mean[0]:.3f}, {train_mean[1]:.3f})\n'
                         f'Train std: ({train_std[0]:.3f}, {train_std[1]:.3f})\n'
                         f'Val mean: ({val_mean[0]:.3f}, {val_mean[1]:.3f})\n'
                         f'Val std: ({val_std[0]:.3f}, {val_std[1]:.3f})',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save the plot
    output_path = Path(output_dir) / "latent_space_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to: {output_path}")
    
    # Also save a version with just training data
    plt.figure(figsize=(10, 8))
    plt.scatter(train_latent[:, 0], train_latent[:, 1], 
                alpha=0.6, s=20, color='blue', edgecolors='none')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('VAE Latent Space Visualization\nTraining Data Only')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    plt.text(0.02, 0.98, f'Mean: ({train_mean[0]:.3f}, {train_mean[1]:.3f})\n'
                         f'Std: ({train_std[0]:.3f}, {train_std[1]:.3f})\n'
                         f'Samples: {len(train_latent)}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    train_only_path = Path(output_dir) / "latent_space_training_only.png"
    plt.savefig(train_only_path, dpi=300, bbox_inches='tight')
    print(f"Training-only scatter plot saved to: {train_only_path}")
    
    # Print some statistics
    print("\nLatent Space Statistics:")
    print(f"Training data range - X: [{train_latent[:, 0].min():.3f}, {train_latent[:, 0].max():.3f}]")
    print(f"Training data range - Y: [{train_latent[:, 1].min():.3f}, {train_latent[:, 1].max():.3f}]")
    print(f"Validation data range - X: [{val_latent[:, 0].min():.3f}, {val_latent[:, 0].max():.3f}]")
    print(f"Validation data range - Y: [{val_latent[:, 1].min():.3f}, {val_latent[:, 1].max():.3f}]")
    
    plt.show()


if __name__ == "__main__":
    main()
