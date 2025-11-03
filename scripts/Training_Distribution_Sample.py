#!/usr/bin/env python3
"""
Script to sample the latent space using per-dimension mean and standard deviation 
from the training set, then generate PDB structures.
"""

import torch
import numpy as np
from pathlib import Path
import sys
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from VAE.Simple_VAE import LGL_VAE
from DataLoader.Chromosome21PDBDataset import Chromosome21PDBDataset
from Utils.Utils import Utils


def encode_training_set(model, dataset, device, batch_size=64):
    """
    Encode the entire training set into latent space to compute statistics.
    
    Args:
        model: Trained VAE model
        dataset: Training dataset
        device: Device to run inference on
        batch_size: Batch size for encoding
    
    Returns:
        latent_means: Mean for each latent dimension
        latent_stddevs: Standard deviation for each latent dimension
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latent_representations = []
    
    print("Encoding training set into latent space...")
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            # Encode to latent space using the mean (mu)
            x = batch.view(batch.shape[0], -1).float()
            hidden = model.encoder(x)
            mu = model.to_mu(hidden)
            
            latent_representations.append(mu.cpu().numpy())
    
    # Concatenate all batches
    latent_representations = np.concatenate(latent_representations, axis=0)
    
    # Compute mean and stddev for each latent dimension
    latent_means = np.mean(latent_representations, axis=0)
    latent_stddevs = np.std(latent_representations, axis=0)
    
    print(f"Encoded {len(latent_representations)} samples into {latent_representations.shape[1]}D latent space")
    print(f"Mean across dimensions: {np.mean(latent_means):.4f} ± {np.std(latent_means):.4f}")
    print(f"Stddev across dimensions: {np.mean(latent_stddevs):.4f} ± {np.std(latent_stddevs):.4f}")
    
    return latent_means, latent_stddevs


def main():
    # Configuration
    model_path = "./outputs/trained_vae_model.pt"
    dataset_path = "./Data/chromosome21_aligned.pdb"
    output_dir = "./outputs/Generated_Samples"
    num_samples = 1000
    HIDDEN_DIM = 4096
    LATENT_DIM = 64
    VARIANCE_MULTIPLIER = 1.0  # Multiplier for stddev (1.0 = use exact training stddev)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load dataset to get dimensions
    print("Loading dataset...")
    dataset = Chromosome21PDBDataset(
        pdb_path=dataset_path,
        record_types=("ATOM",),
        center=False,
        scale=1.0
    )
    input_dim = dataset.vector_length
    print(f"Input dimension: {input_dim}")
    
    # Load model
    print("Loading trained model...")
    utils = Utils()
    model = LGL_VAE(hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, input_dim=input_dim)
    model = model.to(utils.device)
    
    checkpoint = torch.load(model_path, map_location=utils.device)
    
    # Handle the case where the model was saved with DataParallel/DistributedDataParallel
    # which adds "module." prefix to all keys
    if any(key.startswith('module.') for key in checkpoint.keys()):
        # Create a new state dict with "module." prefix removed
        new_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith('module.'):
                new_key = key[7:]  # Remove 'module.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        checkpoint = new_state_dict
    
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Encode training set to get per-dimension statistics
    latent_means, latent_stddevs = encode_training_set(model, dataset, utils.device)
    
    # Generate samples using training set statistics
    print(f"\nGenerating {num_samples} samples...")
    print(f"Using training set mean and stddev for each latent dimension")
    print(f"Stddev multiplier: {VARIANCE_MULTIPLIER}")
    
    with torch.no_grad():
        # Sample from latent space using per-dimension statistics
        # z ~ N(mean_i, (stddev_i * multiplier)^2) for each dimension i
        z_samples = np.random.randn(num_samples, LATENT_DIM)
        z_samples = z_samples * (latent_stddevs * VARIANCE_MULTIPLIER) + latent_means
        
        # Convert to torch tensor
        z = torch.tensor(z_samples, dtype=torch.float32, device=utils.device)
        
        # Decode to generate structures
        samples = model.decoder(z)
    
    # Convert to PDB format - single file with multiple models
    samples_np = samples.cpu().numpy()
    
    # Create single output PDB file
    output_file = Path(output_dir) / "generated_samples_training_distribution.pdb"
    
    # Write header
    with open(output_file, 'w') as f:
        f.write("HEADER    GENERATED CHROMOSOME STRUCTURES FROM VAE LATENT SPACE\n")
        f.write("REMARK    Each MODEL represents one sampled point from the latent space\n")
        f.write("REMARK    Sampled using per-dimension mean and stddev from training set\n")
        f.write("REMARK    Generated by ChromatinVAE training distribution sampling script\n")
    
    # Add each sample as a model
    for i, sample in enumerate(samples_np):
        with open(output_file, 'a') as f:  # Append mode
            f.write(f"MODEL        {i+1:4d}\n")
            
            num_atoms = len(sample) // 3
            for j in range(num_atoms):
                x, y, z = sample[j*3:(j+1)*3]
                f.write(f"ATOM  {j+1:5d}  CA  UNK A{j+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
            
            f.write("ENDMDL\n")
        
        if (i + 1) % 100 == 0:
            print(f"Added {i+1} samples...")
    
    print(f"\nGenerated {num_samples} samples in single PDB file: {output_file}")
    print("You can view this with PyMOL, VMD, or other molecular viewers!")
    print("Each MODEL in the file represents one sampled structure from the latent space.")
    print(f"\nSampling statistics:")
    print(f"  Used per-dimension means (range: {np.min(latent_means):.4f} to {np.max(latent_means):.4f})")
    print(f"  Used per-dimension stddevs (range: {np.min(latent_stddevs):.4f} to {np.max(latent_stddevs):.4f})")


if __name__ == "__main__":
    main()
