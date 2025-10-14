#!/usr/bin/env python3

"""
Simplified aggregate contact map generator.

This script generates two specific outputs from a multi-model PDB:
1. Inverse distance map - 1/average pairwise distances across all models
2. Inverse standard deviation map - 1/stddev of pairwise distance distributions

Static configuration - modify paths directly in the script.
"""

import os
import sys
from typing import Generator, List, Optional, Tuple

import numpy as np

try:
    from matplotlib import cm
    import matplotlib.pyplot as plt
except ImportError:
    cm = None  # type: ignore
    plt = None  # type: ignore

# Static configuration - modify these paths as needed
PDB_PATH = "/scratch/amk19/ChromVAE/ChromVAE/outputs/Generated_Samples/generated_samples.pdb"
INVERSE_DISTANCE_OUTPUT = "/scratch/amk19/ChromVAE/ChromVAE/outputs/Generated_Samples/inverse_distance_map.png"
INVERSE_STDDEV_OUTPUT = "/scratch/amk19/ChromVAE/ChromVAE/outputs/Generated_Samples/inverse_stddev_map.png"


def stream_models_from_pdb(pdb_path: str) -> Generator[Tuple[int, np.ndarray], None, None]:
    model_idx = 0
    in_model = False
    coords: List[Tuple[float, float, float]] = []

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = line[:6].strip().upper()
            if rec == "MODEL":
                if in_model and coords:
                    arr = np.asarray(coords, dtype=np.float64)
                    yield (model_idx, arr)
                    coords = []
                    model_idx += 1
                in_model = True
                continue
            if rec in ("ATOM", "HETATM") and in_model:
                try:
                    x = float(line[31:39])
                    y = float(line[39:47])
                    z = float(line[47:55])
                    coords.append((x, y, z))
                except Exception:
                    continue
            elif rec == "ENDMDL" and in_model:
                arr = np.asarray(coords, dtype=np.float64)
                yield (model_idx, arr)
                coords = []
                model_idx += 1
                in_model = False

    if in_model and coords:
        arr = np.asarray(coords, dtype=np.float64)
        yield (model_idx, arr)


def compute_pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise distances between all coordinates."""
    dtype = np.float64
    xyz = coords.astype(dtype, copy=False)
    n = xyz.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=dtype)

    dot = xyz @ xyz.T
    sq_norms = np.sum(xyz * xyz, axis=1, dtype=dtype)
    dist2 = np.maximum(sq_norms[:, None] + sq_norms[None, :] - 2.0 * dot, 0.0)
    dist = np.sqrt(dist2, dtype=dtype)
    np.fill_diagonal(dist, 0.0)
    return dist


def save_heatmap(matrix: np.ndarray, out_img: str, title: str, vmin: float = 0.0, vmax: float = 1.0) -> None:
    """Save a heatmap image with the given matrix and title."""
    if plt is None:
        raise RuntimeError("matplotlib is required to save the heatmap image. Install via `pip install matplotlib`.")
    
    plt.figure(figsize=(8, 8), dpi=200)
    plt.imshow(matrix, cmap="magma", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("Bead Index")
    plt.ylabel("Bead Index")
    plt.tight_layout()
    plt.savefig(out_img, dpi=200)
    plt.close()


def main() -> None:
    """Generate average distance map and inverse standard deviation map."""
    pdb_path = PDB_PATH
    if not os.path.isfile(pdb_path):
        print(f"Error: PDB not found: {pdb_path}", file=sys.stderr)
        sys.exit(1)

    # Create output directories
    os.makedirs(os.path.dirname(INVERSE_DISTANCE_OUTPUT) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(INVERSE_STDDEV_OUTPUT) or ".", exist_ok=True)

    # Online variance calculation using Welford's algorithm
    mean_dist: Optional[np.ndarray] = None
    m2_dist: Optional[np.ndarray] = None
    model_count = 0
    expected_n: Optional[int] = None

    print("Processing models and computing pairwise distances...")
    
    for model_idx, coords in stream_models_from_pdb(pdb_path):
        if expected_n is None:
            expected_n = coords.shape[0]
            print(f"Detected {expected_n} particles per model (from model {model_idx}).")
        elif coords.shape[0] != expected_n:
            print(f"Warning: model {model_idx} has {coords.shape[0]} particles (expected {expected_n}).", file=sys.stderr)

        # Compute pairwise distances for this model
        dist = compute_pairwise_distances(coords)

        if mean_dist is None:
            mean_dist = dist.astype(np.float64, copy=True)
            m2_dist = np.zeros_like(mean_dist, dtype=np.float64)
            model_count = 1
        else:
            model_count += 1
            delta = dist - mean_dist
            mean_dist += delta / model_count
            m2_dist += delta * (dist - mean_dist)

        if model_count % 10 == 0:
            print(f"Processed {model_count} models...")

    if mean_dist is None or m2_dist is None:
        print("No models were processed; nothing to aggregate.", file=sys.stderr)
        sys.exit(2)

    print(f"Processed {model_count} models total.")

    # Calculate standard deviation
    if model_count < 2:
        std = np.zeros_like(mean_dist)
    else:
        var = m2_dist / (model_count - 1)
        var[var < 0.0] = 0.0
        std = np.sqrt(var)

    # Generate outputs
    print("Generating inverse distance map...")
    eps = 1e-6  # Small epsilon to avoid division by zero
    inv_dist = 1.0 / (mean_dist + eps)
    np.fill_diagonal(inv_dist, 0.0)  # Set diagonal to 0
    
    # Apply maximum threshold of 1
    inv_dist = np.minimum(inv_dist, 1.0)
    
    save_heatmap(inv_dist, INVERSE_DISTANCE_OUTPUT, "Inverse Average Distance Map", vmin=0.0, vmax=1.0)
    print(f"Saved inverse distance map to {INVERSE_DISTANCE_OUTPUT}")

    print("Generating inverse standard deviation map...")
    eps = 1e-6  # Small epsilon to avoid division by zero
    inv_std = 1.0 / (std + eps)
    np.fill_diagonal(inv_std, 0.0)  # Set diagonal to 0
    
    # Apply maximum threshold of 1
    inv_std = np.minimum(inv_std, 1.0)
    
    save_heatmap(inv_std, INVERSE_STDDEV_OUTPUT, "Inverse Standard Deviation Map", vmin=0.0, vmax=1.0)
    print(f"Saved inverse standard deviation map to {INVERSE_STDDEV_OUTPUT}")

    print("Done!")


if __name__ == "__main__":
    main()