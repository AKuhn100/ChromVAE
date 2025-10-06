#!/usr/bin/env python3

"""
Generate an MP4 video of contact maps from a multi-model PDB file.

This script streams the input PDB file model-by-model (MODEL ... ENDMDL), computes a
contact map for each model from the atom coordinates, and writes the sequence of contact
maps into a single MP4 video.

Usage example:
  python Generate_ContactMap_Video.py \
    --pdb /Users/amk19/Desktop/ChromatinVAE/Data/chromosome21_aligned.pdb \
    --out /Users/amk19/Desktop/ChromatinVAE/Chrom_21_ContactMaps.mp4 \
    --cutoff 300.0 \
    --mode binary \
    --fps 10 \
    --size 768

Notes:
  - Distances are in the same units as the PDB coordinates (often Angstrom for MD/CG models).
  - The script uses all ATOM/HETATM records inside each MODEL block. If your file encodes
    beads/particles per locus, that should correspond one-to-one to matrix indices.
  - To speed up rendering, we avoid per-frame Matplotlib plotting; instead we normalize the
    contact map and map it to colors (colormap) as an image array and write to the video.
"""

import argparse
import os
import sys
from typing import Generator, List, Optional, Tuple

import numpy as np

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - fallback
    import imageio  # type: ignore

try:
    from matplotlib import cm
except ImportError as _e:
    cm = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate contact map video from multi-model PDB")
    parser.add_argument("--pdb", required=True, help="Path to input multi-model PDB file")
    parser.add_argument("--out", required=True, help="Path to output MP4 file")
    parser.add_argument("--cutoff", type=float, default=300.0, help="Distance cutoff for 'binary' mode")
    parser.add_argument("--mode", choices=["binary", "inverse", "gaussian"], default="binary",
                        help="Contact map mode: 'binary' (d<=cutoff), 'inverse' (1/d), 'gaussian' (exp(-(d/sigma)^2))")
    parser.add_argument("--sigma", type=float, default=200.0, help="Sigma for 'gaussian' mode (same units as coords)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for output video")
    parser.add_argument("--size", type=int, default=768, help="Square pixel size of output frames (e.g., 512, 768, 1024)")
    parser.add_argument("--colormap", type=str, default="viridis", help="Matplotlib colormap name for intensity mapping")
    parser.add_argument("--vmin", type=float, default=None, help="Optional fixed vmin for intensity normalization")
    parser.add_argument("--vmax", type=float, default=None, help="Optional fixed vmax for intensity normalization")
    parser.add_argument("--max_models", type=int, default=None, help="Optional cap on number of models to process")
    parser.add_argument("--frame_stride", type=int, default=1, help="Write every Nth model as a frame (subsample)")
    parser.add_argument("--float32", action="store_true", help="Use float32 for computations to reduce memory")
    parser.add_argument("--symmetric_only", action="store_true", help="Compute upper triangle and mirror for speed")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    return parser.parse_args()


def stream_models_from_pdb(pdb_path: str) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Stream models from a PDB file. Yields (model_index, coords) where coords is (N,3) float array.
    Assumes MODEL ... ENDMDL blocks. All ATOM/HETATM lines inside a model are used.
    """
    model_idx = 0
    in_model = False
    coords: List[Tuple[float, float, float]] = []

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = line[:6].strip().upper()
            if rec == "MODEL":
                # If nested MODEL found without ENDMDL, flush previous if any
                if in_model and coords:
                    arr = np.asarray(coords, dtype=np.float64)
                    yield (model_idx, arr)
                    coords = []
                    model_idx += 1
                in_model = True
                continue
            if rec in ("ATOM", "HETATM") and in_model:
                try:
                    # PDB columns: x[30:38], y[38:46], z[46:54] (1-based: 31-38, 39-46, 47-54)
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x, y, z))
                except Exception:
                    # Skip malformed lines
                    continue
            elif rec == "ENDMDL" and in_model:
                arr = np.asarray(coords, dtype=np.float64)
                yield (model_idx, arr)
                coords = []
                model_idx += 1
                in_model = False

    # Handle files that end without ENDMDL
    if in_model and coords:
        arr = np.asarray(coords, dtype=np.float64)
        yield (model_idx, arr)


def compute_contact_map(coords: np.ndarray,
                        mode: str,
                        cutoff: float,
                        sigma: float,
                        use_float32: bool,
                        symmetric_only: bool) -> np.ndarray:
    """
    Compute a contact/intensity matrix from Nx3 coordinates.
    - binary: 1 if dist <= cutoff else 0
    - inverse: value = 1 / (dist + eps)
    - gaussian: value = exp(- (dist/sigma)^2)
    Returns an NxN float array.
    """
    dtype = np.float32 if use_float32 else np.float64
    xyz = coords.astype(dtype, copy=False)
    n = xyz.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=dtype)

    # Compute pairwise distances efficiently
    # dist(i,j) = ||xi - xj||
    # Use (x^2 + y^2 + z^2) trick: ||a-b||^2 = |a|^2 + |b|^2 - 2 a.b
    # For memory balance, handle full matrix; if very large, consider block computation.
    dot = xyz @ xyz.T  # (n,n)
    sq_norms = np.sum(xyz * xyz, axis=1, dtype=dtype)
    dist2 = np.maximum(sq_norms[:, None] + sq_norms[None, :] - 2.0 * dot, 0.0)

    if mode == "binary":
        contacts = (dist2 <= cutoff * cutoff).astype(dtype, copy=False)
        # Zero diagonal
        np.fill_diagonal(contacts, 0.0)
        return contacts

    # For non-binary modes we need distances
    dist = np.sqrt(dist2, dtype=dtype)
    np.fill_diagonal(dist, 0.0)

    if mode == "inverse":
        eps = np.finfo(dtype).eps
        inv = 1.0 / (dist + eps)
        inv[dist == 0.0] = 0.0
        return inv
    elif mode == "gaussian":
        if sigma <= 0:
            raise ValueError("Sigma must be positive for gaussian mode")
        val = np.exp(- (dist / dtype(sigma)) ** 2)
        np.fill_diagonal(val, 0.0)
        return val
    else:
        raise ValueError(f"Unknown mode: {mode}")


def normalize_to_uint8(matrix: np.ndarray,
                       vmin: Optional[float],
                       vmax: Optional[float],
                       colormap_name: Optional[str],
                       out_size: int) -> np.ndarray:
    """
    Normalize a matrix to an RGB image (H,W,3) uint8 array.
    If colormap_name is provided and matplotlib is available, apply the colormap; otherwise grayscale.
    The matrix is resized to (out_size, out_size) using simple nearest-neighbor scaling via numpy repeat.
    """
    mat = matrix
    if mat.size == 0:
        img = np.zeros((out_size, out_size, 3), dtype=np.uint8)
        return img

    # Determine vmin/vmax
    if vmin is None:
        vmin = float(np.nanmin(mat))
    if vmax is None:
        vmax = float(np.nanmax(mat))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    # Clip and scale to [0,1]
    mat_clipped = np.clip(mat, vmin, vmax)
    norm = (mat_clipped - vmin) / (vmax - vmin)

    # Apply colormap if available
    if colormap_name is not None and cm is not None:
        cmap = cm.get_cmap(colormap_name)
        rgba = cmap(norm)  # (N,N,4) float in [0,1]
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    else:
        # Grayscale
        rgb = (norm * 255.0).astype(np.uint8)
        rgb = np.stack([rgb, rgb, rgb], axis=-1)

    # Resize to (out_size, out_size) via integer scaling
    h, w, _ = rgb.shape
    if h != out_size or w != out_size:
        # Compute scale factors as integers if possible, else approximate
        scale_h = max(1, int(round(out_size / h)))
        scale_w = max(1, int(round(out_size / w)))
        img = np.repeat(np.repeat(rgb, scale_h, axis=0), scale_w, axis=1)
        img = img[:out_size, :out_size, :]
    else:
        img = rgb
    return img


def main() -> None:
    args = parse_args()

    pdb_path = args.pdb
    out_path = args.out
    if not os.path.isfile(pdb_path):
        print(f"Error: PDB not found: {pdb_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    total_models = 0
    expected_n: Optional[int] = None

    writer = imageio.get_writer(out_path, fps=args.fps, codec="libx264", quality=8)
    try:
        for model_idx, coords in stream_models_from_pdb(pdb_path):
            if args.max_models is not None and total_models >= args.max_models:
                break

            if expected_n is None:
                expected_n = coords.shape[0]
                if not args.quiet:
                    print(f"Detected {expected_n} particles per model (from model {model_idx}).")
            else:
                if coords.shape[0] != expected_n and not args.quiet:
                    print(f"Warning: model {model_idx} has {coords.shape[0]} particles (expected {expected_n}).", file=sys.stderr)

            # Frame sub-sampling
            if (model_idx % args.frame_stride) != 0:
                total_models += 1
                continue

            contact = compute_contact_map(
                coords=coords,
                mode=args.mode,
                cutoff=args.cutoff,
                sigma=args.sigma,
                use_float32=args.float32,
                symmetric_only=args.symmetric_only,
            )

            frame = normalize_to_uint8(
                matrix=contact,
                vmin=args.vmin,
                vmax=args.vmax,
                colormap_name=args.colormap,
                out_size=args.size,
            )
            writer.append_data(frame)

            total_models += 1
            if not args.quiet and (total_models % 10 == 0):
                print(f"Processed {total_models} models...")
    finally:
        writer.close()

    if not args.quiet:
        print(f"Done. Wrote {total_models} frame(s) to {out_path}")


if __name__ == "__main__":
    main()


