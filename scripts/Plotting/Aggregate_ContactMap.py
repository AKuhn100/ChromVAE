#!/usr/bin/env python3

"""
Aggregate per-model contact maps from a multi-model PDB into a single heatmap.

This script streams a multi-model PDB (MODEL/ENDMDL), computes a contact map per model,
and aggregates them using mean, sum, or median. The result is saved as an image and,
optionally, a matrix file (.npy or .npz).

Example:
  python Aggregate_ContactMap.py \
    --pdb /Users/amk19/Desktop/ChromatinVAE/Data/chromosome21_aligned.pdb \
    --out_img /Users/amk19/Desktop/ChromatinVAE/Chrom_21_ContactMap_mean.png \
    --aggregate mean \
    --mode binary --cutoff 300

"""

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-model contact maps into one heatmap")
    parser.add_argument("--pdb", required=True, help="Path to input multi-model PDB file")
    parser.add_argument("--out_img", required=True, help="Path to output heatmap image (e.g., .png)")
    parser.add_argument("--out_matrix", default=None, help="Optional path to save matrix (.npy or .npz)")
    parser.add_argument("--aggregate", choices=["mean", "sum", "median", "invstd"], default="mean",
                        help="Aggregation across models; 'invstd' computes 1/std of pairwise distances")
    parser.add_argument("--mode", choices=["binary", "inverse", "gaussian"], default="binary",
                        help="Contact map mode")
    parser.add_argument("--cutoff", type=float, default=300.0, help="Cutoff for binary mode")
    parser.add_argument("--sigma", type=float, default=200.0, help="Sigma for gaussian mode")
    parser.add_argument("--float32", action="store_true", help="Use float32 for computations")
    parser.add_argument("--max_models", type=int, default=None, help="Optional cap on number of models")
    parser.add_argument("--colormap", type=str, default="viridis", help="Matplotlib colormap for image")
    parser.add_argument("--dpi", type=int, default=200, help="Image DPI")
    parser.add_argument("--vmin", type=float, default=None, help="Fixed vmin for rendering")
    parser.add_argument("--vmax", type=float, default=None, help="Fixed vmax for rendering")
    parser.add_argument("--eps", type=float, default=1e-6, help="Small epsilon to avoid division by zero in invstd")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    return parser.parse_args()


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


def compute_contact_map(coords: np.ndarray,
                        mode: str,
                        cutoff: float,
                        sigma: float,
                        use_float32: bool) -> np.ndarray:
    dtype = np.float32 if use_float32 else np.float64
    xyz = coords.astype(dtype, copy=False)
    n = xyz.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=dtype)

    dot = xyz @ xyz.T
    sq_norms = np.sum(xyz * xyz, axis=1, dtype=dtype)
    dist2 = np.maximum(sq_norms[:, None] + sq_norms[None, :] - 2.0 * dot, 0.0)

    if mode == "binary":
        contacts = (dist2 <= cutoff * cutoff).astype(dtype, copy=False)
        np.fill_diagonal(contacts, 0.0)
        return contacts

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


def aggregate_matrices(mats: List[np.ndarray], how: str) -> np.ndarray:
    if not mats:
        return np.zeros((0, 0), dtype=np.float64)
    # Ensure consistent shapes
    shapes = {m.shape for m in mats}
    if len(shapes) != 1:
        raise ValueError(f"Matrices have inconsistent shapes: {shapes}")
    stack = np.stack(mats, axis=0)
    if how == "mean":
        return stack.mean(axis=0)
    if how == "sum":
        return stack.sum(axis=0)
    if how == "median":
        return np.median(stack, axis=0)
    raise ValueError(f"Unknown aggregate: {how}")


def save_image(matrix: np.ndarray, out_img: str, colormap: str, dpi: int,
               vmin: Optional[float], vmax: Optional[float]) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to save the heatmap image. Install via `pip install matplotlib`." )
    plt.figure(figsize=(6, 6), dpi=dpi)
    plt.imshow(matrix, cmap=colormap, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_img, dpi=dpi)
    plt.close()




def main() -> None:
    args = parse_args()

    pdb_path = args.pdb
    if not os.path.isfile(pdb_path):
        print(f"Error: PDB not found: {pdb_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out_img) or ".", exist_ok=True)
    if args.out_matrix is not None:
        os.makedirs(os.path.dirname(args.out_matrix) or ".", exist_ok=True)

    matrices: List[np.ndarray] = []
    model_count = 0
    expected_n: Optional[int] = None

    if args.aggregate == "invstd":
        # Online variance for distance matrices using Welford's algorithm
        mean_dist: Optional[np.ndarray] = None
        m2_dist: Optional[np.ndarray] = None
        for model_idx, coords in stream_models_from_pdb(pdb_path):
            if args.max_models is not None and model_count >= args.max_models:
                break

            if expected_n is None:
                expected_n = coords.shape[0]
                if not args.quiet:
                    print(f"Detected {expected_n} particles per model (from model {model_idx}).")
            elif coords.shape[0] != expected_n:
                if not args.quiet:
                    print(f"Warning: model {model_idx} has {coords.shape[0]} particles (expected {expected_n}).", file=sys.stderr)

            # Compute pairwise distances for this model
            dtype = np.float32 if args.float32 else np.float64
            xyz = coords.astype(dtype, copy=False)
            dot = xyz @ xyz.T
            sq_norms = np.sum(xyz * xyz, axis=1, dtype=dtype)
            dist2 = np.maximum(sq_norms[:, None] + sq_norms[None, :] - 2.0 * dot, 0.0)
            dist = np.sqrt(dist2, dtype=dtype)
            np.fill_diagonal(dist, 0.0)

            if mean_dist is None:
                mean_dist = dist.astype(np.float64, copy=True)
                m2_dist = np.zeros_like(mean_dist, dtype=np.float64)
                model_count = 1
            else:
                model_count += 1
                delta = dist - mean_dist
                mean_dist += delta / model_count
                m2_dist += delta * (dist - mean_dist)

            if not args.quiet and (model_count % 10 == 0):
                print(f"Processed {model_count} models...")

        if mean_dist is None or m2_dist is None:
            print("No models were processed; nothing to aggregate.", file=sys.stderr)
            sys.exit(2)

        if model_count < 2:
            std = np.zeros_like(mean_dist)
        else:
            var = m2_dist / (model_count - 1)
            var[var < 0.0] = 0.0
            std = np.sqrt(var)
        invstd = 1.0 / (std + float(args.eps))
        np.fill_diagonal(invstd, 0.0)
        agg = invstd
    else:
        for model_idx, coords in stream_models_from_pdb(pdb_path):
            if args.max_models is not None and model_count >= args.max_models:
                break

            if expected_n is None:
                expected_n = coords.shape[0]
                if not args.quiet:
                    print(f"Detected {expected_n} particles per model (from model {model_idx}).")
            elif coords.shape[0] != expected_n:
                if not args.quiet:
                    print(f"Warning: model {model_idx} has {coords.shape[0]} particles (expected {expected_n}).", file=sys.stderr)

            mat = compute_contact_map(coords, args.mode, args.cutoff, args.sigma, args.float32)
            matrices.append(mat)
            model_count += 1
            if not args.quiet and (model_count % 10 == 0):
                print(f"Processed {model_count} models...")

        if not matrices:
            print("No models were processed; nothing to aggregate.", file=sys.stderr)
            sys.exit(2)

        agg = aggregate_matrices(matrices, args.aggregate)

    # Save image
    save_image(agg, args.out_img, args.colormap, args.dpi, args.vmin, args.vmax)

    # Optionally save matrix
    if args.out_matrix is not None:
        if args.out_matrix.lower().endswith(".npz"):
            np.savez_compressed(args.out_matrix, matrix=agg)
        else:
            np.save(args.out_matrix, agg)

    if not args.quiet:
        print(f"Saved aggregated heatmap to {args.out_img}")
        if args.out_matrix is not None:
            print(f"Saved matrix to {args.out_matrix}")


if __name__ == "__main__":
    main()