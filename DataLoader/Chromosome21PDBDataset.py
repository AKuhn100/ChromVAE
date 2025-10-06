import os
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class Chromosome21PDBDataset(Dataset):
    """
    PyTorch Dataset that loads per-MODEL coordinates from a PDB and returns
    each model as a flattened vector [x1, y1, z1, x2, y2, z2, ...] (float32).

    Assumptions/behavior:
    - Parses MODEL/ENDMDL blocks. Lines between these markers with record type in
      `record_types` are considered. By default only `ATOM` records are used.
    - The order of atoms within a model is preserved. All models must have the
      same number of atoms (validation runs at construction).
    - Coordinates are parsed using the PDB fixed-width fields: x[30:38],
      y[38:46], z[46:54].

    Parameters
    - pdb_path: Absolute or relative path to the PDB file (e.g., Data/chromosome21_aligned.pdb)
    - record_types: Tuple of record names to include (e.g., ("ATOM",) or ("ATOM", "HETATM"))
    - atom_name_filter: Optional set of atom names to include (e.g., {"CA"}). If None, all are used.
    - center: If True, subtract per-model centroid from coordinates.
    - scale: Optional scalar to divide coordinates by (applied after centering if enabled).

    Returns
    - __getitem__(i): torch.FloatTensor of shape [3 * num_atoms]

    Example
    -------
    >>> from torch.utils.data import DataLoader
    >>> ds = Chromosome21PDBDataset(
    ...     pdb_path="Data/chromosome21_aligned.pdb",
    ...     record_types=("ATOM",),
    ... )
    >>> len(ds), ds.vector_length, ds.atoms_per_model
    ... # doctest: +SKIP
    (N_models, 3 * N_atoms, N_atoms)
    >>> loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=Chromosome21PDBDataset.collate_batch)
    >>> batch = next(iter(loader))
    >>> batch.shape  # [B, 3 * N_atoms]
    torch.Size([16, 3 * N_atoms])
    """

    def __init__(
        self,
        pdb_path: str,
        record_types: Tuple[str, ...] = ("ATOM",),
        atom_name_filter: Optional[Sequence[str]] = None,
        center: bool = False,
        scale: Optional[float] = None,
    ) -> None:
        if not os.path.isfile(pdb_path):
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")

        self.pdb_path = pdb_path
        self.record_types = tuple(record_types)
        self.atom_name_filter = set(atom_name_filter) if atom_name_filter is not None else None
        self.center = center
        self.scale = scale

        # Read and parse once; keep all models in memory for simplicity
        self._models_xyz: List[torch.Tensor] = self._parse_models()

        if len(self._models_xyz) == 0:
            raise ValueError("No models were parsed from the PDB. Check formatting and filters.")

        # Validate consistent atom count across models
        atom_counts = {xyz.shape[0] for xyz in self._models_xyz}
        if len(atom_counts) != 1:
            details = ", ".join(str(c) for c in sorted(atom_counts))
            raise ValueError(
                f"Inconsistent atom counts across models: {details}. "
                f"Ensure identical records and filters per model."
            )

        self.atoms_per_model: int = next(iter(atom_counts))
        self.vector_length: int = 3 * self.atoms_per_model

        # Optionally center/scale
        if self.center or self.scale not in (None, 1.0):
            normalized: List[torch.Tensor] = []
            for xyz in self._models_xyz:
                xyz_adj = xyz
                if self.center:
                    centroid = xyz_adj.mean(dim=0, keepdim=True)
                    xyz_adj = xyz_adj - centroid
                if self.scale not in (None, 1.0):
                    xyz_adj = xyz_adj / float(self.scale)  # type: ignore[arg-type]
                normalized.append(xyz_adj)
            self._models_xyz = normalized

        # Pre-flattened views for fast indexing
        self._flat_vectors: List[torch.Tensor] = [xyz.reshape(-1).to(torch.float32) for xyz in self._models_xyz]

    def __len__(self) -> int:
        return len(self._flat_vectors)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._flat_vectors[index]

    @staticmethod
    def collate_batch(items: Iterable[torch.Tensor]) -> torch.Tensor:
        """Collate a batch of flattened vectors into a [B, L] float32 tensor.
        Validates that all items share the same length.
        """
        items_list = list(items)
        if len(items_list) == 0:
            return torch.empty(0, 0, dtype=torch.float32)
        lengths = {int(t.numel()) for t in items_list}
        if len(lengths) != 1:
            raise ValueError(f"Batch items have differing lengths: {sorted(lengths)}")
        return torch.stack(items_list, dim=0).to(torch.float32)

    # -------------------------
    # Internal parsing helpers
    # -------------------------
    def _parse_models(self) -> List[torch.Tensor]:
        models: List[List[Tuple[float, float, float]]] = []
        current: Optional[List[Tuple[float, float, float]]] = None

        with open(self.pdb_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("MODEL"):
                    if current is not None:
                        # Nested MODEL without ENDMDL; start new and keep previous
                        models.append(current)
                    current = []
                    continue
                if line.startswith("ENDMDL"):
                    if current is not None:
                        models.append(current)
                        current = None
                    continue

                if current is None:
                    # Skip any coordinates outside MODEL/ENDMDL blocks
                    continue

                rec = line[0:6].strip()
                if rec not in self.record_types:
                    continue

                # Optional atom name filter (cols 12-16 per PDB)
                if self.atom_name_filter is not None:
                    atom_name = line[12:16].strip()
                    if atom_name not in self.atom_name_filter:
                        continue

                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    # Malformed coordinate line; skip
                    continue
                current.append((x, y, z))

        # If file ended inside a model without ENDMDL, keep it
        if current is not None:
            models.append(current)

        # Convert to tensors
        tensors: List[torch.Tensor] = []
        for coords in models:
            if len(coords) == 0:
                continue
            xyz = torch.tensor(coords, dtype=torch.float32)  # [N, 3]
            tensors.append(xyz)
        return tensors


# Convenience function to infer dimensions for a simple MLP/AE
def infer_input_dim_from_pdb(pdb_path: str, **dataset_kwargs) -> Tuple[int, int]:
    """Return (num_models, vector_length) from a PDB by constructing the dataset.

    Example
    -------
    >>> infer_input_dim_from_pdb("Data/chromosome21_aligned.pdb")  # doctest: +SKIP
    (N_models, 3 * N_atoms)
    """
    ds = Chromosome21PDBDataset(pdb_path=pdb_path, **dataset_kwargs)
    return len(ds), ds.vector_length


