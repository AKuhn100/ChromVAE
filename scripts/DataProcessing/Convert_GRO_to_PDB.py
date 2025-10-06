#!/usr/bin/env python3

"""
Convert GRO format file to PDB format for use with Aggregate_ContactMap.py

This script uses MDAnalysis to convert a multi-model GRO file to PDB format,
preserving the MODEL/ENDMDL structure that Aggregate_ContactMap.py expects.

Usage:
    python Convert_GRO_to_PDB.py --gro chr10_10.gro --pdb chr10_10.pdb
"""

import argparse
import os
import sys

try:
    import MDAnalysis as mda
    from MDAnalysis.coordinates.PDB import PDBWriter
except ImportError:
    print("Error: MDAnalysis is required. Install with: pip install MDAnalysis", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert GRO format to PDB format using MDAnalysis")
    parser.add_argument("--gro", required=True, help="Path to input GRO file")
    parser.add_argument("--pdb", required=True, help="Path to output PDB file")
    parser.add_argument("--max_models", type=int, default=None, help="Maximum number of models to convert")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    return parser.parse_args()


def convert_gro_to_pdb(gro_path: str, pdb_path: str, max_models: int = None, quiet: bool = False) -> None:
    """Convert GRO file to PDB format using MDAnalysis."""
    
    if not os.path.isfile(gro_path):
        print(f"Error: GRO file not found: {gro_path}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(pdb_path) or ".", exist_ok=True)
    
    if not quiet:
        print(f"Loading GRO file: {gro_path}")
    
    # Since the GRO file has MODEL/ENDMDL markers, we need to parse it manually
    # but use MDAnalysis for coordinate handling
    models_processed = 0
    total_atoms = 0
    
    with open(pdb_path, "w") as pdb_file:
        with open(gro_path, "r") as gro_file:
            in_model = False
            model_atoms = []
            model_idx = 0
            
            for line in gro_file:
                line = line.strip()
                
                if line.startswith("MODEL"):
                    if in_model and model_atoms:
                        # Process previous model
                        if max_models is None or models_processed < max_models:
                            # Write MODEL line
                            pdb_file.write(f"MODEL     {model_idx + 1:4d}\n")
                            
                            # Write atoms
                            for atom_idx, atom_data in enumerate(model_atoms, 1):
                                residue_num, residue_name, atom_name, x, y, z = atom_data
                                atom_line = f"ATOM  {atom_idx:5d}  {atom_name:>4s} {residue_name:>3s} A{residue_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                                pdb_file.write(atom_line)
                                total_atoms += 1
                            
                            # Write ENDMDL line
                            pdb_file.write("ENDMDL\n")
                            models_processed += 1
                            
                            if not quiet and (models_processed % 100 == 0):
                                print(f"Processed {models_processed} models...")
                        
                        model_atoms = []
                        model_idx += 1
                    
                    in_model = True
                    continue
                
                if in_model and line and not line.startswith("ENDMDL"):
                    # First line after MODEL is the number of atoms - skip it
                    if len(model_atoms) == 0 and line.isdigit():
                        continue
                    
                    # Parse atom line
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            # GRO format: residue_number residue_name atom_name atom_number x y z
                            # But residue_number and residue_name are concatenated
                            residue_info = parts[0]
                            atom_name = parts[1]
                            atom_number = int(parts[2])
                            x = float(parts[3])
                            y = float(parts[4])
                            z = float(parts[5])
                            
                            # Extract residue number and name
                            residue_num_str = ""
                            residue_name = ""
                            for i, char in enumerate(residue_info):
                                if char.isdigit():
                                    residue_num_str += char
                                else:
                                    residue_name = residue_info[i:]
                                    break
                            
                            if residue_num_str and residue_name:
                                residue_num = int(residue_num_str)
                                model_atoms.append((residue_num, residue_name, atom_name, x, y, z))
                        except (ValueError, IndexError):
                            continue
                
                elif line.startswith("ENDMDL") and in_model:
                    # End of model - process it
                    if max_models is None or models_processed < max_models:
                        # Write MODEL line
                        pdb_file.write(f"MODEL     {model_idx + 1:4d}\n")
                        
                        # Write atoms
                        for atom_idx, atom_data in enumerate(model_atoms, 1):
                            residue_num, residue_name, atom_name, x, y, z = atom_data
                            atom_line = f"ATOM  {atom_idx:5d}  {atom_name:>4s} {residue_name:>3s} A{residue_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                            pdb_file.write(atom_line)
                            total_atoms += 1
                        
                        # Write ENDMDL line
                        pdb_file.write("ENDMDL\n")
                        models_processed += 1
                        
                        if not quiet and (models_processed % 100 == 0):
                            print(f"Processed {models_processed} models...")
                    
                    model_atoms = []
                    model_idx += 1
                    in_model = False
            
            # Handle last model if file doesn't end with ENDMDL
            if in_model and model_atoms and (max_models is None or models_processed < max_models):
                # Write MODEL line
                pdb_file.write(f"MODEL     {model_idx + 1:4d}\n")
                
                # Write atoms
                for atom_idx, atom_data in enumerate(model_atoms, 1):
                    residue_num, residue_name, atom_name, x, y, z = atom_data
                    atom_line = f"ATOM  {atom_idx:5d}  {atom_name:>4s} {residue_name:>3s} A{residue_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                    pdb_file.write(atom_line)
                    total_atoms += 1
                
                # Write ENDMDL line
                pdb_file.write("ENDMDL\n")
                models_processed += 1
    
    if not quiet:
        print(f"Conversion complete!")
        print(f"Processed {models_processed} models")
        print(f"Total atoms: {total_atoms}")
        print(f"Output saved to: {pdb_path}")


def main() -> None:
    args = parse_args()
    convert_gro_to_pdb(args.gro, args.pdb, args.max_models, args.quiet)


if __name__ == "__main__":
    main()