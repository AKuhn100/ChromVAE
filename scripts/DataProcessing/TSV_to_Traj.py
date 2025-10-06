import pandas as pd

# Load data
df = pd.read_csv('Data/chromosome21.tsv', sep='\t')

print("Columns:", df.columns.tolist())
print("Unique chromosome copies:", df['Chromosome copy number'].unique())

# Get all unique chromosome copies
copy_numbers = sorted(df['Chromosome copy number'].unique())
print(f"\nFound {len(copy_numbers)} chromosome copies")

# Write multi-model PDB file
with open('chromosome21_trajectory.pdb', 'w') as f:
    
    for model_num, copy_num in enumerate(copy_numbers, start=1):
        # Filter for this chromosome copy
        single_chr = df[df['Chromosome copy number'] == copy_num].copy()
        
        # Remove rows with missing coordinates
        single_chr = single_chr.dropna(subset=['X(nm)', 'Y(nm)', 'Z(nm)'])
        
        if len(single_chr) == 0:
            continue
        
        print(f"Copy {copy_num}: {len(single_chr)} beads")
        
        # Convert to Angstroms
        single_chr['X_A'] = single_chr['X(nm)'] * 10
        single_chr['Y_A'] = single_chr['Y(nm)'] * 10
        single_chr['Z_A'] = single_chr['Z(nm)'] * 10
        
        # Write MODEL header
        f.write(f"MODEL     {model_num:4d}\n")
        
        # Write atoms
        for idx, (i, row) in enumerate(single_chr.iterrows()):
            f.write(f"ATOM  {idx+1:5d}  CA  ALA A{idx+1:4d}    "
                    f"{row['X_A']:8.3f}{row['Y_A']:8.3f}{row['Z_A']:8.3f}"
                    f"  1.00  0.00           C\n")
        
        # Write ENDMDL
        f.write("ENDMDL\n")
    
    # CONECT records only need to be written once (connectivity is same for all models)
    # Write them after all models
    single_chr = df[df['Chromosome copy number'] == copy_numbers[0]].dropna(subset=['X(nm)', 'Y(nm)', 'Z(nm)'])
    for i in range(len(single_chr)-1):
        f.write(f"CONECT{i+1:5d}{i+2:5d}\n")
    
    f.write("END\n")

print(f"\nSaved chromosome21_trajectory.pdb with {len(copy_numbers)} frames")