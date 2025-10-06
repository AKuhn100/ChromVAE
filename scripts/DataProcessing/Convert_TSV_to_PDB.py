import pandas as pd

# Load data
df = pd.read_csv('Data/chromosome21.tsv', sep='\t')

# Check the structure
print("Columns:", df.columns.tolist())
print("\nUnique chromosome copies:", df['Chromosome copy number'].unique())
print("Total rows:", len(df))

# Filter for just ONE chromosome copy from ONE cell
# The data is organized by chromosome copy number
# Let's take copy number 1 (first homolog)
single_chr = df[df['Chromosome copy number'] == 1].copy()

# Remove rows with missing coordinates
single_chr = single_chr.dropna(subset=['X(nm)', 'Y(nm)', 'Z(nm)'])

print(f"\nFiltered to {len(single_chr)} beads for one chromosome copy")

# Convert to Angstroms
single_chr['X_A'] = single_chr['X(nm)'] * 10
single_chr['Y_A'] = single_chr['Y(nm)'] * 10
single_chr['Z_A'] = single_chr['Z(nm)'] * 10

# Write PDB file
with open('chromosome21_single.pdb', 'w') as f:
    for idx, (i, row) in enumerate(single_chr.iterrows()):
        # PDB ATOM format
        f.write(f"ATOM  {idx+1:5d}  CA  ALA A{idx+1:4d}    "
                f"{row['X_A']:8.3f}{row['Y_A']:8.3f}{row['Z_A']:8.3f}"
                f"  1.00  0.00           C\n")
    
    # Add connectivity only for consecutive beads
    for i in range(len(single_chr)-1):
        f.write(f"CONECT{i+1:5d}{i+2:5d}\n")
    
    f.write("END\n")

print("Saved chromosome21_single.pdb")
print(f"File has {len(single_chr)} atoms and {len(single_chr)-1} bonds")