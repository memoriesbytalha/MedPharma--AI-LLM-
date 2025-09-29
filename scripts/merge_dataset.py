import pandas as pd
import os

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(MAIN_DIR)
# Load datasets
interactions = pd.read_csv(os.path.join(MAIN_DIR, "data", "drug_interactions_no_sym.csv"))   # CSV1
drugbank = pd.read_csv(os.path.join(MAIN_DIR, "data", "structure links.csv"))      # CSV2

# Normalize drug names (lowercase for safe merging)
interactions['Drug_A'] = interactions['Drug_A'].str.lower().str.strip()
interactions['Drug_B'] = interactions['Drug_B'].str.lower().str.strip()
drugbank['Name'] = drugbank['Name'].str.lower().str.strip()

# Merge DrugA
merged = interactions.merge(
    drugbank[['Name','DrugBank ID','SMILES','Formula']],
    left_on='Drug_A', right_on='Name', how='left'
).rename(columns={
    'DrugBank ID': 'DrugA_ID',
    'SMILES': 'DrugA_SMILES',
    'Formula': 'DrugA_Formula'
}).drop(columns=['Name'])

# Merge DrugB
merged = merged.merge(
    drugbank[['Name','DrugBank ID','SMILES','Formula']],
    left_on='Drug_B', right_on='Name', how='left'
).rename(columns={
    'DrugBank ID': 'DrugB_ID',
    'SMILES': 'DrugB_SMILES',
    'Formula': 'DrugB_Formula'
}).drop(columns=['Name'])

# Save merged dataset
merged.to_csv(os.path.join(MAIN_DIR, "data", "merged_interactions_new.csv"), index=False)

print("✅ Merged dataset saved: merged_interactions.csv")
