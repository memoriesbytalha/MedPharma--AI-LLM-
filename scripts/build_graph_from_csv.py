

import pandas as pd
import torch, pickle
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
import os

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# Root directory (parent of current file)
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("Main dir:", MAIN_DIR)

# ===============================
# Convert SMILES → Morgan Fingerprint
# ===============================
def smiles_to_fp(smiles, n_bits=256):
    """Convert SMILES string to Morgan fingerprint (bit vector)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        arr = np.array(fp.ToList(), dtype=np.float32)  # ExplicitBitVect → numpy
        return arr
    except Exception as e:
        print(f"⚠️ SMILES parse failed for {smiles}: {e}")
        return np.zeros(n_bits, dtype=np.float32)

# ===============================
# Build Graph from CSV
# ===============================
def build_graph_from_csv(csv_path, undirected=True, node_feature_type='smiles'):
    df = pd.read_csv(csv_path)
    df['Drug_A'] = df['Drug_A'].astype(str).str.strip()
    df['Drug_B'] = df['Drug_B'].astype(str).str.strip()
    assert set(['Drug_A', 'Drug_B', 'Level']) <= set(df.columns), "CSV must have Drug_A, Drug_B, Level"

    # --- Create node index mapping ---
    drugs = sorted(set(df['Drug_A']).union(df['Drug_B']))
    node2idx = {d: i for i, d in enumerate(drugs)}
    idx2node = {i: d for d, i in node2idx.items()}

    df['u'] = df['Drug_A'].map(node2idx)
    df['v'] = df['Drug_B'].map(node2idx)

    # --- Encode edge labels ---
    le = LabelEncoder()
    df['label_idx'] = le.fit_transform(df['Level'].astype(str))

    # --- Build edge list ---
    edge_u, edge_v, edge_labels, explanations = [], [], [], []
    for _, row in df.iterrows():
        u, v = int(row['u']), int(row['v'])
        edge_u.append(u)
        edge_v.append(v)
        edge_labels.append(int(row['label_idx']))
        explanations.append(str(row.get('Interaction_Explanation', '')))
        if undirected:
            edge_u.append(v)
            edge_v.append(u)
            edge_labels.append(int(row['label_idx']))
            explanations.append(str(row.get('Interaction_Explanation', '')))

    edge_index = torch.tensor([edge_u, edge_v], dtype=torch.long)
    y = torch.tensor(edge_labels, dtype=torch.long)

    # --- Node features ---
    features = []
    if node_feature_type == 'smiles':
        smiles_map = {}
        if "DrugA_SMILES" in df.columns and "DrugB_SMILES" in df.columns:
            for _, row in df.iterrows():
                if row["Drug_A"] not in smiles_map and pd.notna(row["DrugA_SMILES"]):
                    smiles_map[row["Drug_A"]] = row["DrugA_SMILES"]
                if row["Drug_B"] not in smiles_map and pd.notna(row["DrugB_SMILES"]):
                    smiles_map[row["Drug_B"]] = row["DrugB_SMILES"]

        for d in drugs:
            smi = smiles_map.get(d, "")
            fp = smiles_to_fp(smi, n_bits=256)
            features.append(fp)
    else:
        # One-hot identity features (learnable embedding fallback)
        features = np.eye(len(drugs), dtype=np.float32)

    x = torch.tensor(np.array(features), dtype=torch.float32)

    # --- Package graph ---
    data = Data(x=x, edge_index=edge_index, edge_attr=None)
    data.y = y
    data.explanations = explanations
    data.num_nodes = len(drugs)

    # --- Save graph + metadata ---
    meta = {'node2idx': node2idx, 'idx2node': idx2node, 'label_encoder': le, 'df': df}
    torch.save(data, csv_path + '.pt')
    with open(csv_path + '.meta.pkl', 'wb') as f:
        pickle.dump(meta, f)

    print(f"✅ Saved graph: nodes={len(drugs)} edges={edge_index.size(1)} features={x.size(1)} -> {csv_path}.pt")
    return data, meta

# ===============================
# Main
# ===============================
if __name__ == '__main__':
    csv_path = os.path.join(MAIN_DIR, "data", "drugs_data.csv")
    build_graph_from_csv(csv_path)
