import pandas as pd
import torch, pickle
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
import os
import os

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(MAIN_DIR)

# import pdb; pdb.set_trace()
def build_graph_from_csv(csv_path, undirected=True, node_feature_type='learnable'):
    df = pd.read_csv(csv_path)
    df['Drug_A'] = df['Drug_A'].astype(str).str.strip()
    df['Drug_B'] = df['Drug_B'].astype(str).str.strip()
    assert set(['Drug_A','Drug_B','Level']) <= set(df.columns), "CSV must have Drug_A,Drug_B,Level"

    drugs = sorted(set(df['Drug_A']).union(df['Drug_B']))
    node2idx = {d:i for i,d in enumerate(drugs)}
    idx2node = {i:d for d,i in node2idx.items()}

    df['u'] = df['Drug_A'].map(node2idx)
    df['v'] = df['Drug_B'].map(node2idx)

    le = LabelEncoder()
    df['label_idx'] = le.fit_transform(df['Level'].astype(str))

    edge_u = []
    edge_v = []
    edge_labels = []
    explanations = []
    for _, row in df.iterrows():
        u = int(row['u']); v = int(row['v'])
        edge_u.append(u); edge_v.append(v)
        edge_labels.append(int(row['label_idx']))
        explanations.append(str(row.get('Explanation','')))
        if undirected:
            edge_u.append(v); edge_v.append(u)
            edge_labels.append(int(row['label_idx']))
            explanations.append(str(row.get('Explanation','')))

    edge_index = torch.tensor([edge_u, edge_v], dtype=torch.long)
    y = torch.tensor(edge_labels, dtype=torch.long)

    # No explicit node features (we'll use learnable Embedding in model)
    data = Data(x=None, edge_index=edge_index, edge_attr=None)
    data.y = y
    data.explanations = explanations
    data.num_nodes = len(drugs)

    meta = {'node2idx': node2idx, 'idx2node': idx2node, 'label_encoder': le, 'df': df}
    torch.save(data, csv_path + '.pt')
    with open(csv_path + '.meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    print(f"Saved graph: nodes={len(drugs)} edges={edge_index.size(1)} -> {csv_path}.pt")
    return data, meta

if __name__ == '__main__':
    import sys
    import os
    
    csv_path = os.path.join(MAIN_DIR, "data", "interaction_explanations.csv")
    # import pdb ; pdb.set_trace()
    build_graph_from_csv(csv_path)