# import os, pickle, torch, numpy as np, pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
# from torch_geometric.data import Data
# from rdkit import Chem
# from rdkit.Chem import AllChem
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import seaborn as sns
# from models.edge_gnn import EdgeGNN
# import warnings
# from torch_geometric.data import Data
# warnings.filterwarnings("ignore")



# # --------------------------
# # Convert SMILES to fingerprints
# # --------------------------
# def smiles_to_fp(smiles, radius=2, n_bits=128):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return np.zeros(n_bits)
#     fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
#     arr = np.zeros((1,))
#     Chem.DataStructs.ConvertToNumpyArray(fp, arr)
#     return arr

# # --------------------------
# # Prepare PyG Data
# # --------------------------
# def prepare_graph(csv_path, fp_dim=128):
#     df = pd.read_csv(csv_path)
#     le = LabelEncoder()
#     df['Level_enc'] = le.fit_transform(df['Level'])

#     # Create node features from all unique drugs
#     drugs = pd.concat([df['Drug_A'], df['Drug_B']]).unique()
#     drug2idx = {d:i for i,d in enumerate(drugs)}
#     node_features = []
#     for d in drugs:
#         smiles = df[df['Drug_A']==d]['DrugA_SMILES'].iloc[0] if d in df['Drug_A'].values else df[df['Drug_B']==d]['DrugB_SMILES'].iloc[0]
#         node_features.append(smiles_to_fp(smiles, n_bits=fp_dim))
#     node_features = torch.tensor(np.array(node_features), dtype=torch.float)

#     # Edge indices
#     edge_index = torch.tensor([
#         [drug2idx[a] for a in df['Drug_A']],
#         [drug2idx[b] for b in df['Drug_B']]
#     ], dtype=torch.long)

#     # Edge labels
#     y = torch.tensor(df['Level_enc'].values, dtype=torch.long)

#     # PyG Data
#     data = Data(x=node_features, edge_index=edge_index, y=y)
#     meta = {'label_encoder': le, 'drug2idx': drug2idx}
#     torch.save(data, csv_path + '.pt')
#     pickle.dump(meta, open(csv_path + '.meta.pkl','wb'))
#     print("Graph prepared and saved.")
#     return data, meta

# # --------------------------
# # Plot confusion matrix
# # --------------------------
# def plot_confusion_matrix(y_true, y_pred, labels, out_path="confusion_matrix.png"):
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(6,5))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#                 xticklabels=labels, yticklabels=labels)
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.title("Confusion Matrix")
#     plt.tight_layout()
#     plt.savefig(out_path)
#     plt.close()

# # --------------------------
# # Split edges into train/val/test
# # --------------------------
# def split_edges(num_edges, train_frac=0.8, val_frac=0.1, seed=123):
#     idx = np.arange(num_edges)
#     np.random.seed(seed)
#     np.random.shuffle(idx)
#     n_train = int(train_frac * num_edges)
#     n_val = int(val_frac * num_edges)
#     return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

# # --------------------------
# # Training function
# # --------------------------
# def train_edge_gnn(csv_path, epochs=200, lrs=[0.0005, 0.001, 0.002, 0.005], device='cuda'):
#     device = torch.device(device if torch.cuda.is_available() else 'cpu')
#     with torch.serialization.safe_globals([Data]):
#         data = torch.load(csv_path + '.pt', weights_only=False)
#     meta = pickle.load(open(csv_path + '.meta.pkl','rb'))

#     num_nodes = data.num_nodes
#     num_edges = data.edge_index.size(1)
#     print(f"Nodes: {num_nodes}, Edges: {num_edges}")

#     y = data.y
#     train_idx, val_idx, test_idx = split_edges(num_edges)

#     y_numpy = y.cpu().numpy()
#     class_weights_np = compute_class_weight(
#         class_weight='balanced',
#         classes=np.unique(y_numpy),
#         y=y_numpy
#     )
#     class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(device)

#     for lr in lrs:
#         print(f"\n--- Training with LR={lr} ---")
#         model = EdgeGNN(
#             num_nodes=num_nodes,
#             node_feat_dim=data.x.size(1),
#             node_embed_dim=256,
#             hidden_dim=512,
#             num_classes=len(meta['label_encoder'].classes_),
#             dropout=0.3,
#             use_node_features=True
#         ).to(device)

#         opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
#         criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
#         data = data.to(device)
#         y = y.to(device)

#         # For metrics
#         train_accs, val_f1s, train_losses = [], [], []
#         best_val = 0.0
#         best_state = None

#         for epoch in range(1, epochs+1):
#             model.train()
#             opt.zero_grad()
#             logits, _ = model(data)
#             loss = criterion(logits[train_idx], y[train_idx])
#             loss.backward()
#             opt.step()

#             # Train accuracy
#             preds_train = logits[train_idx].argmax(dim=1).cpu().numpy()
#             acc_train = accuracy_score(y[train_idx].cpu().numpy(), preds_train)

#             # Validation F1
#             model.eval()
#             with torch.no_grad():
#                 logits_eval, _ = model(data)
#                 preds_eval = logits_eval.argmax(dim=1).cpu().numpy()
#                 val_f1 = f1_score(y.cpu().numpy()[val_idx], preds_eval[val_idx], average='macro', zero_division=0)

#                 if val_f1 > best_val:
#                     best_val = val_f1
#                     best_state = model.state_dict()

#             train_accs.append(acc_train)
#             val_f1s.append(val_f1)
#             train_losses.append(loss.item())

#             if epoch % 10 == 0 or epoch == 1:
#                 print(f"Epoch {epoch}: loss={loss.item():.4f}, trainAcc={acc_train:.4f}, valF1={val_f1:.4f}")

#         # Save metrics
#         metrics_df = pd.DataFrame({
#             'epoch': list(range(1, epochs+1)),
#             'train_loss': train_losses,
#             'train_accuracy': train_accs,
#             'val_f1': val_f1s
#         })
#         metrics_csv_path = os.path.join(MAIN_DIR, f"training_metrics_lr_{lr}.csv")
#         metrics_df.to_csv(metrics_csv_path, index=False)
#         print(f"Training metrics saved as {metrics_csv_path}")

#         # Save graphs
#         plt.figure()
#         plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
#         plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'Train Loss LR={lr}')
#         plt.grid(True); plt.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(MAIN_DIR, f"train_loss_lr_{lr}.png"))

#         plt.figure()
#         plt.plot(range(1, epochs+1), train_accs, label='Train Acc')
#         plt.plot(range(1, epochs+1), val_f1s, label='Val F1')
#         plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title(f'Train Acc & Val F1 LR={lr}')
#         plt.grid(True); plt.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(MAIN_DIR, f"train_val_metrics_lr_{lr}.png"))
#         plt.close()
#         print("Training graphs saved.")

#         # Load best model
#         if best_state:
#             model.load_state_dict(best_state)

#         # Test metrics
#         model.eval()
#         with torch.no_grad():
#             logits_final, _ = model(data)
#             preds_test = logits_final.argmax(dim=1).cpu().numpy()
#             y_test = y.cpu().numpy()[test_idx]

#         print("\nTest classification report:")
#         print(classification_report(y_test, preds_test[test_idx], target_names=list(meta['label_encoder'].classes_)))

#         # Confusion matrix
#         cm_path = os.path.join(MAIN_DIR, f"confusion_matrix_lr_{lr}.png")
#         plot_confusion_matrix(y_test, preds_test[test_idx], list(meta['label_encoder'].classes_), cm_path)
#         print(f"Confusion matrix saved as {cm_path}")

#         # Save model
#         model_path = os.path.join(MAIN_DIR, f"drugs_data_edge_gnn_lr_{lr}.pt")
#         torch.save(model.state_dict(), model_path)
#         print(f"Saved model: {model_path}")

# # --------------------------
# if __name__ == "__main__":
#     MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#     csv_path = os.path.join(MAIN_DIR, "data", "drugs_data.csv")
#     prepare_graph(csv_path)
#     train_edge_gnn(csv_path, epochs=200, lrs=[0.005])

import os, pickle, torch, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from rdkit import Chem
from rdkit.Chem import AllChem
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from models.edge_gnn import EdgeGNN
import warnings
warnings.filterwarnings("ignore")
from torch_geometric.data import Data

# --------------------------
# Convert SMILES to fingerprints
# --------------------------
def smiles_to_fp(smiles, radius=3, n_bits=512):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,))
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# --------------------------
# Prepare PyG Data from CSV
# --------------------------
def prepare_graph(csv_path):
    df = pd.read_csv(csv_path)
    le = LabelEncoder()
    df['Level_enc'] = le.fit_transform(df['Level'])

    drugs = pd.concat([df['Drug_A'], df['Drug_B']]).unique()
    drug2idx = {d: i for i, d in enumerate(drugs)}

    node_features = []
    for d in drugs:
        if d in df['Drug_A'].values:
            smiles = df[df['Drug_A']==d]['DrugA_SMILES'].iloc[0]
        else:
            smiles = df[df['Drug_B']==d]['DrugB_SMILES'].iloc[0]
        node_features.append(smiles_to_fp(smiles))

    node_features = torch.tensor(np.array(node_features), dtype=torch.float)

    edge_index = torch.tensor([
        [drug2idx[a] for a in df['Drug_A']],
        [drug2idx[b] for b in df['Drug_B']]
    ], dtype=torch.long)

    y = torch.tensor(df['Level_enc'].values, dtype=torch.long)

    data = Data(x=node_features, edge_index=edge_index, y=y)
    meta = {'label_encoder': le, 'drug2idx': drug2idx}
    return data, meta

# --------------------------
# Confusion matrix
# --------------------------
def plot_confusion_matrix(y_true, y_pred, labels, out_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# --------------------------
# Train/Test split
# --------------------------
def split_edges(num_edges, train_frac=0.8, val_frac=0.1, seed=123):
    idx = np.arange(num_edges)
    np.random.seed(seed)
    np.random.shuffle(idx)
    n_train = int(train_frac*num_edges)
    n_val = int(val_frac*num_edges)
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

# --------------------------
# Training
# --------------------------
def train_edge_gnn(csv_path, epochs, lr=0.005, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    data, meta = prepare_graph(csv_path)
    data = data.to(device)

    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    y = data.y.to(device)

    train_idx, val_idx, test_idx = split_edges(num_edges)

    class_weights = torch.tensor(compute_class_weight(
        'balanced', classes=np.unique(y.cpu().numpy()), y=y.cpu().numpy()
    ), dtype=torch.float).to(device)

    model = EdgeGNN(
        num_nodes=num_nodes,
        node_feat_dim=data.x.size(1),
        node_embed_dim=512,
        hidden_dim=768,
        num_classes=len(meta['label_encoder'].classes_),
        dropout=0.2,
        num_layers=3,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_val = 0.0
    best_state = None
    train_accs, val_f1s, train_losses = [], [], []

    for epoch in range(1, epochs+1):
        model.train()
        opt.zero_grad()
        logits, _ = model(data)
        loss = criterion(logits[train_idx], y[train_idx])
        loss.backward()
        opt.step()

        preds_train = logits[train_idx].argmax(dim=1)
        acc_train = (preds_train == y[train_idx]).float().mean().item()

        model.eval()
        with torch.no_grad():
            logits_eval, _ = model(data)
            preds_eval = logits_eval.argmax(dim=1)
            val_f1 = f1_score(y[val_idx].cpu(), preds_eval[val_idx].cpu(), average='macro', zero_division=0)
            if val_f1 > best_val:
                best_val = val_f1
                best_state = model.state_dict()

        train_accs.append(acc_train)
        val_f1s.append(val_f1)
        train_losses.append(loss.item())

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch}: loss={loss.item():.4f}, trainAcc={acc_train:.4f}, valF1={val_f1:.4f}")

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    with torch.no_grad():
        logits_test, _ = model(data)
        preds_test = logits_test[test_idx].argmax(dim=1)
        y_test = y[test_idx].cpu()

    print("\nTest Classification Report:")
    print(classification_report(y_test, preds_test.cpu(), target_names=list(meta['label_encoder'].classes_)))

    # Confusion matrix
    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(y_test, preds_test.cpu(), list(meta['label_encoder'].classes_), "confusion_matrix_final.png")
    print(confusion_matrix(y_test, preds_test.cpu()))
    print("Confusion matrix saved.")
    
    # Save model
    import os

    save_path = 'output/edge_gnn_best.pt'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # <-- create parent folder
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
if __name__ == "__main__":
    MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(MAIN_DIR, "data", "drugs_data.csv")
    train_edge_gnn(csv_path, epochs=200, lr=0.005)