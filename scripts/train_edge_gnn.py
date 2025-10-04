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

        if epoch % 1 == 0 or epoch == 1:
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

    test_acc = accuracy_score(y_test, preds_test.cpu())
    test_f1 = f1_score(y_test, preds_test.cpu(), average='macro', zero_division=0)
    print(f"\nTest Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    # Save training curves as images
    # os.makedirs('output', exist_ok=True)
    epochs_range = range(1, len(train_accs) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_accs, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('output', 'train_accuracy.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, val_f1s, label='Validation F1', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('output', 'val_f1.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_losses, label='Train Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('output', 'train_loss.png'))
    plt.close()

    # Optionally save numeric curves for later inspection
    np.savez(os.path.join('output', 'training_curves.npz'),
             train_accs=np.array(train_accs),
             val_f1s=np.array(val_f1s),
             train_losses=np.array(train_losses))

    print("Training curves saved to output/")

    print("\nTest Classification Report:")
    class_report = classification_report(y_test, preds_test.cpu(), target_names=list(meta['label_encoder'].classes_))
    print(class_report)
    save_path = 'output/classification_report.txt'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(class_report)

    # Confusion matrix
    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(y_test, preds_test.cpu(), list(meta['label_encoder'].classes_), "confusion_matrix_final.png")
    print(confusion_matrix(y_test, preds_test.cpu()))
    print("Confusion matrix saved.")

    confusion_mtx = confusion_matrix(y_test, preds_test.cpu())
    save_path = 'output/confusion_matrix_final.png'
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(meta['label_encoder'].classes_),
                yticklabels=list(meta['label_encoder'].classes_))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    save_path = 'output/edge_gnn_best.pt'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # <-- create parent folder
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
if __name__ == "__main__":
    MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(MAIN_DIR, "data", "drugs_data.csv")
    train_edge_gnn(csv_path, epochs=200, lr=0.005)