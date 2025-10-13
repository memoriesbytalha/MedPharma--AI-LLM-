import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, confusion_matrix
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data
from models.edge_gnn import EdgeGNN

warnings.filterwarnings("ignore")


# ============================================================
# SMILES → Morgan Fingerprint
# ============================================================
def smiles_to_fp(smiles, radius=3, n_bits=512):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,))
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# ============================================================
# Prepare Graph Data
# ============================================================
def prepare_graph(csv_path):
    df = pd.read_csv(csv_path)
    le = LabelEncoder()
    df["Level_enc"] = le.fit_transform(df["Level"])

    drugs = pd.concat([df["Drug_A"], df["Drug_B"]]).unique()
    drug2idx = {d: i for i, d in enumerate(drugs)}

    node_features = []
    for d in drugs:
        if d in df["Drug_A"].values:
            smiles = df[df["Drug_A"] == d]["DrugA_SMILES"].iloc[0]
        else:
            smiles = df[df["Drug_B"] == d]["DrugB_SMILES"].iloc[0]
        node_features.append(smiles_to_fp(smiles))

    node_features = StandardScaler().fit_transform(np.array(node_features))
    node_features = torch.tensor(node_features, dtype=torch.float)

    edge_index = torch.tensor([
        [drug2idx[a] for a in df["Drug_A"]],
        [drug2idx[b] for b in df["Drug_B"]]
    ], dtype=torch.long)

    y = torch.tensor(df["Level_enc"].values, dtype=torch.long)
    data = Data(x=node_features, edge_index=edge_index, y=y)
    meta = {"label_encoder": le, "drug2idx": drug2idx}
    return data, meta


# ============================================================
# Stratified Edge Split
# ============================================================
def stratified_edge_split(y, train_frac=0.8, val_frac=0.1):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_frac))
    for train_val_idx, test_idx in sss.split(np.arange(len(y)), y):
        y_train_val = y[train_val_idx]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac / (train_frac + val_frac))
        for train_idx, val_idx in sss2.split(train_val_idx, y_train_val):
            return train_idx, val_idx, test_idx


# ============================================================
# Confusion Matrix Plot
# ============================================================
def plot_confusion_matrix(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test Data)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ============================================================
# Training Function
# ============================================================
def train_edge_gnn(csv_path, epochs=200, lr=0.005, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    data, meta = prepare_graph(csv_path)
    data = data.to(device)

    y = data.y.cpu().numpy()
    train_idx, val_idx, test_idx = stratified_edge_split(y)
    train_idx, val_idx, test_idx = (
        torch.tensor(train_idx, dtype=torch.long, device=device),
        torch.tensor(val_idx, dtype=torch.long, device=device),
        torch.tensor(test_idx, dtype=torch.long, device=device),
    )

    class_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.unique(y), y=y),
        dtype=torch.float, device=device
    )

    model = EdgeGNN(
        num_nodes=data.num_nodes,
        node_feat_dim=data.x.size(1),
        node_embed_dim=512,
        hidden_dim=768,
        num_classes=len(meta["label_encoder"].classes_),
        dropout=0.3,
        num_layers=3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    best_val, patience, wait = 0, 20, 0
    best_state = None
    train_accs, val_f1s, losses = [], [], []

    print("\n🚀 Training Started\n")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data)
        loss = criterion(logits[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()
        scheduler.step()

        preds_train = logits[train_idx].argmax(dim=1)
        acc_train = (preds_train == data.y[train_idx]).float().mean().item()

        model.eval()
        with torch.no_grad():
            logits_val, _ = model(data)
            preds_val = logits_val[val_idx].argmax(dim=1)
            val_f1 = f1_score(data.y[val_idx].cpu(), preds_val.cpu(), average="macro", zero_division=0)

        train_accs.append(acc_train)
        val_f1s.append(val_f1)
        losses.append(loss.item())

        if val_f1 > best_val:
            best_val, best_state = val_f1, model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"⏹ Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch:03d} | Loss={loss.item():.4f} | TrainAcc={acc_train:.4f} | ValF1={val_f1:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    # ============================ Testing ============================
    model.eval()
    with torch.no_grad():
        logits_test, _ = model(data)
        preds_test = logits_test[test_idx].argmax(dim=1)
        y_test = data.y[test_idx].cpu()

    acc = accuracy_score(y_test, preds_test.cpu())
    f1 = f1_score(y_test, preds_test.cpu(), average="macro", zero_division=0)

    os.makedirs("output/Final", exist_ok=True)

    print("\n✅ Classification Report\n")
    class_report = classification_report(
        y_test, preds_test.cpu(), target_names=list(meta["label_encoder"].classes_), digits=3
    )
    print(class_report)
    with open("output/Final/classification_report.txt", "w") as f:
        f.write("Classification Report\n\n")
        f.write(class_report)

    # ============================ Plots ============================
    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label="Train Accuracy", color="tab:blue", linewidth=2)
    plt.plot(val_f1s, label="Validation F1", color="tab:orange", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Training vs Validation F1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/Final/training_curves.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(losses, color="tab:red", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/Final/training_loss_curve.png", dpi=300)
    plt.close()

    # Confusion Matrix
    plot_confusion_matrix(
        y_test, preds_test.cpu(),
        list(meta["label_encoder"].classes_),
        "output/Final/confusion_matrix.png"
    )

    # ============================ Final Summary ============================
    print("\n📈 Final Results Summary")
    print(f"Train Accuracy: {np.mean(train_accs[-10:]) * 100:.2f}%")
    print(f"Validation F1: {best_val * 100:.2f}%")
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Test F1: {f1 * 100:.2f}%")
    print("\n✅ All graphs & reports saved in 'output/Final'")

    torch.save(model.state_dict(), "output/Final/edge_gnn.pt")
    print("✅ Model saved to output/Final/edge_gnn.pt")


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(MAIN_DIR, "data", "drugs_data.csv")
    train_edge_gnn(csv_path, epochs=100, lr=0.005)
