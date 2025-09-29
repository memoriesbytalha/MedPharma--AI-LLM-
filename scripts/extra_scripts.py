import torch, random, numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from models.edge_gnn import EdgeGNN
import pickle, os

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------
# Focal Loss Implementation
# --------------------------
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# --------------------------
# Split edges into train/val/test
# --------------------------
def split_edges(num_edges, train_frac=0.8, val_frac=0.1, seed=123):
    idx = np.arange(num_edges)
    np.random.seed(seed)
    np.random.shuffle(idx)
    n_train = int(train_frac * num_edges)
    n_val = int(val_frac * num_edges)
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

# --------------------------
# Plot confusion matrix
# --------------------------
def plot_confusion(y_true, y_pred, labels, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return cm

# --------------------------
# Main training function
# --------------------------
def main(csv_path, epochs=100, lr=5e-4, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # 🔹 Load the pre-saved PyG graph safely
    with torch.serialization.safe_globals([Data]):
        data = torch.load(csv_path + '.pt', weights_only=False)

    meta = pickle.load(open(csv_path + '.meta.pkl','rb'))

    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    print("Nodes:", num_nodes, "Edges:", num_edges)

    y = data.y
    train_idx, val_idx, test_idx = split_edges(num_edges, 0.8, 0.1)

    # --------------------------
    # Compute class weights
    # --------------------------
    y_numpy = y.cpu().numpy()
    class_weights_np = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_numpy),
        y=y_numpy
    )
    # Extra boost for Moderate (last class index)
    class_weights_np[-1] *= 1.3
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(device)

    criterion = FocalLoss(alpha=class_weights, gamma=2)

    # 🔹 Initialize EdgeGNN model
    model = EdgeGNN(
        num_nodes=num_nodes,
        node_embed_dim=128,
        hidden_dim=256,
        num_classes=len(meta['label_encoder'].classes_)
    )
    model.to(device)
    data = data.to(device)
    y = y.to(device)

    # 🔹 Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # --------------------------
    # Training loop
    # --------------------------
    best_val = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits, _ = model(data)
        loss = criterion(logits[train_idx], y[train_idx])
        loss.backward()
        opt.step()

        # --------------------------
        # Validation
        # --------------------------
        model.eval()
        with torch.no_grad():
            logits_eval, _ = model(data)
            preds = logits_eval.argmax(dim=1).cpu().numpy()
            y_all = y.cpu().numpy()

            # Macro and per-class F1
            val_f1_macro = f1_score(y_all[val_idx], preds[val_idx], average='macro')
            per_class_f1 = f1_score(y_all[val_idx], preds[val_idx], average=None)

            moderate_f1 = per_class_f1[-1]  # Assuming last index is Moderate

            if moderate_f1 > best_val:  # Save based on Moderate performance
                best_val = moderate_f1
                best_state = model.state_dict()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch} loss {loss.item():.4f} "
                  f"valF1_macro {val_f1_macro:.4f} "
                  f"ModerateF1 {moderate_f1:.4f}")

    # --------------------------
    # Load best model
    # --------------------------
    if best_state:
        model.load_state_dict(best_state)

    # --------------------------
    # Test metrics
    # --------------------------
    model.eval()
    with torch.no_grad():
        logits_final, node_emb = model(data)
        preds = logits_final.argmax(dim=1).cpu().numpy()
        y_all = y.cpu().numpy()

    print("\nTest classification report:")
    print(classification_report(
        y_all[test_idx],
        preds[test_idx],
        target_names=list(meta['label_encoder'].classes_)
    ))

    cm = plot_confusion(y_all[test_idx], preds[test_idx],
                        labels=list(meta['label_encoder'].classes_),
                        save_path=csv_path + "_confusion.png")
    print("Confusion matrix:\n", cm)

    # --------------------------
    # Save trained model weights
    # --------------------------
    torch.save(model.state_dict(), csv_path + '.edge_gnn.pt')
    print("Saved model:", csv_path + '.edge_gnn.pt')
    print("Saved confusion matrix image:", csv_path + "_confusion.png")

# --------------------------
# Entry point
# --------------------------
if __name__ == '__main__':
    csv_path = os.path.join(MAIN_DIR, "data", "interaction_explanations.csv")
    main(csv_path=csv_path, epochs=100)





















import torch, random, numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data
from models.edge_gnn import EdgeGNN
import pickle, os
import matplotlib.pyplot as plt
import seaborn as sns

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------
# Split edges into train/val/test
# --------------------------
def split_edges(num_edges, train_frac=0.8, val_frac=0.1, seed=123):
    idx = np.arange(num_edges)
    np.random.seed(seed)
    np.random.shuffle(idx)
    n_train = int(train_frac * num_edges)
    n_val = int(val_frac * num_edges)
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

# --------------------------
# Plot confusion matrix
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
    print(f"Confusion matrix saved as {out_path}")

# --------------------------
# Main training function
# --------------------------
def main(csv_path, epochs, lr=5e-4, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # 🔹 Load the pre-saved PyG graph safely
    with torch.serialization.safe_globals([Data]):
        data = torch.load(csv_path + '.pt', weights_only=False)

    meta = pickle.load(open(csv_path + '.meta.pkl','rb'))

    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    print("Nodes:", num_nodes, "Edges:", num_edges)

    y = data.y
    train_idx, val_idx, test_idx = split_edges(num_edges, 0.8, 0.1)

    # --------------------------
    # Compute class weights
    # --------------------------
    y_numpy = y.cpu().numpy()
    class_weights_np = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_numpy),
        y=y_numpy
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # 🔹 Initialize EdgeGNN model
    model = EdgeGNN(
        num_nodes=num_nodes,
        node_embed_dim=128,
        hidden_dim=256,
        num_classes=len(meta['label_encoder'].classes_)
    )
    model.to(device)
    data = data.to(device)
    y = y.to(device)

    # 🔹 Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # --------------------------
    # Training loop
    # --------------------------
    best_val = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits, _ = model(data)
        loss = criterion(logits[train_idx], y[train_idx])
        loss.backward()
        opt.step()

        # --------------------------
        # Train accuracy
        # --------------------------
        preds_train = logits[train_idx].argmax(dim=1).cpu().numpy()
        acc_train = accuracy_score(y[train_idx].cpu().numpy(), preds_train)

        # --------------------------
        # Validation
        # --------------------------
        model.eval()
        with torch.no_grad():
            logits_eval, _ = model(data)
            preds = logits_eval.argmax(dim=1).cpu().numpy()
            y_all = y.cpu().numpy()
            val_f1 = f1_score(y_all[val_idx], preds[val_idx], average='macro', zero_division=0)

            if val_f1 > best_val:
                best_val = val_f1
                best_state = model.state_dict()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch} loss {loss.item():.4f} "
                  f"trainAcc {acc_train:.4f} valF1 {val_f1:.4f}")

    # --------------------------
    # Load best model
    # --------------------------
    if best_state:
        model.load_state_dict(best_state)

    # --------------------------
    # Test metrics
    # --------------------------
    model.eval()
    with torch.no_grad():
        logits_final, node_emb = model(data)
        preds = logits_final.argmax(dim=1).cpu().numpy()
        y_all = y.cpu().numpy()

    print("\nTest classification report:")
    print(classification_report(
        y_all[test_idx],
        preds[test_idx],
        target_names=list(meta['label_encoder'].classes_)
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_all[test_idx], preds[test_idx]))

    # Save confusion matrix as image
    plot_confusion_matrix(
        y_true=y_all[test_idx],
        y_pred=preds[test_idx],
        labels=list(meta['label_encoder'].classes_),
        out_path=os.path.join(MAIN_DIR, "confusion_matrix.png")
    )

    # --------------------------
    # Save trained model weights
    # --------------------------
    torch.save(model.state_dict(), csv_path + '.edge_gnn.pt')
    print("Saved model:", csv_path + '.edge_gnn.pt')


# --------------------------
# Entry point
# --------------------------
if __name__ == '__main__':
    csv_path = os.path.join(MAIN_DIR, "data", "interaction_explanations.csv")
    main(csv_path=csv_path, epochs=100)





















# import pandas as pd
# import torch, pickle
# from sklearn.preprocessing import LabelEncoder
# from torch_geometric.data import Data
# import os
# import os

# MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(MAIN_DIR)

# # import pdb; pdb.set_trace()
# def build_graph_from_csv(csv_path, undirected=True, node_feature_type='learnable'):
#     df = pd.read_csv(csv_path)
#     df['Drug_A'] = df['Drug_A'].astype(str).str.strip()
#     df['Drug_B'] = df['Drug_B'].astype(str).str.strip()
#     assert set(['Drug_A','Drug_B','Level']) <= set(df.columns), "CSV must have Drug_A,Drug_B,Level"

#     drugs = sorted(set(df['Drug_A']).union(df['Drug_B']))
#     node2idx = {d:i for i,d in enumerate(drugs)}
#     idx2node = {i:d for d,i in node2idx.items()}

#     df['u'] = df['Drug_A'].map(node2idx)
#     df['v'] = df['Drug_B'].map(node2idx)

#     le = LabelEncoder()
#     df['label_idx'] = le.fit_transform(df['Level'].astype(str))

#     edge_u = []
#     edge_v = []
#     edge_labels = []
#     explanations = []
#     for _, row in df.iterrows():
#         u = int(row['u']); v = int(row['v'])
#         edge_u.append(u); edge_v.append(v)
#         edge_labels.append(int(row['label_idx']))
#         explanations.append(str(row.get('Explanation','')))
#         if undirected:
#             edge_u.append(v); edge_v.append(u)
#             edge_labels.append(int(row['label_idx']))
#             explanations.append(str(row.get('Explanation','')))

#     edge_index = torch.tensor([edge_u, edge_v], dtype=torch.long)
#     y = torch.tensor(edge_labels, dtype=torch.long)

#     # No explicit node features (we'll use learnable Embedding in model)
#     data = Data(x=None, edge_index=edge_index, edge_attr=None)
#     data.y = y
#     data.explanations = explanations
#     data.num_nodes = len(drugs)

#     meta = {'node2idx': node2idx, 'idx2node': idx2node, 'label_encoder': le, 'df': df}
#     torch.save(data, csv_path + '.pt')
#     with open(csv_path + '.meta.pkl', 'wb') as f:
#         pickle.dump(meta, f)
#     print(f"Saved graph: nodes={len(drugs)} edges={edge_index.size(1)} -> {csv_path}.pt")
#     return data, meta

# if __name__ == '__main__':
#     import sys
#     import os
    
#     csv_path = os.path.join(MAIN_DIR, "data", "interaction_explanations.csv")
#     # import pdb ; pdb.set_trace()
#     build_graph_from_csv(csv_path)