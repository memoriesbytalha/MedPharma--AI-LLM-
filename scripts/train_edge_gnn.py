import os, random, pickle
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data
from models.edge_gnn import EdgeGNN

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL_DIR = os.path.join(MAIN_DIR, "data", "output", "final")
os.makedirs(FINAL_DIR, exist_ok=True)

# --------------------------
# Split edges into train/val/test
# --------------------------
def split_edges(num_edges, train_frac=0.8, val_frac=0.1, seed=123):
    idx = np.arange(num_edges)
    np.random.seed(seed)
    np.random.shuffle(idx)
    n_train = int(train_frac * num_edges)
    n_val = int(val_frac * num_edges)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

# --------------------------
# Early Stopping
# --------------------------
class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4, save_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f"🟡 EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        if self.save_path:
            torch.save(model.state_dict(), self.save_path)
            print(f"✅ Validation improved → Saving best model to {self.save_path}")

# --------------------------
# Plot confusion matrix
# --------------------------
def plot_confusion_matrix(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Confusion matrix saved as {out_path}")

# --------------------------
# Plot training curves
# --------------------------
def plot_training_curves(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "training_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, history["val_f1"], label="Val F1", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Validation F1")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "validation_f1.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "train_val_accuracy.png"))
    plt.close()

    print(f"✅ Training graphs saved in {out_dir}")

# --------------------------
# Main training
# --------------------------
def main(csv_path, epochs=300, lr=0.005, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    with torch.serialization.safe_globals([Data]):
        data = torch.load(csv_path + '.pt', weights_only=False)
    meta = pickle.load(open(csv_path + '.meta.pkl', 'rb'))

    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    print(f"Nodes: {num_nodes} | Edges: {num_edges}")

    y = data.y
    train_idx, val_idx, test_idx = split_edges(num_edges, 0.8, 0.1)

    y_numpy = y.cpu().numpy()
    class_weights_np = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_numpy),
        y=y_numpy
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    model = EdgeGNN(
        num_nodes=num_nodes,
        node_embed_dim=128,
        hidden_dim=256,
        num_classes=len(meta['label_encoder'].classes_),
    ).to(device)

    data = data.to(device)
    y = y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val = 0.0
    best_state = None
    history = {"train_loss": [], "val_f1": [], "train_acc": [], "val_acc": []}

    early_stopper = EarlyStopping(
        patience=20,
        min_delta=1e-4,
        save_path=os.path.join(FINAL_DIR, "edge_gnn_best.pt")
    )

    # --------------------------
    # Training Loop
    # --------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits, _ = model(data)
        loss = criterion(logits[train_idx], y[train_idx])
        loss.backward()
        opt.step()

        preds_train = logits[train_idx].argmax(dim=1).cpu().numpy()
        y_train_np = y[train_idx].cpu().numpy()
        acc_train = accuracy_score(y_train_np, preds_train)

        model.eval()
        with torch.no_grad():
            logits_eval, _ = model(data)
            preds_eval = logits_eval.argmax(dim=1).cpu().numpy()
            y_val_np = y[val_idx].cpu().numpy()
            val_f1 = f1_score(y_val_np, preds_eval[val_idx],
                              average='macro', zero_division=0)
            val_acc = accuracy_score(y_val_np, preds_eval[val_idx])

        history["train_loss"].append(loss.item())
        history["val_f1"].append(val_f1)
        history["train_acc"].append(acc_train)
        history["val_acc"].append(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
                  f"TrainAcc: {acc_train:.4f} | ValAcc: {val_acc:.4f} | ValF1: {val_f1:.4f}")

        # Early stopping check
        early_stopper(val_f1, model)
        if early_stopper.early_stop:
            print(f"🛑 Early stopping triggered at epoch {epoch}")
            break

    # --------------------------
    # Load best model
    # --------------------------
    model.load_state_dict(torch.load(os.path.join(FINAL_DIR, "edge_gnn_best.pt")))
    print(f"✅ Loaded best model for testing.")

    # --------------------------
    # Testing
    # --------------------------
    model.eval()
    with torch.no_grad():
        logits_final, _ = model(data)
        preds = logits_final.argmax(dim=1).cpu().numpy()
        y_all = y.cpu().numpy()

    labels = list(meta['label_encoder'].classes_)
    report_dict = classification_report(
        y_all[test_idx],
        preds[test_idx],
        target_names=labels,
        output_dict=True,
        zero_division=0
    )

    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv(os.path.join(FINAL_DIR, "classification_report.csv"))
    print(f"✅ Classification report saved at {os.path.join(FINAL_DIR, 'classification_report.csv')}")

    plot_confusion_matrix(
        y_true=y_all[test_idx],
        y_pred=preds[test_idx],
        labels=labels,
        out_path=os.path.join(FINAL_DIR, "confusion_matrix.png")
    )

    plot_training_curves(history, FINAL_DIR)

    print("\n📊 Final Test Report:")
    print(classification_report(y_all[test_idx], preds[test_idx], target_names=labels))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_all[test_idx], preds[test_idx]))


if __name__ == "__main__":
    csv_path = os.path.join(MAIN_DIR, "data", "balanced_drugs_data.csv")
    main(csv_path=csv_path, epochs=300)

