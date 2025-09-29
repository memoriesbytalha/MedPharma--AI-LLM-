import torch, random, numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data
from models.edge_gnn import EdgeGNN
import pickle, os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
def main(csv_path, epochs, lr=0.005, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load the pre-saved PyG graph safely
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

    # Initialize EdgeGNN model
    model = EdgeGNN(
        num_nodes=num_nodes,
        node_embed_dim=128,
        hidden_dim=256,
        num_classes=len(meta['label_encoder'].classes_)
    )
    model.to(device)
    data = data.to(device)
    y = y.to(device)

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # --------------------------
    # For saving metrics per epoch
    # --------------------------
    train_accs = []
    val_f1s = []
    train_losses = []

    best_val = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits, _ = model(data)
        loss = criterion(logits[train_idx], y[train_idx])
        loss.backward()
        opt.step()

        # Train accuracy
        preds_train = logits[train_idx].argmax(dim=1).cpu().numpy()
        acc_train = accuracy_score(y[train_idx].cpu().numpy(), preds_train)

        # Validation
        model.eval()
        with torch.no_grad():
            logits_eval, _ = model(data)
            preds = logits_eval.argmax(dim=1).cpu().numpy()
            y_all = y.cpu().numpy()
            val_f1 = f1_score(y_all[val_idx], preds[val_idx], average='macro', zero_division=0)

            if val_f1 > best_val:
                best_val = val_f1
                best_state = model.state_dict()

        # Save metrics
        train_accs.append(acc_train)
        val_f1s.append(val_f1)
        train_losses.append(loss.item())

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch} loss {loss.item():.4f} "
                  f"trainAcc {acc_train:.4f} valF1 {val_f1:.4f}")

    # --------------------------
    # Save metrics to CSV
    # --------------------------
    metrics_df = pd.DataFrame({
        'epoch': list(range(1, epochs+1)),
        'train_loss': train_losses,
        'train_accuracy': train_accs,
        'val_f1': val_f1s
    })
    metrics_csv_path = os.path.join(MAIN_DIR, "training_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Training metrics saved as {metrics_csv_path}")

    # --------------------------
    # Plot metrics
    # --------------------------
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MAIN_DIR, "train_loss.png"))

    plt.figure()
    plt.plot(range(1, epochs+1), train_accs, label='Train Accuracy')
    plt.plot(range(1, epochs+1), val_f1s, label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Training Accuracy & Validation F1')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MAIN_DIR, "train_val_metrics.png"))

    print(f"Training loss and accuracy/F1 plots saved.")

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
    csv_path = os.path.join(MAIN_DIR, "data", "drugs_data.csv")
    main(csv_path=csv_path, epochs=200)
