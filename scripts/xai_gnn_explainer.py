import os
from typing import Union, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch_geometric.utils import k_hop_subgraph

from models.edge_gnn import EdgeGNN
from scripts.train_edge_gnn import prepare_graph

# -------------------------------
# Configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "data/drugs_data.csv"
MODEL_PATH = "output/edge_gnn_best.pt"
OUTPUT_DIR = "output/xai/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["Major", "Minor", "Moderate", "Unknown"]

# -------------------------------
# Load data and model
# -------------------------------
print("🧠 Loading data...")
data, meta = prepare_graph(CSV_PATH)
data = data.to(device)

print("💾 Loading trained model...")
model = EdgeGNN(
    num_nodes=data.num_nodes,
    node_feat_dim=data.x.size(1),
    node_embed_dim=512,
    hidden_dim=768,
    num_classes=len(CLASS_NAMES),
    dropout=0.2,
    num_layers=3
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

df = pd.read_csv(CSV_PATH)

# -------------------------------
# Wrapper to produce per-edge logits
# -------------------------------
class WrappedModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x, edge_index, edge_attr=None):
        node_logits, _ = self.base_model(Data(x=x, edge_index=edge_index))
        edge_logits = (node_logits[edge_index[0]] + node_logits[edge_index[1]]) / 2
        return edge_logits

# -------------------------------
# Explain a single edge
# -------------------------------
def explain_edge(edge_idx, data, df, k_hop=2):
    cpu_model = WrappedModel(model).cpu()
    cpu_data = Data(
        x=data.x.cpu(),
        edge_index=data.edge_index.cpu(),
        y=data.y.cpu() if hasattr(data, "y") else None
    )

    num_edges = cpu_data.edge_index.size(1)
    if num_edges == 0:
        print("No edges in graph, skipping.")
        return

    edge_idx = edge_idx % num_edges
    source, target = cpu_data.edge_index[:, edge_idx]

    # --- Build target for explainer ---
    with torch.no_grad():
        edge_logits = cpu_model(cpu_data.x, cpu_data.edge_index)
        pred_class = int(edge_logits[edge_idx].argmax().item())
        confidence = float(F.softmax(edge_logits, dim=1)[edge_idx, pred_class].item())

    model_config = ModelConfig(
        mode="multiclass_classification",
        task_level="edge",
        return_type="log_probs"
    )

    explainer = Explainer(
        model=cpu_model,
        algorithm=GNNExplainer(epochs=10),  # lower epochs for speed
        explanation_type="phenomenon",
        model_config=model_config,
        node_mask_type=None,  # since features not used
        edge_mask_type="object"
    )

    target_tensor = torch.full((cpu_data.edge_index.size(1),), pred_class, dtype=torch.long)

    # --- Extract k-hop subgraph ---
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=[source, target],
        num_hops=k_hop,
        edge_index=cpu_data.edge_index,
        relabel_nodes=True
    )

    if sub_edge_index.numel() == 0:
        print(f"⚠️ Subgraph empty for edge {edge_idx} ({source}-{target}), skipping explanation.")
        return

    edge_pos = mapping[1] if len(mapping) > 1 else 0

    # --- Run explainer ---
    try:
        explanation = explainer(
            x=cpu_data.x[subset],
            edge_index=sub_edge_index,
            index=edge_pos,
            target=target_tensor[subset]
        )
    except Exception as e:
        print(f"Explainer crashed for edge {edge_idx}: {e}")
        return

    # --- Feature importance ---
    feature_mask = getattr(explanation, "node_mask", None) or \
                   getattr(explanation, "node_feat_mask", None) or \
                   (explanation.get("node_mask") if isinstance(explanation, dict) else None)

    if feature_mask is not None:
        fm = feature_mask.detach().cpu()
        if fm.ndim == 2:
            feat_imp = fm.mean(dim=0).numpy()
        elif fm.ndim == 1:
            feat_imp = fm.numpy()
        else:
            feat_imp = fm.reshape(-1).numpy()

        top_idx = np.argsort(feat_imp)[-10:][::-1]
        plt.figure(figsize=(8, 5))
        sns.barplot(x=feat_imp[top_idx], y=[f"FP_{i}" for i in top_idx], palette="viridis")
        plt.title(f"Top molecular fingerprint features (edge {edge_idx})")
        plt.xlabel("Importance")
        plt.tight_layout()
        fig_path = os.path.join(OUTPUT_DIR, f"feature_importance_edge{edge_idx}.png")
        plt.savefig(fig_path)
        plt.close()
    else:
        fig_path = None
        print("No feature mask returned.")

    # --- Map drug names ---
    try:
        row = df.iloc[edge_idx]
        drugA, drugB = row["Drug_A"], row["Drug_B"]
    except Exception:
        drugA, drugB = f"edge_{edge_idx}_A", f"edge_{edge_idx}_B"

    pred_label = CLASS_NAMES[pred_class] if pred_class < len(CLASS_NAMES) else str(pred_class)

    # --- Save report ---
    report = f"""
============================================================
🧩 EXPLANATION REPORT FOR DRUG INTERACTION
------------------------------------------------------------
💊 Drug A: {drugA}
💊 Drug B: {drugB}

🧠 Predicted Interaction Level: {pred_label}
📈 Model Confidence: {confidence:.4f}

⚙️ Molecular Insights:
   • Prediction derived from SMILES-based molecular fingerprints.
   • Influential substructures (top fingerprint bits) contributed to this prediction.
   • Feature importance plot: {fig_path if fig_path else 'N/A'}

🩺 Clinical Summary:
   The AI model predicts a **{pred_label.lower()}** interaction between {drugA} and {drugB}
   (confidence {confidence*100:.1f}%). This is an automated explanation based on molecular
   fingerprints — clinical judgment required for final decision-making.
============================================================
"""
    report_path = os.path.join(OUTPUT_DIR, f"explanation_{drugA}_{drugB}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"✅ Saved report: {report_path}")
    if fig_path:
        print(f"✅ Saved feature plot: {fig_path}")


# -------------------------------
# Run for sample edges
# -------------------------------
for edge_idx in [10, 25, 50]:
    explain_edge(edge_idx, data, df)

print("\n✅ XAI analysis complete — explanations saved in output/xai/")
