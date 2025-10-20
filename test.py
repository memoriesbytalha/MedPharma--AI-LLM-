import os
import torch
import pickle
import streamlit as st
import pandas as pd
from torch_geometric.data import Data
from models.edge_gnn import EdgeGNN

# === Paths ===
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MAIN_DIR,"MED-PHARMA AI", "data")
MODEL_PATH = os.path.join(DATA_DIR, "output", "final", "edge_gnn_best.pt")
DATA_PT = os.path.join(DATA_DIR, "balanced_drugs_data.csv.pt")
META_PATH = os.path.join(DATA_DIR, "balanced_drugs_data.csv.meta.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Model & Metadata ===
@st.cache_resource
def load_model_and_data():
    with torch.serialization.safe_globals([Data]):
        data = torch.load(DATA_PT, map_location=DEVICE)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    node2idx = meta["node2idx"]
    idx2node = meta["idx2node"]
    label_encoder = meta["label_encoder"]

    model = EdgeGNN(
        num_nodes=data.num_nodes,
        node_embed_dim=128,
        hidden_dim=256,
        num_classes=len(label_encoder.classes_)
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model, data.to(DEVICE), node2idx, idx2node, label_encoder

model, data, node2idx, idx2node, label_encoder = load_model_and_data()

# === Prediction Function ===
def predict_interaction(drug_a, drug_b):
    drug_a, drug_b = drug_a.strip(), drug_b.strip()
    if not drug_a or not drug_b:
        return None, "⚠️ Please enter both drug names."

    if drug_a not in node2idx:
        return None, f"❌ Drug '{drug_a}' not found in dataset."
    if drug_b not in node2idx:
        return None, f"❌ Drug '{drug_b}' not found in dataset."

    src, dst = node2idx[drug_a], node2idx[drug_b]

    with torch.no_grad():
        x = model.node_emb(torch.arange(data.num_nodes, device=DEVICE))
        x = model.input_proj(x)
        for conv in model.convs:
            h = conv(x, data.edge_index)
            x = torch.relu(h) + x
        edge_feat = torch.cat([x[src].unsqueeze(0), x[dst].unsqueeze(0)], dim=1)
        pred = model.edge_mlp(edge_feat)
        probs = torch.softmax(pred, dim=1).cpu().numpy()[0]
        pred_label = pred.argmax(dim=1).item()
        class_name = label_encoder.inverse_transform([pred_label])[0]

    result = {
        "Predicted Interaction": class_name,
        "Probabilities": dict(zip(label_encoder.classes_, probs.round(3)))
    }
    return result, None

# === Streamlit UI ===
st.set_page_config(page_title="MedPharma AI - Drug Interaction Predictor", layout="centered")
st.title("💊 MedPharma AI - Drug Interaction Predictor")
st.write("Enter two drug names to predict their interaction using your trained **EdgeGNN model**.")

col1, col2 = st.columns(2)
drug_a = col1.text_input("Drug A", placeholder="e.g. Aspirin")
drug_b = col2.text_input("Drug B", placeholder="e.g. Paracetamol")

if st.button("🔍 Predict Interaction"):
    with st.spinner("Analyzing drug interaction..."):
        result, error = predict_interaction(drug_a, drug_b)
        if error:
            st.error(error)
        else:
            st.success(f"💡 Predicted Interaction: **{result['Predicted Interaction']}**")

            # Show probabilities
            probs_df = pd.DataFrame(
                {"Interaction Type": list(result["Probabilities"].keys()),
                 "Probability": list(result["Probabilities"].values())}
            )
            st.bar_chart(probs_df.set_index("Interaction Type"))

st.markdown("---")
st.caption("Developed by Muhammad Talha | MedPharma AI | Powered by EdgeGNN")

