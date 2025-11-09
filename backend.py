import torch
import pandas as pd
import pickle
import os
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_geometric.data import Data
from fastapi.middleware.cors import CORSMiddleware
from models.edge_gnn import EdgeGNN

app = FastAPI()

# Fix CORS for Streamlit later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === PATHS ===
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(MAIN_DIR, "data", "final.csv")
DATA_PT = os.path.join(MAIN_DIR, "data", "balanced_drugs_data.csv.pt")
META_PATH = os.path.join(MAIN_DIR, "data", "balanced_drugs_data.csv.meta.pkl")
MODEL_PATH = os.path.join(MAIN_DIR, "data/output/final/edge_gnn_best.pt")

# === Load CSV ===
df = pd.read_csv(DATA_CSV)

# === Load model exactly like your working code ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.serialization.safe_globals([Data]):
    data = torch.load(DATA_PT, map_location=DEVICE)

with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

node2idx = meta["node2idx"]
label_encoder = meta["label_encoder"]

model_gnn = EdgeGNN(
    num_nodes=data.num_nodes,
    node_embed_dim=128,
    hidden_dim=256,
    num_classes=len(label_encoder.classes_)
).to(DEVICE)

model_gnn.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model_gnn.eval()
data = data.to(DEVICE)

# === BioMistral ===
device_llm = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    "BioMistral/BioMistral-7B",
    trust_remote_code=True,
    use_fast=False
)

model = AutoModelForCausalLM.from_pretrained(
    "BioMistral/BioMistral-7B",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)


def predict_with_gnn(drugA, drugB):
    if drugA not in node2idx or drugB not in node2idx:
        return "Unknown"

    src, dst = node2idx[drugA], node2idx[drugB]

    with torch.no_grad():
        x = model_gnn.node_emb(torch.arange(data.num_nodes, device=DEVICE))
        x = model_gnn.input_proj(x)
        for conv in model_gnn.convs:
            h = conv(x, data.edge_index)
            x = torch.relu(h) + x

        edge_feat = torch.cat([x[src].unsqueeze(0), x[dst].unsqueeze(0)], dim=1)
        pred = model_gnn.edge_mlp(edge_feat)
        pred_label = pred.argmax(dim=1).item()

    return label_encoder.inverse_transform([pred_label])[0]

# === Fingerprint interpretation ===
def interpret_fp(fps):
    keywords = []
    if max(fps) > 0.5: keywords.append("High reactive aromatic groups")
    if sum(fps) > 10: keywords.append("Multiple metabolic binding sites")
    if any(v < 0 for v in fps): keywords.append("Enzyme competition risks")
    return ", ".join(keywords) if keywords else "General chemical risks"

@app.get("/explain")
def explain_interaction(drugA: str, drugB: str):

    row = df[(df["Drug_A"] == drugA) & (df["Drug_B"] == drugB)]
    if row.empty:
        return {"error": "Drug pair not found in dataset"}

    smilesA = row["DrugA_SMILES"].values[0]
    smilesB = row["DrugB_SMILES"].values[0]
    fp_values = eval(row["FP_Attr_Values"].values[0])

    pred = predict_with_gnn(drugA, drugB)
    fp_keywords = interpret_fp(fp_values)

    prompt = f"""
Drug A: {drugA}
SMILES: {smilesA}

Drug B: {drugB}
SMILES: {smilesB}

Predicted Severity: {pred}

Fingerprints Suggest:
{fp_keywords}

Explain why these features cause interaction:
"""

    input_ids = tokenizer(prompt, return_tensors="pt").to(device_llm)
    output = model_llm.generate(input_ids["input_ids"], max_new_tokens=300)
    explanation = tokenizer.decode(output[0], skip_special_tokens=True)

    return {
        "severity": pred,
        "explanation": explanation
}
