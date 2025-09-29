# frontend_dashboard.py
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import io
import difflib
import os

st.set_page_config(page_title="MedPharma AI — Interaction Explorer", layout="wide")

# ---------- Helpers ----------
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def mol_image_from_smiles(smiles, size=(250,250)):
    if not smiles or pd.isna(smiles) or str(smiles).strip() == "":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Image.open(buf)

def top_matches(query, choices, n=10):
    if not query:
        return sorted(choices)[:n]
    q = query.lower()
    starts = [c for c in choices if c.lower().startswith(q)]
    if len(starts) >= n:
        return starts[:n]
    fuzzy = difflib.get_close_matches(query, choices, n*2)
    combined = []
    for c in starts + fuzzy + choices:
        if c not in combined:
            combined.append(c)
        if len(combined) >= n:
            break
    return combined

def safe_get(df, drug_col, smiles_col, formula_col, drug):
    """Safely return (SMILES, Formula) for a given drug without crashing"""
    smi, formula = None, ""
    if drug and drug_col in df.columns:
        sub = df[df[drug_col] == drug]
        if smiles_col in df.columns and not sub.empty:
            vals = sub[smiles_col].dropna()
            if not vals.empty:
                smi = vals.iloc[0]
        if formula_col in df.columns and not sub.empty:
            vals = sub[formula_col].dropna()
            if not vals.empty:
                formula = vals.iloc[0]
    return smi, formula

# ---------- Load dataset ----------
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(MAIN_DIR, "data", "drugs_data.csv")   # change if needed
if not os.path.exists(DATA_PATH):
    st.error(f"CSV not found at {DATA_PATH}. Put drugs_data.csv in data/ or edit DATA_PATH.")
    st.stop()

df = pd.read_csv(DATA_PATH)

drugA_col = find_col(df, ['Drug_A','DrugA','DrugA_Name','DrugA_ID','Drug_A_Name'])
drugB_col = find_col(df, ['Drug_B','DrugB','DrugB_Name','DrugB_ID','Drug_B_Name'])
smilesA_col = find_col(df, ['DrugA_SMILES','Drug_A_SMILES','SMILES_A','SMILES'])
smilesB_col = find_col(df, ['DrugB_SMILES','Drug_B_SMILES','SMILES_B'])
formulaA_col = find_col(df, ['DrugA_Formula','Drug_A_Formula','Formula_A','Formula'])
formulaB_col = find_col(df, ['DrugB_Formula','Drug_B_Formula','Formula_B'])
level_col = find_col(df, ['Level','Interaction_Level','level'])
# explain_col = find_col(df, ['Interaction_Explanation','Explanation','explanation','interaction_explanation'])

if drugA_col is None and 'Drug' in df.columns:
    drugA_col = 'Drug'
if drugB_col is None and 'Drug' in df.columns:
    drugB_col = 'Drug'
if level_col is None:
    st.error("Could not find Level/Interaction column in CSV. Rename your column to 'Level' or 'Interaction_Level'.")
    st.stop()

names_a = df[drugA_col].astype(str).tolist() if drugA_col in df.columns else []
names_b = df[drugB_col].astype(str).tolist() if drugB_col in df.columns else []
drug_names = sorted(list(set([n.strip() for n in names_a + names_b if str(n).strip() != ""])))

# ---------- Sidebar ----------
st.sidebar.title("MedPharma AI — Explorer")
st.sidebar.markdown("Search two drugs (autocomplete). App shows interaction + chemical info.")
st.sidebar.write(f"Dataset: `{os.path.basename(DATA_PATH)}`")
st.sidebar.write(f"Drugs: {len(drug_names)}")
st.sidebar.write(f"Interactions (rows): {len(df)}")

# ---------- Main layout ----------
col1, col2 = st.columns([1,2])

with col1:
    st.header("Query drugs")
    q1 = st.text_input("Drug A (type to search)", "")
    matches1 = top_matches(q1, drug_names, n=20)
    sel_a = st.selectbox("Suggestions (Drug A)", options=matches1, index=0 if matches1 else -1)

    q2 = st.text_input("Drug B (type to search)", "")
    matches2 = top_matches(q2, drug_names, n=20)
    sel_b = st.selectbox("Suggestions (Drug B)", options=matches2, index=0 if matches2 else -1)

    if st.button("Check interaction"):
        if not sel_a or not sel_b:
            st.warning("Please select both drugs.")
        else:
            st.session_state['sel_a'] = sel_a
            st.session_state['sel_b'] = sel_b

with col2:
    st.header("Selected pair")
    sel_a = st.session_state.get('sel_a', None)
    sel_b = st.session_state.get('sel_b', None)
    if sel_a and sel_b:
        st.subheader(f"{sel_a}  ↔  {sel_b}")
        left, right = st.columns(2)
        with left:
            st.markdown("**Drug A**")
            smi_a, form_a = safe_get(df, drugA_col, smilesA_col, formulaA_col, sel_a)
            if smi_a:
                img = mol_image_from_smiles(smi_a)
                if img: st.image(img, caption=f"SMILES: {smi_a}", use_column_width=False)
            st.write("Formula:", form_a)
        with right:
            st.markdown("**Drug B**")
            smi_b, form_b = safe_get(df, drugB_col, smilesB_col, formulaB_col, sel_b)
            if smi_b:
                img = mol_image_from_smiles(smi_b)
                if img: st.image(img, caption=f"SMILES: {smi_b}", use_column_width=False)
            st.write("Formula:", form_b)

        cond1 = (df[drugA_col] == sel_a) & (df[drugB_col] == sel_b)
        cond2 = (df[drugA_col] == sel_b) & (df[drugB_col] == sel_a)
        match_rows = df[cond1 | cond2]
        st.divider()
        if not match_rows.empty:
            st.success("✅ Interaction found in dataset")
            levels = match_rows[level_col].unique().tolist()
            st.write("Level(s):", ", ".join(map(str, levels)))
            st.write("Source rows:", match_rows.shape[0])
        else:
            st.warning("No direct interaction found in the dataset for this pair.")

        df_a = df[(df[drugA_col] == sel_a) | (df[drugB_col] == sel_a)]
        df_b = df[(df[drugA_col] == sel_b) | (df[drugB_col] == sel_b)]

        neighbors_a = set(df_a[drugA_col].tolist() + df_a[drugB_col].tolist())
        neighbors_b = set(df_b[drugA_col].tolist() + df_b[drugB_col].tolist())
        neighbors_a.discard(sel_a); neighbors_a.discard("")
        neighbors_b.discard(sel_b); neighbors_b.discard("")

        st.divider()
        st.subheader("Neighbors")
        coln1, coln2, coln3 = st.columns([1,1,1])
        with coln1:
            st.write(f"Neighbors of {sel_a} (count {len(neighbors_a)}):")
            st.write(", ".join(list(neighbors_a)[:30]))
        with coln2:
            st.write(f"Neighbors of {sel_b} (count {len(neighbors_b)}):")
            st.write(", ".join(list(neighbors_b)[:30]))
        with coln3:
            common = neighbors_a.intersection(neighbors_b)
            st.write(f"Common neighbors (count {len(common)}):")
            st.write(", ".join(list(common)[:30]))

        st.divider()
        st.subheader("Local interaction subgraph")
        G = nx.Graph()
        keep = set([sel_a, sel_b]) | set(list(neighbors_a)[:30]) | set(list(neighbors_b)[:30])
        for _, r in df.iterrows():
            a = r[drugA_col]; b = r[drugB_col]
            if a in keep and b in keep:
                G.add_node(a); G.add_node(b)
                lvl = r[level_col] if level_col in df.columns else ""
                G.add_edge(a, b, level=lvl)
        if G.number_of_nodes() == 0:
            st.info("Not enough data to plot subgraph.")
        else:
            plt.figure(figsize=(8,6))
            pos = nx.spring_layout(G, k=0.5, seed=42)
            levels = nx.get_edge_attributes(G, 'level')
            edge_colors = []
            for u,v in G.edges():
                lvl = levels.get((u,v), "")
                if str(lvl).lower().startswith("maj"):
                    edge_colors.append('red')
                elif str(lvl).lower().startswith("min"):
                    edge_colors.append('green')
                else:
                    edge_colors.append('orange')
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=600)
            nx.draw_networkx_labels(G, pos, font_size=8)
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors)
            st.pyplot(plt)
    else:
        st.info("Select two drugs and press 'Check interaction' to view details.")

# ---------- Footer ----------
st.markdown("---")
st.markdown("MedPharma AI — interactive demo. Shows chemical structures (SMILES), formulas and interaction details from your merged CSV.")



