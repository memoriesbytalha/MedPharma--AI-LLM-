import base64
import streamlit as st
from pathlib import Path
import numpy as np
from pymongo import MongoClient
from passlib.hash import pbkdf2_sha256
import re
import datetime
import pandas as pd
import os
import warnings
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
from rdkit.Chem.Draw import rdMolDraw2D
import io
import torch
from torch_geometric.data import Data
from models.edge_gnn import EdgeGNN
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import pickle
import os
import requests
from dotenv import load_dotenv
import pickle
import torch
from torch_geometric.data import Data
import streamlit as st
from utils.bots import *
from utils.explainer import *
import sys
from utils.background import *
sys.path.append(os.getcwd())
load_dotenv()  # Load .env variables

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
warnings.warn("deprecated", DeprecationWarning)

OPENROUTER_API_KEY = os.getenv('OPEN_API_ROUTER_KEY')

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MAIN_DIR, "MED-PHARMA AI", "data")
MODEL_PATH = os.path.join(DATA_DIR, "output", "final", "edge_gnn_best.pt")
DATA_PT = os.path.join(DATA_DIR, "balanced_drugs_data.csv.pt")
META_PATH = os.path.join(DATA_DIR, "balanced_drugs_data.csv.meta.pkl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = os.path.join(DATA_DIR, "balanced_drugs_data.csv")
# import pdb;pdb.set_trace()
# ------Data Loading--------#
df = pd.read_csv(dataset_path)
drug_list = sorted(list(set(df['Drug_A'].tolist() + df['Drug_B'].tolist())))



# ==================== CACHED LOADING ====================

@st.cache_resource
def load_model_data():
    # ✅ Register safe globals for torch.load
    torch.serialization.add_safe_globals([Data])

    # Load graph data
    data = torch.load(DATA_PT, map_location=DEVICE, weights_only=False)

    # Load metadata
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    node2idx = meta["node2idx"]
    idx2node = meta["idx2node"]
    label_encoder = meta["label_encoder"]
    df = meta["df"]

    # Initialize model
    model = EdgeGNN(
        num_nodes=data.num_nodes,
        node_embed_dim=128,
        hidden_dim=256,
        num_classes=len(label_encoder.classes_)
    ).to(DEVICE)

    # Load model weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    return model, data, node2idx, idx2node, label_encoder, df


@st.cache_resource
def load_pathway_explainer():
    """Cache the Pathway Explainer instance"""
    model, data, node2idx, idx2node, label_encoder, df = load_model_data()
    explainer = PathwayExplainer(model, data, node2idx, idx2node, label_encoder, DEVICE)
    return explainer

@st.cache_resource
def load_explainer():
    """Cache the GNN Explainer instance"""
    model, data, node2idx, idx2node, label_encoder, df = load_model_data()
    explainer = GNNExplainer(model, data, node2idx, idx2node, DEVICE)
    return explainer

# ==================== PREDICTION FUNCTION ====================

def predict_interaction(drug_a, drug_b, model, data, node2idx, label_encoder):
    drug_a, drug_b = drug_a.strip(), drug_b.strip()

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
        confidence = float(probs[pred_label])

    result = {
        "Predicted Interaction": class_name,
        "Confidence": confidence,
    }

    return result, None

# ==================== MONGODB & AUTH ====================

client = MongoClient("mongodb://localhost:27017/")
db = client['medpharmaai']
users = db['users']

def hash_password(password):
    return pbkdf2_sha256.hash(password)

def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

# ==================== STYLING ====================

st.set_page_config(
    page_title="MedPharma AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BACKGROUND = Path(r"images\1.jpeg")
set_page_background(BACKGROUND)
def smiles_to_image(smiles, size=(350, 350)):
    mol = Chem.MolFromSmiles(smiles)
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    img_bytes = drawer.GetDrawingText()
    return Image.open(io.BytesIO(img_bytes))

@st.cache_data
def smiles_to_image_cached(smiles, size=(350, 350)):
    return smiles_to_image(smiles, size)
# ==================== PAGES ====================

def home():
    st.markdown("""
    <div style="height: 100vh; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
        <h2 style="font-size: 3.5rem; margin-bottom: 1rem;">Bridging the Gap Between General Physicians and Pharmacologists</h2>
        <h2 style="font-size: 1.8rem; margin-bottom: 2rem;">From Molecules to Medicine — Know Before You Prescribe.</h2>
        <p style="font-size: 1.2rem; max-width: 800px; margin-bottom: 2rem;">
        Our AI-powered platform provides clear explanations, interactive graphs, and insights to support clinical decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

def login():
    st.title("🔐 Authentication")
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                user = users.find_one({"username": username})
                if user and pbkdf2_sha256.verify(password, user["password"]):
                    st.session_state.logged_in = True
                    st.session_state.user = username
                    st.success("Logged in successfully!")
                    st.session_state.page = "inference"
                    st.rerun()
                else:
                    st.error("Invalid credentials")

    with tab2:
        with st.form("registration_form"):
            st.subheader("Create New Account")
            new_username = st.text_input("Choose Username")
            new_email = st.text_input("Email Address")
            new_password = st.text_input("Create Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submitted_reg = st.form_submit_button("Register")
            
            if submitted_reg:
                if not all([new_username, new_email, new_password, confirm_password]):
                    st.error("Please fill all fields")
                elif new_password != confirm_password:
                    st.error("Passwords don't match")
                elif not is_valid_email(new_email):
                    st.error("Please enter a valid email address")
                elif users.find_one({"$or": [{"username": new_username}, {"email": new_email}]}):
                    st.error("Username or email already exists")
                else:
                    user_data = {
                        "username": new_username,
                        "email": new_email,
                        "password": hash_password(new_password),
                        "created_at": datetime.datetime.now()
                    }
                    users.insert_one(user_data)
                    st.success("Account created successfully! Please login.")

def about():
    st.title("📖 About Us")
    st.markdown("""
    We are final year students from **Bahria University Islamabad**, developing **MedPharma AI** — an intelligent platform for predicting and explaining **drug-drug interactions**.

    Our solution leverages **graph neural networks, molecular fingerprints, and explainable AI (XAI)** to assist healthcare professionals in making informed decisions safely and efficiently.

    The platform features a **dynamic, interactive interface**, offering visual drug interaction graphs, confidence-based predictions, and AI-powered explanations to support clinical judgment.
    """)

def contact():
    st.title("Contact Us")
    st.subheader("Direct Contacts")
    st.markdown("""
    📞 **Phone**: +92 317 5994687
    📞 **Phone**: +92 332 1200260
    ✉️ **Email**: muhammadtalha7893@yahoo.com 
    ✉️ **Email**: zainulabadiennaqvi@gmail.com
    """)
    
    st.subheader("Office Hours")
    st.markdown("""
    - Monday - Friday: 9:00 AM - 5:00 PM  
    - Saturday: 10:00 AM - 2:00 PM  
    - Sunday: Closed
    """)
    
    st.subheader("Send Us a Message")
    with st.form("contact_form"):
        name = st.text_input("Your Name*")
        email = st.text_input("Email Address*")
        subject = st.text_input("Subject")
        message = st.text_area("Your Message*")
        submitted = st.form_submit_button("Send Message")
        
        if submitted:
            if not name or not email or not message:
                st.error("Please fill in all required fields (*)")
            else:
                user_data = {
                    "username": name,
                    "email": email,
                    "subject": subject,
                    "message": message,
                }
                db.contact.insert_one(user_data)
                st.success("Thank you for your message! We'll respond within 24 hours.")

def inference_with_explanation():
    """Enhanced inference with Pathway Explainer"""
    
    st.title("💊 Drug Interaction Prediction with Pathway Analysis")
    st.markdown("Enter two drugs to visualize their molecular structure, predict interaction, and see **connecting pathways**:")

    model, data, node2idx, idx2node, label_encoder, df = load_model_data()
    pathway_explainer = load_pathway_explainer()
    
    drug_list = sorted(list(set(df["Drug_A"]).union(set(df["Drug_B"]))))
    col1, col2 = st.columns(2)
    with col1:
        drug1 = st.selectbox("Drug 1", options=drug_list, key="drug1_select")
    with col2:
        drug2 = st.selectbox("Drug 2", options=drug_list, key="drug2_select")

    # Explainer settings
    with st.expander("⚙️ Pathway Analysis Settings"):
        max_paths = st.slider("Number of pathways to show", 3, 10, 5,
                             help="How many connection pathways to display")
        max_length = st.slider("Maximum pathway length", 2, 5, 4,
                              help="Maximum number of intermediate drugs in a pathway")

    if st.button("🔍 Predict Interaction & Analyze Pathways", type="primary"):
        with st.spinner("Analyzing drug interaction and pathways..."):
            pair_df = df[((df['Drug_A'] == drug1) & (df['Drug_B'] == drug2)) |
                         ((df['Drug_A'] == drug2) & (df['Drug_B'] == drug1))]

            if pair_df.empty:
                st.warning(f"⚠️ No interaction data found for {drug1} and {drug2}.")
                return

            row = pair_df.iloc[0]

            # Assign SMILES + formulas
            if row["Drug_A"] == drug1:
                smiles1, smiles2 = row["DrugA_SMILES"], row["DrugB_SMILES"]
                formula1, formula2 = row["DrugA_Formula"], row["DrugB_Formula"]
            else:
                smiles1, smiles2 = row["DrugB_SMILES"], row["DrugA_SMILES"]
                formula1, formula2 = row["DrugB_Formula"], row["DrugA_Formula"]

            # Molecular visualization
            st.markdown("### 🧪 Molecular Structures")
            c1, c2 = st.columns(2)
            with c1:
                st.image(smiles_to_image_cached(smiles1), caption=f"{drug1}\nFormula: {formula1}")
            with c2:
                st.image(smiles_to_image_cached(smiles2), caption=f"{drug2}\nFormula: {formula2}")

            # Predict interaction
            result, error = predict_interaction(drug1, drug2, model, data, node2idx, label_encoder)
            predicted = result.get("Predicted Interaction")

            if predicted == "Unknown":
                st.info("⚠️ Interaction is unknown. This may be updated when more data becomes available.")
            else:
                st.success(f"✅ Predicted Interaction: {predicted}")
                
                predicted_interaction = result['Predicted Interaction']
                confidence = result['Confidence']

                st.success(f"💡 **Predicted Interaction:** {predicted_interaction} ({confidence*100:.2f}% confidence)")

                # ====== PATHWAY EXPLAINER ======
                st.markdown("---")
                st.markdown("### 🛤️ Interaction Pathway Analysis")

                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Finding pathways between drugs...")
                progress_bar.progress(30)

                explanation = pathway_explainer.explain_pathways(
                    drug1, drug2,
                    max_paths=max_paths,
                    max_path_length=max_length
                )

                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()

                if 'error' in explanation:
                    st.error(f"⚠️ {explanation['error']}")
                    return

                if 'warning' in explanation:
                    st.warning(f"⚠️ {explanation['warning']}")
                    st.info("💡 The drugs may not have direct or indirect connections in the knowledge graph. The prediction is based on learned patterns from similar drug structures.")
                    return

                # Display summary
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0;'>
                    <h3 style='color: white; margin: 0;'>📊 Pathway Summary</h3>
                    <p style='color: white; font-size: 18px; margin: 10px 0;'>
                        <strong>Query:</strong> {explanation['drug_a']} + {explanation['drug_b']}<br>
                        <strong>Prediction:</strong> {explanation['prediction']} ({explanation['confidence']*100:.1f}% confidence)<br>
                        <strong>Total Pathways Found:</strong> {explanation['total_paths_found']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Display pathways
                st.markdown("#### 🔗 Top Connecting Pathways")

                for pathway in explanation['pathways']:
                    with st.expander(f"**Pathway {pathway['rank']}** | Importance: {pathway['importance']:.3f} | Length: {pathway['length']} hops", expanded=(pathway['rank'] <= 2)):
                        st.markdown(f"**Path:** `{pathway['path']}`")
                        st.markdown(f"**Explanation:** {pathway['explanation']}")

                        # Color-code intermediate drugs
                        if pathway['length'] > 1:
                            st.markdown("**Intermediate Drugs:**")
                            intermediate = pathway['path_drugs'][1:-1]
                            cols = st.columns(len(intermediate))
                            for idx, drug in enumerate(intermediate):
                                with cols[idx]:
                                    st.markdown(f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 5px; text-align: center;'>{drug}</div>", unsafe_allow_html=True)
                layman_chain = create_explanation_bot(OPENROUTER_API_KEY)

                layman_text = generate_layman_explanation_with_formula(layman_chain, {
                    "drug_a": drug1,
                    "drug_b": drug2,
                    "drug_a_formula": formula1,
                    "drug_b_formula": formula2,
                    "prediction": predicted_interaction,
                    "confidence": confidence,
                    "pathways": explanation.get('pathways', [])
                })

                st.markdown("### 📝 Layman-Friendly Explanation")
                st.info(layman_text)


                # Technical details
                with st.expander("📊 Technical Details"):
                    st.write(f"**Drug A SMILES:** {smiles1}")
                    st.write(f"**Drug B SMILES:** {smiles2}")
                    st.write(f"**Prediction Confidence:** {confidence*100:.2f}%")
                    st.write(f"**Total Pathways Discovered:** {explanation['total_paths_found']}")
                    st.write(f"**Pathways Shown:** {len(explanation['pathways'])}")

                    st.markdown("**Full Pathway Data:**")
                    st.json(explanation)

# ==================== NAVIGATION ====================

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "home"
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False


cols = st.columns([2,1,1,1,1])
with cols[0]:
    if st.button("🏠 MedPharma AI"):
        st.session_state.page = "home"
with cols[1]:
    if st.button("About"):
        st.session_state.page = "about"
with cols[2]:
    if st.button("Contact"):
        st.session_state.page = "contact"
with cols[3]:
    if st.button("Inference"):
        st.session_state.page = "inference" if st.session_state.logged_in else "login"
with cols[4]:
    if st.session_state.logged_in:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.page = "home"
            st.rerun()
    else:
        if st.button("Register"):
            st.session_state.page = "login"
# nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1.5, 1, 1, 1, 1])

# with nav_col1:
#     if st.button("🏠 MedPharma AI", use_container_width=True, key="nav_home"):
#         st.session_state.page = "home"
#         st.rerun()

# with nav_col2:
#     if st.button("📖 About", use_container_width=True, key="nav_about"):
#         st.session_state.page = "about"
#         st.rerun()

# with nav_col3:
#     if st.button("📞 Contact", use_container_width=True, key="nav_contact"):
#         st.session_state.page = "contact"
#         st.rerun()

# with nav_col4:
#     if st.button("🔬 Analysis", use_container_width=True, key="nav_analysis"):
#         st.session_state.page = "inference" if st.session_state.logged_in else "login"
#         st.rerun()

# with nav_col5:
#     if st.session_state.logged_in:
#         nav_chat, nav_logout = st.columns(2)
#         with nav_chat:
#             if st.button("💬 Chat", use_container_width=True, key="nav_chat"):
#                 st.session_state.page = "chatbot"
#                 st.rerun()
#         with nav_logout:
#             if st.button("🚪 Logout", use_container_width=True, key="nav_logout"):
#                 st.session_state.logged_in = False
#                 st.session_state.user = None
#                 st.session_state.page = "home"
#                 st.rerun()
#     else:
#         if st.button("🔐 Login", use_container_width=True, key="nav_login"):
#             st.session_state.page = "login"
# Page routing
if st.session_state.page == "home":
    home()
elif st.session_state.page == "login":
    login()
elif st.session_state.page == "about":
    about()
elif st.session_state.page == "contact":
    contact()
elif st.session_state.page == "chatbot":
    pharmacologist_chatbot()
elif st.session_state.page == "inference":
    inference_with_explanation()