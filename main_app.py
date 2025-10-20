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
from neo4j import GraphDatabase
import networkx as nx
import plotly.graph_objects as go
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from models.edge_gnn import EdgeGNN
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, DataStructs
from gpt4all import GPT4All
import pickle

from dotenv import load_dotenv
load_dotenv()  # Load .env variables

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
warnings.warn("deprecated", DeprecationWarning)





MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MAIN_DIR, "MED-PHARMA AI","data")
MODEL_PATH = os.path.join(DATA_DIR, "output", "final", "edge_gnn_best.pt")
DATA_PT = os.path.join(DATA_DIR, "balanced_drugs_data.csv.pt")
META_PATH = os.path.join(DATA_DIR, "balanced_drugs_data.csv.meta.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#------NEO4J Setup-------#
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
# ------Data Loading--------#

CSV_PATH = "data/drugs_data.csv"  # update with your path
df = pd.read_csv(CSV_PATH)
drug_list = sorted(list(set(df['Drug_A'].tolist() + df['Drug_B'].tolist())))


#---------------------------#
@st.cache_resource
def load_model_data():
    with torch.serialization.safe_globals([Data]):
        data = torch.load(DATA_PT, map_location=DEVICE)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    node2idx = meta["node2idx"]
    idx2node = meta["idx2node"]
    label_encoder = meta["label_encoder"]
    df = meta["df"]

    model = EdgeGNN(
        num_nodes=data.num_nodes,
        node_embed_dim=128,
        hidden_dim=256,
        num_classes=len(label_encoder.classes_)
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    return model, data, node2idx, idx2node, label_encoder, df


# === Prediction Function ===
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

# --------------------------
# MongoDB Setup
# --------------------------
client = MongoClient("mongodb://localhost:27017/")  # Update with your MongoDB URI
db = client['medpharmaI']
users = db['users']
# --------------------------

# Password hashing function
def hash_password(password):
    return pbkdf2_sha256.hash(password)

# Email validation function
def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)
# --------------------------
# Page Config & Styles
# --------------------------

st.set_page_config(
    page_title="MedPharma AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Define paths (update these with your actual paths)
BACKGROUND = Path("images/background.jpg")  # Relative path recommended

#---------Helper Functions---------#

def smiles_to_image(smiles, size=(350, 350)):
    mol = Chem.MolFromSmiles(smiles)
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    img_bytes = drawer.GetDrawingText()
    return Image.open(io.BytesIO(img_bytes))

def smiles_to_fp(smiles, radius=3, n_bits=512):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


device = 'cuda' if torch.cuda.is_available() else 'cpu'







# Cache SMILES → image
@st.cache_data
def smiles_to_image_cached(smiles, size=(350, 350)):
    return smiles_to_image(smiles, size)

# Cache SMILES → fingerprint
@st.cache_data
def smiles_to_fp_cached(smiles, radius=3, n_bits=512):
    return smiles_to_fp(smiles, radius, n_bits)

#----------------------------------#

def set_page_background(png_file):
    @st.cache_data()
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    
    try:
        bin_str = get_base64_of_bin_file(png_file)
    except FileNotFoundError:
        bin_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z/C/HgAGgwJ/lK3Q6wAAAABJRU5ErkJggg=="
    
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 20, 0.7)), url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}
    
    /* Input field styling */
    .stTextInput input, .stTextArea textarea {{
        color: #333333 !important;  /* Dark text color */
        background-color: rgba(255, 255, 255, 0.9) !important;
    }}
    
    /* Placeholder text */
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {{
        color: #666666 !important;
        opacity: 1 !important;
    }}
    
    /* Select dropdowns */
    .stSelectbox select {{
        color: #333333 !important;
        background-color: rgba(255, 255, 255, 0.9) !important;
    }}
    
    /* All other text elements */
    body, h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {{
        color: white !important;
    }}
    
    /* Navbar and other existing styles */
    [data-testid="stHeader"] {{
        background-color: rgba(0,0,0,0.5) !important;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_page_background(BACKGROUND)


    

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
    
    # Create tabs for Login and Registration
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
                # Validation checks
                if not all([new_username, new_email, new_password, confirm_password]):
                    st.error("Please fill all fields")
                elif new_password != confirm_password:
                    st.error("Passwords don't match")
                elif not is_valid_email(new_email):
                    st.error("Please enter a valid email address")
                elif users.find_one({"$or": [{"username": new_username}, {"email": new_email}]}):
                    st.error("Username or email already exists")
                else:
                    # Create new user document
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
    # First part - contact info (pure Streamlit)
    st.title("Contact Us")
    
    
    
    st.subheader("Direct Contacts")
    st.markdown("""
    📞 **Phone**: +92 317 5994687.
    📞 **Phone**: +92 332 1200260
    ✉️ **Email**: muhammadtalha7893@yahoo.com 
    ✉️ **Email**: zainulabadiennaqvi@gmail.com
    """)
    
    st.subheader("Office Hours")
    st.markdown("""
    - Monday - Friday: 9:00 AM - 5:00 PM  
    - Saturday: 10:00 AM - 2:00 PM  
    - Sunday: Closed
    """, unsafe_allow_html=True)
    
    # Second part - contact form (Streamlit form)
    
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
                # Insert into MongoDB
                db.contact.insert_one(user_data)
                st.success("Thank you for your message! We'll respond within 24 hours.")


def inference():
    st.title("💊 Drug Interaction Prediction")
    st.markdown("Enter two drugs to visualize their molecular structure and predict their interaction using the EdgeGNN model:")

    model, data, node2idx, idx2node, label_encoder, df = load_model_data()

    # Build dropdowns
    drug_list = sorted(list(set(df["Drug_A"]).union(set(df["Drug_B"]))))

    col1, col2 = st.columns(2)
    with col1:
        drug1 = st.selectbox("Drug 1", options=drug_list)
    with col2:
        drug2 = st.selectbox("Drug 2", options=drug_list)

    if st.button("🔍 Predict Interaction"):
        with st.spinner("Analyzing drug interaction..."):
            pair_df = df[((df['Drug_A'] == drug1) & (df['Drug_B'] == drug2)) |
                     ((df['Drug_A'] == drug2) & (df['Drug_B'] == drug1))]
            if pair_df.empty:
                st.warning(f"⚠️ No interaction data found for {drug1} and {drug2}.")
                return
            else:
                row = pair_df.iloc[0]

                # Assign SMILES + formulas
                if row["Drug_A"] == drug1:
                    smiles1, smiles2 = row["DrugA_SMILES"], row["DrugB_SMILES"]
                    formula1, formula2 = row["DrugA_Formula"], row["DrugB_Formula"]
                else:
                    smiles1, smiles2 = row["DrugB_SMILES"], row["DrugA_SMILES"]
                    formula1, formula2 = row["DrugB_Formula"], row["DrugA_Formula"]

                # Visualization
                st.markdown("### 🧪 Molecular Structures")
                c1, c2 = st.columns(2)
                with c1:
                    st.image(smiles_to_image_cached(smiles1), caption=f"{drug1}\nFormula: {formula1}")
                with c2:
                    st.image(smiles_to_image_cached(smiles2), caption=f"{drug2}\nFormula: {formula2}")
                

                # Find pair in CSV for molecular info
                result, error = predict_interaction(drug1, drug2, model, data, node2idx, label_encoder)
                st.success(f"💡 **Predicted Interaction:** {result['Predicted Interaction']} ({result['Confidence']*100:.2f}% confidence)")

                if error:
                    st.error(error)
                    return
        
        

        
        


    

# --------------------------
# Navigation & App Flow
# --------------------------

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "home"
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Create navbar
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
            st.session_state.show_register = True  # You'll need to modify login() to handle this

# Page routing
if st.session_state.page == "home":
    home()
elif st.session_state.page == "login":
    login()
elif st.session_state.page == "about":
    about()
elif st.session_state.page == "contact":
    contact()
elif st.session_state.page == "inference":
    inference()