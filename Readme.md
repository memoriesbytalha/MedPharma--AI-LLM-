# Med-Pharma AI

<img src="data\output\final\confusion_matrix.png" alt="Confusion Matrix" width="600"/>
<img src="data\output\final\train_val_accuracy.png" alt="Train Val accuracy" width="600"/>
<img src="data\output\final\training_loss.png" alt="Train Loss" width="600"/>
<img src="data\output\final\validation_f1.png" alt="Val F1 accuracy" width="600"/>

# These are declared but NEVER used:
NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD  # No Neo4j queries anywhere
driver = GraphDatabase.driver(...)      # Connection never used
device = 'cuda' if ...                  # Redundant (DEVICE already defined)
```

---

## 🗂️ File Structure (Implied)
```
project/
├── app.py                          # Main Streamlit app (your code)
├── models/
│   └── edge_gnn.py                 # EdgeGNN model definition
├── data/
│   ├── balanced_drugs_data.csv     # Drug interaction dataset
│   ├── balanced_drugs_data.csv.pt  # PyTorch graph data
│   ├── balanced_drugs_data.csv.meta.pkl  # Metadata
│   └── output/final/
│       └── edge_gnn_best.pt        # Trained model weights
├── images/
│   └── background.jpg              # UI background image
└── .env                            # API keys (OpenRouter, Neo4j)
```

---

## 🔄 Data Flow
```
User Input (Drug A + Drug B)
    ↓
MongoDB Authentication Check
    ↓
Load Drugs from CSV → Extract SMILES
    ↓
Generate Molecular Fingerprints
    ↓
Visualize Structures (RDKit)
    ↓
GNN Prediction (EdgeGNN)
    ↓
LangChain Explanation (OpenRouter LLM)
    ↓
Display Results with Confidence Score