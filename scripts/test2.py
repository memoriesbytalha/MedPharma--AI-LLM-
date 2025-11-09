"""
Drug Interaction Prediction Module
Standalone module for predicting drug-drug interactions using EdgeGNN
"""

import torch
import pickle
import pandas as pd
from pathlib import Path
from torch_geometric.data import Data


class DrugInteractionPredictor:
    """
    Handles drug interaction prediction using pre-trained EdgeGNN model
    """
    
    def __init__(self, model_path, data_path, meta_path, device=None):
        """
        Initialize predictor with model and data paths
        
        Args:
            model_path: Path to trained model (.pt file)
            data_path: Path to graph data (.pt file)
            meta_path: Path to metadata (.pkl file)
            device: torch device (cuda/cpu), auto-detected if None
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔧 Loading predictor on device: {self.device}")
        
        # Load graph data
        print("📊 Loading graph data...")
        with torch.serialization.safe_globals([Data]):
            self.data = torch.load(data_path, map_location=self.device)
        
        # Load metadata
        print("📚 Loading metadata...")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        
        self.node2idx = meta["node2idx"]
        self.idx2node = meta["idx2node"]
        self.label_encoder = meta["label_encoder"]
        
        # Import and initialize model
        from models.edge_gnn import EdgeGNN
        
        print("🧠 Initializing model...")
        self.model = EdgeGNN(
            num_nodes=self.data.num_nodes,
            node_embed_dim=128,
            hidden_dim=256,
            num_classes=len(self.label_encoder.classes_)
        ).to(self.device)
        
        # Load trained weights
        print("⚡ Loading model weights...")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print("✅ Predictor ready!\n")
    
    def get_available_drugs(self):
        """Return list of all available drugs in the dataset"""
        return sorted(list(self.node2idx.keys()))
    
    def predict(self, drug_a, drug_b):
        """
        Predict interaction between two drugs
        
        Args:
            drug_a: First drug name (string)
            drug_b: Second drug name (string)
            
        Returns:
            dict: {
                'success': bool,
                'predicted_interaction': str,
                'confidence': float,
                'all_probabilities': dict,
                'error': str (if success=False)
            }
        """
        # Validate inputs
        drug_a, drug_b = drug_a.strip(), drug_b.strip()
        
        if drug_a not in self.node2idx:
            return {
                'success': False,
                'error': f"Drug '{drug_a}' not found in dataset"
            }
        
        if drug_b not in self.node2idx:
            return {
                'success': False,
                'error': f"Drug '{drug_b}' not found in dataset"
            }
        
        # Get node indices
        src = self.node2idx[drug_a]
        dst = self.node2idx[drug_b]
        
        # Run prediction
        with torch.no_grad():
            # Generate node embeddings
            x = self.model.node_emb(torch.arange(self.data.num_nodes, device=self.device))
            x = self.model.input_proj(x)
            
            # Apply graph convolutions
            for conv in self.model.convs:
                h = conv(x, self.data.edge_index)
                x = torch.relu(h) + x
            
            # Create edge feature by concatenating both drug embeddings
            edge_feat = torch.cat([x[src].unsqueeze(0), x[dst].unsqueeze(0)], dim=1)
            
            # Predict interaction
            pred = self.model.edge_mlp(edge_feat)
            probs = torch.softmax(pred, dim=1).cpu().numpy()[0]
            
            # Get predicted class
            pred_label = pred.argmax(dim=1).item()
            class_name = self.label_encoder.inverse_transform([pred_label])[0]
            confidence = float(probs[pred_label])
            
            # Get all class probabilities
            all_probs = {
                self.label_encoder.inverse_transform([i])[0]: float(probs[i])
                for i in range(len(probs))
            }
        
        return {
            'success': True,
            'predicted_interaction': class_name,
            'confidence': confidence,
            'all_probabilities': all_probs,
            'drug_a': drug_a,
            'drug_b': drug_b
        }
    
    def batch_predict(self, drug_pairs):
        """
        Predict interactions for multiple drug pairs
        
        Args:
            drug_pairs: List of tuples [(drug_a, drug_b), ...]
            
        Returns:
            List of prediction results
        """
        results = []
        for drug_a, drug_b in drug_pairs:
            result = self.predict(drug_a, drug_b)
            results.append(result)
        return results


def main():
    """Test the predictor"""
    import os
    
    # Setup paths
    BASE_DIR = r"D:\FYP Try"  # Change this to your actual base path
    DATA_DIR = os.path.join(BASE_DIR, "MED-PHARMA AI", "data")
    
    # Option 2: Use relative path (if file structure is consistent)
    # MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
    # DATA_DIR = os.path.join(MAIN_DIR, "..", "MED-PHARMA AI", "data")
    
    MODEL_PATH = os.path.join(DATA_DIR, "output", "final", "edge_gnn_best.pt")
    DATA_PT = os.path.join(DATA_DIR, "balanced_drugs_data.csv.pt")
    META_PATH = os.path.join(DATA_DIR, "balanced_drugs_data.csv.meta.pkl")
    
    import pdb;pdb.set_trace()
    # Initialize predictor
    predictor = DrugInteractionPredictor(
        model_path=MODEL_PATH,
        data_path=DATA_PT,
        meta_path=META_PATH
    )
    
    # Example predictions
    print("="*60)
    print("TESTING DRUG INTERACTION PREDICTIONS")
    print("="*60)
    
    # Get sample drugs
    drugs = predictor.get_available_drugs()
    print(f"\n📋 Total drugs available: {len(drugs)}")
    print(f"Sample drugs: {drugs[:5]}\n")
    
    # Test prediction 1
    drug1, drug2 = drugs[0], drugs[1]
    print(f"\n🔬 Testing: {drug1} + {drug2}")
    print("-"*60)
    
    result = predictor.predict(drug1, drug2)
    
    if result['success']:
        print(f"✅ Prediction: {result['predicted_interaction']}")
        print(f"🎯 Confidence: {result['confidence']*100:.2f}%")
        print(f"\n📊 All probabilities:")
        for interaction, prob in result['all_probabilities'].items():
            print(f"   {interaction}: {prob*100:.2f}%")
    else:
        print(f"❌ Error: {result['error']}")
    
    # Test with invalid drug
    print(f"\n\n🧪 Testing invalid drug")
    print("-"*60)
    result = predictor.predict("InvalidDrug123", drug1)
    print(f"❌ Error: {result['error']}")
    
    # Batch prediction
    print(f"\n\n📦 Testing batch prediction")
    print("-"*60)
    test_pairs = [(drugs[0], drugs[1]), (drugs[2], drugs[3])]
    results = predictor.batch_predict(test_pairs)
    
    for i, result in enumerate(results, 1):
        if result['success']:
            print(f"{i}. {result['drug_a']} + {result['drug_b']}")
            print(f"   → {result['predicted_interaction']} ({result['confidence']*100:.2f}%)")
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()