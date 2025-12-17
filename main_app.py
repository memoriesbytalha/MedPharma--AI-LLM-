# import base64
# import streamlit as st
# from pathlib import Path
# import numpy as np
# from pymongo import MongoClient
# from passlib.hash import pbkdf2_sha256
# import re
# import datetime
# import pandas as pd
# import os
# import warnings
# from rdkit import Chem
# from rdkit.Chem import Draw
# from PIL import Image
# from neo4j import GraphDatabase
# from rdkit.Chem.Draw import rdMolDraw2D
# from PIL import Image
# import io
# import torch
# from torch_geometric.data import Data
# from torch_geometric.utils import k_hop_subgraph
# from models.edge_gnn import EdgeGNN
# from rdkit import Chem
# from rdkit.Chem import AllChem, DataStructs
# import pickle
# import os
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI
# import requests
# from dotenv import load_dotenv
# import matplotlib.pyplot as plt
# import networkx as nx
# from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate # New/Updated Import
# load_dotenv()  # Load .env variables

# warnings.filterwarnings("ignore")
# warnings.simplefilter("ignore")
# warnings.warn("deprecated", DeprecationWarning)

# OPENROUTER_API_KEY = os.getenv('OPEN_API_ROUTER_KEY')

# MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_DIR = os.path.join(MAIN_DIR, "MED-PHARMA AI", "data")
# MODEL_PATH = os.path.join(DATA_DIR, "output", "final", "edge_gnn_best.pt")
# DATA_PT = os.path.join(DATA_DIR, "balanced_drugs_data.csv.pt")
# META_PATH = os.path.join(DATA_DIR, "balanced_drugs_data.csv.meta.pkl")
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataset_path = os.path.join(DATA_DIR, "balanced_drugs_data.csv")

# # ------Data Loading--------#
# df = pd.read_csv(dataset_path)
# drug_list = sorted(list(set(df['Drug_A'].tolist() + df['Drug_B'].tolist())))
# #====================LLM Seetings========================
# def create_explanation_bot(openrouter_api_key):
#     """Initialize LangChain bot for drug interaction explanations using OpenRouter"""
    
#     llm = ChatOpenAI(
#         model="z-ai/glm-4.5-air:free",  # Free model on OpenRouter
#         openai_api_key=openrouter_api_key,
#         openai_api_base="https://openrouter.ai/api/v1",
#         temperature=0.7,
#         max_tokens=2000
#     )

#     # Define the System (AI Persona) Message
#     system_template = """
#     You are a medical and chemical assistant AI helping explain drug interactions to non-specialists.
#     Avoid complex jargon. Use clear, layman-friendly language.
#     """
#     system_message = SystemMessagePromptTemplate.from_template(system_template)

#     # Define the Human (User Input/Data) Message
#     human_template = """
#         Predicted interaction between:
#         - Drug A: {drug_a} (Formula: {drug_a_formula})
#         - Drug B: {drug_b} (Formula: {drug_b_formula})
#         - Severity: {prediction}
#         - Confidence: {confidence}

#         Important pathways:
#         {pathways}

#         Explain in **clear layman-friendly language**:
#         1. Clinical meaning
#         2. How chemical composition might contribute
#         3. Precautions or concerns
#         """
#     human_message = HumanMessagePromptTemplate.from_template(human_template)
    
#     # Create the ChatPromptTemplate using the messages list
#     prompt_template = ChatPromptTemplate.from_messages([
#         system_message,
#         human_message
#     ])
#     # The chain logic was incorrect in the original code, 
#     # as llm() is not the correct way to initialize the runnable
#     # Correct LangChain V2 (Runnable) approach:
#     layman_chain = prompt_template | llm | StrOutputParser() # Assuming StrOutputParser is used later
    
#     return layman_chain

# def generate_layman_explanation_with_formula(layman_chain, explanation):
#     """Return layman-friendly explanation using LLM"""
#     pathway_summary = "\n".join([f"{p['rank']}. {p['path']}" for p in explanation.get("pathways", [])])
    
#     return layman_chain.invoke({
#         "drug_a": explanation["drug_a"],
#         "drug_b": explanation["drug_b"],
#         "drug_a_formula": explanation.get("drug_a_formula", "N/A"),
#         "drug_b_formula": explanation.get("drug_b_formula", "N/A"),
#         "prediction": explanation["prediction"],
#         "confidence": round(explanation["confidence"], 4),
#         "pathways": pathway_summary or "No pathway information available."
#     })



# # ==================== PATHWAY EXPLAINER CLASS ====================
# class PathwayExplainer:
#     """
#     Path-based explainer that shows actual connection chains between drugs
#     """
    
#     def __init__(self, model, data, node2idx, idx2node, label_encoder, device='cpu'):
#         self.model = model.to(device)
#         self.data = data
#         self.data.edge_index = self.data.edge_index.to(device)
#         self.node2idx = node2idx
#         self.idx2node = idx2node
#         self.label_encoder = label_encoder
#         self.device = device
#         self.model.eval()
        
#         # Build NetworkX graph for path finding
#         self._build_nx_graph()
#         self._cached_embeddings = None
    
#     def _build_nx_graph(self):
#         """Build NetworkX graph from PyG data"""
#         self.nx_graph = nx.DiGraph()
#         edge_index = self.data.edge_index.cpu().numpy()
        
#         for i in range(edge_index.shape[1]):
#             src, dst = edge_index[0, i], edge_index[1, i]
#             src_name = self.idx2node[src]
#             dst_name = self.idx2node[dst]
#             self.nx_graph.add_edge(src_name, dst_name)
    
#     def _get_node_embeddings(self):
#         """Get node embeddings from the model"""
#         if self._cached_embeddings is not None:
#             return self._cached_embeddings
        
#         with torch.no_grad():
#             x = self.model.node_emb(torch.arange(self.data.num_nodes, device=self.device))
#             x = self.model.input_proj(x)
            
#             for conv in self.model.convs:
#                 h = conv(x, self.data.edge_index)
#                 x = torch.relu(h) + x
        
#         self._cached_embeddings = x
#         return x
    
#     def explain_pathways(self, drug_a: str, drug_b: str, 
#                         max_paths: int = 5,
#                         max_path_length: int = 4) -> dict:
#         """Find and rank pathways between two drugs"""
        
#         if drug_a not in self.node2idx or drug_b not in self.node2idx:
#             return {'error': f'Drug not found in dataset'}
        
#         try:
#             src_idx = self.node2idx[drug_a]
#             dst_idx = self.node2idx[drug_b]
            
#             # Get prediction
#             with torch.no_grad():
#                 x = self._get_node_embeddings()
#                 edge_feat = torch.cat([x[src_idx].unsqueeze(0), x[dst_idx].unsqueeze(0)], dim=1)
#                 pred = self.model.edge_mlp(edge_feat)
#                 probs = torch.softmax(pred, dim=1).cpu().numpy()[0]
#                 pred_class = pred.argmax(dim=1).item()
#                 confidence = float(probs[pred_class])
#                 class_name = self.label_encoder.inverse_transform([pred_class])[0]
            
#             # Find all simple paths
#             all_paths = []
#             try:
#                 undirected_graph = self.nx_graph.to_undirected()
#                 for path in nx.all_simple_paths(undirected_graph, drug_a, drug_b, 
#                                                cutoff=max_path_length):
#                     if len(path) <= max_path_length + 1:
#                         all_paths.append(path)
#                         if len(all_paths) >= 50:
#                             break
#             except (nx.NetworkXNoPath, nx.NodeNotFound):
#                 pass
            
#             if not all_paths:
#                 return {
#                     'drug_a': drug_a,
#                     'drug_b': drug_b,
#                     'prediction': class_name,
#                     'confidence': confidence,
#                     'pathways': [],
#                     'warning': 'No connecting paths found - drugs may be disconnected in the graph'
#                 }
            
#             # Score paths
#             scored_paths = []
#             for path in all_paths:
#                 importance = self._compute_path_importance(path, src_idx, dst_idx, x)
#                 scored_paths.append({
#                     'path': path,
#                     'importance': importance,
#                     'length': len(path) - 1
#                 })
            
#             scored_paths.sort(key=lambda x: x['importance'], reverse=True)
#             top_paths = scored_paths[:max_paths]
            
#             # Generate explanations
#             pathway_explanations = []
#             for i, path_info in enumerate(top_paths, 1):
#                 path = path_info['path']
#                 importance = path_info['importance']
#                 explanation = self._generate_path_explanation(path, importance, drug_a, drug_b)
                
#                 pathway_explanations.append({
#                     'rank': i,
#                     'path': ' → '.join(path),
#                     'path_drugs': path,
#                     'importance': float(importance),
#                     'length': path_info['length'],
#                     'explanation': explanation
#                 })
            
#             return {
#                 'drug_a': drug_a,
#                 'drug_b': drug_b,
#                 'prediction': class_name,
#                 'confidence': confidence,
#                 'total_paths_found': len(all_paths),
#                 'pathways': pathway_explanations
#             }
            
#         except Exception as e:
#             return {'error': f'Pathway explanation failed: {str(e)}'}
    
#     def _compute_path_importance(self, path, src_idx, dst_idx, embeddings):
#         """Compute importance score for a path"""
#         path_indices = [self.node2idx[drug] for drug in path]
#         path_embs = embeddings[path_indices]
        
#         # Path length penalty
#         length_penalty = 1.0 / (len(path) ** 0.5)
        
#         # Embedding similarity along path
#         similarities = []
#         for i in range(len(path_embs) - 1):
#             sim = torch.cosine_similarity(
#                 path_embs[i].unsqueeze(0), 
#                 path_embs[i+1].unsqueeze(0)
#             ).item()
#             similarities.append(sim)
#         avg_similarity = np.mean(similarities) if similarities else 0
        
#         # Endpoint relevance
#         src_emb = embeddings[src_idx]
#         dst_emb = embeddings[dst_idx]
#         path_relevance = 0
#         for emb in path_embs[1:-1]:
#             src_sim = torch.cosine_similarity(src_emb.unsqueeze(0), emb.unsqueeze(0)).item()
#             dst_sim = torch.cosine_similarity(dst_emb.unsqueeze(0), emb.unsqueeze(0)).item()
#             path_relevance += (src_sim + dst_sim) / 2
#         path_relevance = path_relevance / max(len(path) - 2, 1)
        
#         importance = (0.3 * length_penalty + 0.4 * avg_similarity + 0.3 * path_relevance)
#         return importance
    
#     def _generate_path_explanation(self, path, importance, drug_a, drug_b):
#         """Generate natural language explanation"""
#         path_length = len(path) - 1
#         intermediate_drugs = path[1:-1]
        
#         if path_length == 1:
#             return f"Direct interaction between {drug_a} and {drug_b} in the knowledge graph"
#         elif path_length == 2:
#             bridge_drug = intermediate_drugs[0]
#             return (f"Both drugs interact with {bridge_drug}, creating an indirect "
#                    f"connection through shared interaction partner")
#         elif path_length == 3:
#             bridge1, bridge2 = intermediate_drugs[0], intermediate_drugs[1]
#             return (f"Interaction pathway through {bridge1} and {bridge2}, "
#                    f"suggesting shared pharmacological mechanisms or metabolic pathways")
#         else:
#             bridge_drugs = ', '.join(intermediate_drugs[:-1]) + f', and {intermediate_drugs[-1]}'
#             return (f"Complex interaction pathway involving {len(intermediate_drugs)} "
#                    f"intermediate drugs ({bridge_drugs}), indicating potential "
#                    f"indirect pharmacokinetic or pharmacodynamic interactions")
    
#     def visualize_pathways_streamlit(self, explanation):
#         """Visualize pathways for Streamlit"""
#         if 'error' in explanation or not explanation.get('pathways'):
#             return None
        
#         G = nx.DiGraph()
#         drug_a, drug_b = explanation['drug_a'], explanation['drug_b']
        
#         G.add_node(drug_a, node_type='target', color='#FF4444')
#         G.add_node(drug_b, node_type='target', color='#FF4444')
        
#         edge_importance = {}
#         for pathway in explanation['pathways'][:3]:
#             path_drugs = pathway['path_drugs']
#             importance = pathway['importance']
            
#             for i in range(len(path_drugs) - 1):
#                 src, dst = path_drugs[i], path_drugs[i+1]
#                 if src not in [drug_a, drug_b]:
#                     G.add_node(src, node_type='intermediate', color='#4DA6FF')
#                 if dst not in [drug_a, drug_b]:
#                     G.add_node(dst, node_type='intermediate', color='#4DA6FF')
                
#                 edge_key = (src, dst)
#                 edge_importance[edge_key] = max(edge_importance.get(edge_key, 0), importance)
#                 G.add_edge(src, dst, importance=edge_importance[edge_key])
        
#         fig, ax = plt.subplots(figsize=(14, 10))
#         pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
#         node_colors = [G.nodes[node].get('color', '#4DA6FF') for node in G.nodes()]
#         node_sizes = [2500 if G.nodes[node].get('node_type') == 'target' else 1500 
#                      for node in G.nodes()]
        
#         nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
#                               node_size=node_sizes, alpha=0.9, ax=ax)
        
#         for (u, v, data) in G.edges(data=True):
#             importance = data.get('importance', 0.5)
#             width = 1 + importance * 4
#             alpha = 0.4 + importance * 0.5
#             nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
#                                  edge_color='#666666', width=width,
#                                  alpha=alpha, arrows=True, arrowsize=20,
#                                  arrowstyle='->', ax=ax)
        
#         nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
        
#         ax.set_title(
#             f"Top Interaction Pathways: {drug_a} ↔ {drug_b}\n"
#             f"Prediction: {explanation['prediction']} ({explanation['confidence']*100:.1f}% confidence)",
#             fontsize=14, fontweight='bold', pad=20
#         )
#         ax.axis('off')
#         plt.tight_layout()
#         return fig

# # ==================== GNN EXPLAINER CLASS ====================
# class GNNExplainer:
#     """
#     Fixed Graph Neural Network Explainer for drug interaction predictions.
#     Integrated for Streamlit UI
#     """
    
#     def __init__(self, model, data, node2idx, idx2node, device='cpu'):
#         self.model = model.to(device)
#         self.data = data
#         self.data.edge_index = self.data.edge_index.to(device)
#         if hasattr(self.data, 'x') and self.data.x is not None:
#             self.data.x = self.data.x.to(device)
        
#         self.node2idx = node2idx
#         self.idx2node = idx2node
#         self.device = device
#         self.model.eval()
#         self._cached_embeddings = None
    
#     def _get_node_embeddings(self, use_cache=True):
#         """Get node embeddings from the model"""
#         if use_cache and self._cached_embeddings is not None:
#             return self._cached_embeddings
        
#         with torch.no_grad():
#             x = self.model.node_emb(torch.arange(self.data.num_nodes, device=self.device))
#             x = self.model.input_proj(x)
            
#             for conv in self.model.convs:
#                 h = conv(x, self.data.edge_index)
#                 x = torch.relu(h) + x
        
#         if use_cache:
#             self._cached_embeddings = x
#         return x
    
#     def explain_edge(self, drug_a: str, drug_b: str, 
#                      num_hops: int = 1,
#                      epochs: int = 100,  # Increased default
#                      lr: float = 0.05,   # Increased learning rate
#                      reg_coef: float = 0.05) -> dict:  # Reduced regularization
#         """
#         Explain prediction for a drug pair by learning important subgraph
#         """
#         if num_hops < 1 or num_hops > 3:
#             return {'error': 'num_hops must be between 1 and 3'}
        
#         if drug_a not in self.node2idx or drug_b not in self.node2idx:
#             return {'error': f'Drug not found in dataset'}
        
#         try:
#             src_idx = self.node2idx[drug_a]
#             dst_idx = self.node2idx[drug_b]
            
#             # Get k-hop subgraph
#             subset, edge_index, mapping, edge_mask = k_hop_subgraph(
#                 node_idx=[src_idx, dst_idx],
#                 num_hops=num_hops,
#                 edge_index=self.data.edge_index,
#                 relabel_nodes=True
#             )
            
#             subset = subset.to(self.device)
#             edge_index = edge_index.to(self.device)
            
#             if edge_index.size(1) == 0:
#                 return {'error': 'No edges found in subgraph'}
            
#             # Initialize learnable edge mask (must be a leaf tensor)
#             edge_mask_param = torch.nn.Parameter(
#                 torch.randn(edge_index.size(1), device=self.device) * 0.1
#             )
#             optimizer = torch.optim.Adam([edge_mask_param], lr=lr)
            
#             # Get original prediction
#             with torch.no_grad():
#                 orig_pred = self._predict_pair(src_idx, dst_idx)
#                 orig_probs = torch.softmax(orig_pred, dim=1)  # Apply softmax!
#                 target_class = orig_probs.argmax(dim=1)
#                 target_prob = orig_probs[0, target_class].item()  # Now between 0-1
            
#             # Optimize edge mask
#             best_loss = float('inf')
#             best_mask = None
            
#             for epoch in range(epochs):
#                 optimizer.zero_grad()
#                 mask = torch.sigmoid(edge_mask_param)
#                 pred = self._predict_with_mask(src_idx, dst_idx, subset, edge_index, mask)
                
#                 # Apply softmax to get probabilities
#                 probs = torch.softmax(pred, dim=1)
                
#                 prediction_loss = -torch.log(probs[0, target_class] + 1e-10)  # Log probability
#                 sparsity_loss = reg_coef * torch.sum(mask)
#                 entropy_loss = 0.01 * torch.sum(mask * torch.log(mask + 1e-10))  # Encourage diversity
                
#                 loss = prediction_loss + sparsity_loss + entropy_loss
                
#                 loss.backward()
#                 optimizer.step()
                
#                 if loss.item() < best_loss:
#                     best_loss = loss.item()
#                     best_mask = mask.detach().clone()
            
#             final_mask = torch.sigmoid(best_mask).cpu().numpy()
            
#             # Better filtering: use top-k instead of percentile for more diverse scores
#             top_k = min(15, len(final_mask))
#             top_indices = np.argsort(final_mask)[-top_k:][::-1]
            
#             # Calculate threshold from top-k
#             threshold = final_mask[top_indices[-1]] if len(top_indices) > 0 else 0.5
            
#             important_edges = []
#             for idx in top_indices:
#                 src = subset[edge_index[0, idx]].item()
#                 dst = subset[edge_index[1, idx]].item()
#                 importance = final_mask[idx]
                
#                 # Check if this is the target edge
#                 is_target = (src == src_idx and dst == dst_idx) or \
#                            (src == dst_idx and dst == src_idx) or \
#                            (dst == src_idx and src == dst_idx) or \
#                            (dst == dst_idx and src == src_idx)
                
#                 important_edges.append({
#                     'source': self.idx2node[src],
#                     'target': self.idx2node[dst],
#                     'importance': float(importance),
#                     'is_target_edge': is_target
#                 })
            
#             return {
#                 'drug_a': drug_a,
#                 'drug_b': drug_b,
#                 'original_prediction': target_class.item(),
#                 'prediction_confidence': target_prob,
#                 'important_edges': important_edges,
#                 'subgraph_size': len(subset),
#                 'num_edges_analyzed': len(final_mask),
#                 'importance_threshold': float(threshold),
#                 'avg_importance': float(final_mask.mean()),
#                 'max_importance': float(final_mask.max()),
#                 'min_importance': float(final_mask.min()),
#                 'std_importance': float(final_mask.std()),
#                 'optimization_loss': float(best_loss)
#             }
            
#         except Exception as e:
#             return {'error': f'Explanation failed: {str(e)}'}
    
#     def _predict_pair(self, src_idx: int, dst_idx: int) -> torch.Tensor:
#         """Helper: Predict interaction for a drug pair"""
#         x = self._get_node_embeddings()
#         edge_feat = torch.cat([x[src_idx].unsqueeze(0), x[dst_idx].unsqueeze(0)], dim=1)
#         pred = self.model.edge_mlp(edge_feat)
#         return pred
    
#     def _predict_with_mask(self, src_idx: int, dst_idx: int, 
#                           subset: torch.Tensor, edge_index: torch.Tensor, 
#                           mask: torch.Tensor) -> torch.Tensor:
#         """Predict with masked edges"""
#         x = self.model.node_emb(torch.arange(self.data.num_nodes, device=self.device))
#         x = self.model.input_proj(x)
        
#         for conv in self.model.convs:
#             full_edge_weight = torch.zeros(self.data.edge_index.size(1), device=self.device)
            
#             for i, (src, dst) in enumerate(edge_index.t()):
#                 src_global = subset[src]
#                 dst_global = subset[dst]
#                 edge_matches = (self.data.edge_index[0] == src_global) & \
#                               (self.data.edge_index[1] == dst_global)
#                 full_edge_weight[edge_matches] = mask[i]
            
#             try:
#                 h = conv(x, self.data.edge_index, edge_weight=full_edge_weight)
#             except TypeError:
#                 h = conv(x, self.data.edge_index)
            
#             x = torch.relu(h) + x
        
#         edge_feat = torch.cat([x[src_idx].unsqueeze(0), x[dst_idx].unsqueeze(0)], dim=1)
#         pred = self.model.edge_mlp(edge_feat)
#         return pred
    
#     def visualize_explanation_streamlit(self, explanation: dict):
#         """
#         Create visualization for Streamlit
#         Returns matplotlib figure
#         """
#         if 'error' in explanation:
#             return None
        
#         G = nx.DiGraph()
        
#         # Add target drugs
#         G.add_node(explanation['drug_a'], node_type='target')
#         G.add_node(explanation['drug_b'], node_type='target')
        
#         # Add important edges
#         threshold = explanation.get('importance_threshold', 0.5)
#         for edge in explanation['important_edges']:
#             if edge['importance'] >= threshold:
#                 G.add_node(edge['source'], node_type='neighbor')
#                 G.add_node(edge['target'], node_type='neighbor')
#                 edge_type = 'target' if edge.get('is_target_edge') else 'explanation'
#                 G.add_edge(edge['source'], edge['target'], 
#                           importance=edge['importance'], 
#                           edge_type=edge_type)
        
#         # Create figure
#         fig, ax = plt.subplots(figsize=(12, 8))
#         pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
        
#         # Node colors and sizes
#         node_colors = ['#FF4444' if G.nodes[node].get('node_type') == 'target' else '#4DA6FF' 
#                       for node in G.nodes()]
#         node_sizes = [2000 if G.nodes[node].get('node_type') == 'target' else 1000 
#                      for node in G.nodes()]
        
#         # Draw nodes
#         nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
#                               node_size=node_sizes, alpha=0.9, ax=ax)
        
#         # Draw edges
#         target_edges = [(u, v) for u, v, d in G.edges(data=True) 
#                        if d.get('edge_type') == 'target']
#         explain_edges = [(u, v) for u, v, d in G.edges(data=True) 
#                         if d.get('edge_type') == 'explanation']
        
#         if target_edges:
#             nx.draw_networkx_edges(G, pos, edgelist=target_edges, 
#                                   edge_color='#FF4444', width=4, alpha=0.8,
#                                   arrows=True, arrowsize=25, arrowstyle='->', ax=ax)
        
#         for u, v in explain_edges:
#             importance = G[u][v]['importance']
#             width = 1 + importance * 4
#             alpha = 0.3 + importance * 0.5
#             nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
#                                  edge_color='gray', width=width,
#                                  alpha=alpha, arrows=True, arrowsize=20,
#                                  arrowstyle='->', ax=ax)
        
#         # Labels
#         nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
        
#         conf = explanation.get('prediction_confidence', 0)
#         pred_class = explanation.get('original_prediction', '?')
#         ax.set_title(f"GNN Explanation: {explanation['drug_a']} ↔ {explanation['drug_b']}\n"
#                     f"Prediction: Class {pred_class} (confidence: {conf:.3f})", 
#                     fontsize=14, fontweight='bold', pad=20)
#         ax.axis('off')
        
#         return fig

# # ==================== CACHED LOADING ====================

# @st.cache_resource
# def load_model_data():
#     with torch.serialization.safe_globals([Data]):
#         data = torch.load(DATA_PT, map_location=DEVICE)
#     with open(META_PATH, "rb") as f:
#         meta = pickle.load(f)
#     node2idx = meta["node2idx"]
#     idx2node = meta["idx2node"]
#     label_encoder = meta["label_encoder"]
#     df = meta["df"]

#     model = EdgeGNN(
#         num_nodes=data.num_nodes,
#         node_embed_dim=128,
#         hidden_dim=256,
#         num_classes=len(label_encoder.classes_)
#     ).to(DEVICE)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#     model.eval()

#     return model, data, node2idx, idx2node, label_encoder, df

# @st.cache_resource
# def load_pathway_explainer():
#     """Cache the Pathway Explainer instance"""
#     model, data, node2idx, idx2node, label_encoder, df = load_model_data()
#     explainer = PathwayExplainer(model, data, node2idx, idx2node, label_encoder, DEVICE)
#     return explainer

# @st.cache_resource
# def load_explainer():
#     """Cache the GNN Explainer instance"""
#     model, data, node2idx, idx2node, label_encoder, df = load_model_data()
#     explainer = GNNExplainer(model, data, node2idx, idx2node, DEVICE)
#     return explainer

# # ==================== PREDICTION FUNCTION ====================

# def predict_interaction(drug_a, drug_b, model, data, node2idx, label_encoder):
#     drug_a, drug_b = drug_a.strip(), drug_b.strip()

#     if drug_a not in node2idx:
#         return None, f"❌ Drug '{drug_a}' not found in dataset."
#     if drug_b not in node2idx:
#         return None, f"❌ Drug '{drug_b}' not found in dataset."

#     src, dst = node2idx[drug_a], node2idx[drug_b]

#     with torch.no_grad():
#         x = model.node_emb(torch.arange(data.num_nodes, device=DEVICE))
#         x = model.input_proj(x)
#         for conv in model.convs:
#             h = conv(x, data.edge_index)
#             x = torch.relu(h) + x

#         edge_feat = torch.cat([x[src].unsqueeze(0), x[dst].unsqueeze(0)], dim=1)
#         pred = model.edge_mlp(edge_feat)
#         probs = torch.softmax(pred, dim=1).cpu().numpy()[0]
#         pred_label = pred.argmax(dim=1).item()
#         class_name = label_encoder.inverse_transform([pred_label])[0]
#         confidence = float(probs[pred_label])

#     result = {
#         "Predicted Interaction": class_name,
#         "Confidence": confidence,
#     }

#     return result, None

# # ==================== MONGODB & AUTH ====================

# client = MongoClient("mongodb://localhost:27017/")
# db = client['medpharmaI']
# users = db['users']

# def hash_password(password):
#     return pbkdf2_sha256.hash(password)

# def is_valid_email(email):
#     return re.match(r"[^@]+@[^@]+\.[^@]+", email)

# # ==================== STYLING ====================

# st.set_page_config(
#     page_title="MedPharma AI",
#     page_icon="💊",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# BACKGROUND = Path("images/1.jpeg")

# def smiles_to_image(smiles, size=(350, 350)):
#     mol = Chem.MolFromSmiles(smiles)
#     drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
#     rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
#     drawer.FinishDrawing()
#     img_bytes = drawer.GetDrawingText()
#     return Image.open(io.BytesIO(img_bytes))

# @st.cache_data
# def smiles_to_image_cached(smiles, size=(350, 350)):
#     return smiles_to_image(smiles, size)

# def set_page_background(png_file):
#     @st.cache_data()
#     def get_base64_of_bin_file(bin_file):
#         with open(bin_file, 'rb') as f:
#             data = f.read()
#         return base64.b64encode(data).decode()
    
#     try:
#         bin_str = get_base64_of_bin_file(png_file)
#     except FileNotFoundError:
#         bin_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z/C/HgAGgwJ/lK3Q6wAAAABJRU5ErkJggg=="
    
#     page_bg_img = f'''
#     <style>
#     .stApp {{
#         background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 20, 0.7)), url("data:image/png;base64,{bin_str}");
#         background-size: cover;
#         background-position: center;
#         background-attachment: fixed;
#         background-repeat: no-repeat;
#     }}
#     .stTextInput input, .stTextArea textarea {{
#         color: #333333 !important;
#         background-color: rgba(255, 255, 255, 0.9) !important;
#     }}
#     .stTextInput input::placeholder, .stTextArea textarea::placeholder {{
#         color: #666666 !important;
#         opacity: 1 !important;
#     }}
#     .stSelectbox select {{
#         color: #333333 !important;
#         background-color: rgba(255, 255, 255, 0.9) !important;
#     }}
#     body, h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {{
#         color: white !important;
#     }}
#     [data-testid="stHeader"] {{
#         background-color: rgba(0,0,0,0.5) !important;
#     }}
#     </style>
#     '''
#     st.markdown(page_bg_img, unsafe_allow_html=True)

# set_page_background(BACKGROUND)

# # ==================== PAGES ====================

# def home():
#     st.markdown("""
#     <div style="height: 100vh; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
#         <h2 style="font-size: 3.5rem; margin-bottom: 1rem;">Bridging the Gap Between General Physicians and Pharmacologists</h2>
#         <h2 style="font-size: 1.8rem; margin-bottom: 2rem;">From Molecules to Medicine — Know Before You Prescribe.</h2>
#         <p style="font-size: 1.2rem; max-width: 800px; margin-bottom: 2rem;">
#         Our AI-powered platform provides clear explanations, interactive graphs, and insights to support clinical decisions.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)

# def login():
#     st.title("🔐 Authentication")
#     tab1, tab2 = st.tabs(["Login", "Register"])
    
#     with tab1:
#         with st.form("login_form"):
#             username = st.text_input("Username")
#             password = st.text_input("Password", type="password")
#             submitted = st.form_submit_button("Login")
            
#             if submitted:
#                 user = users.find_one({"username": username})
#                 if user and pbkdf2_sha256.verify(password, user["password"]):
#                     st.session_state.logged_in = True
#                     st.session_state.user = username
#                     st.success("Logged in successfully!")
#                     st.session_state.page = "inference"
#                     st.rerun()
#                 else:
#                     st.error("Invalid credentials")

#     with tab2:
#         with st.form("registration_form"):
#             st.subheader("Create New Account")
#             new_username = st.text_input("Choose Username")
#             new_email = st.text_input("Email Address")
#             new_password = st.text_input("Create Password", type="password")
#             confirm_password = st.text_input("Confirm Password", type="password")
#             submitted_reg = st.form_submit_button("Register")
            
#             if submitted_reg:
#                 if not all([new_username, new_email, new_password, confirm_password]):
#                     st.error("Please fill all fields")
#                 elif new_password != confirm_password:
#                     st.error("Passwords don't match")
#                 elif not is_valid_email(new_email):
#                     st.error("Please enter a valid email address")
#                 elif users.find_one({"$or": [{"username": new_username}, {"email": new_email}]}):
#                     st.error("Username or email already exists")
#                 else:
#                     user_data = {
#                         "username": new_username,
#                         "email": new_email,
#                         "password": hash_password(new_password),
#                         "created_at": datetime.datetime.now()
#                     }
#                     users.insert_one(user_data)
#                     st.success("Account created successfully! Please login.")

# def about():
#     st.title("📖 About Us")
#     st.markdown("""
#     We are final year students from **Bahria University Islamabad**, developing **MedPharma AI** — an intelligent platform for predicting and explaining **drug-drug interactions**.

#     Our solution leverages **graph neural networks, molecular fingerprints, and explainable AI (XAI)** to assist healthcare professionals in making informed decisions safely and efficiently.

#     The platform features a **dynamic, interactive interface**, offering visual drug interaction graphs, confidence-based predictions, and AI-powered explanations to support clinical judgment.
#     """)

# def contact():
#     st.title("Contact Us")
#     st.subheader("Direct Contacts")
#     st.markdown("""
#     📞 **Phone**: +92 317 5994687
#     📞 **Phone**: +92 332 1200260
#     ✉️ **Email**: muhammadtalha7893@yahoo.com 
#     ✉️ **Email**: zainulabadiennaqvi@gmail.com
#     """)
    
#     st.subheader("Office Hours")
#     st.markdown("""
#     - Monday - Friday: 9:00 AM - 5:00 PM  
#     - Saturday: 10:00 AM - 2:00 PM  
#     - Sunday: Closed
#     """)
    
#     st.subheader("Send Us a Message")
#     with st.form("contact_form"):
#         name = st.text_input("Your Name*")
#         email = st.text_input("Email Address*")
#         subject = st.text_input("Subject")
#         message = st.text_area("Your Message*")
#         submitted = st.form_submit_button("Send Message")
        
#         if submitted:
#             if not name or not email or not message:
#                 st.error("Please fill in all required fields (*)")
#             else:
#                 user_data = {
#                     "username": name,
#                     "email": email,
#                     "subject": subject,
#                     "message": message,
#                 }
#                 db.contact.insert_one(user_data)
#                 st.success("Thank you for your message! We'll respond within 24 hours.")

# def inference_with_explanation():
#     """Enhanced inference with Pathway Explainer"""
    
#     st.title("💊 Drug Interaction Prediction with Pathway Analysis")
#     st.markdown("Enter two drugs to visualize their molecular structure, predict interaction, and see **connecting pathways**:")

#     model, data, node2idx, idx2node, label_encoder, df = load_model_data()
#     pathway_explainer = load_pathway_explainer()
    
#     drug_list = sorted(list(set(df["Drug_A"]).union(set(df["Drug_B"]))))
#     col1, col2 = st.columns(2)
#     with col1:
#         drug1 = st.selectbox("Drug 1", options=drug_list, key="drug1_select")
#     with col2:
#         drug2 = st.selectbox("Drug 2", options=drug_list, key="drug2_select")

#     # Explainer settings
#     with st.expander("⚙️ Pathway Analysis Settings"):
#         max_paths = st.slider("Number of pathways to show", 3, 10, 5,
#                              help="How many connection pathways to display")
#         max_length = st.slider("Maximum pathway length", 2, 5, 4,
#                               help="Maximum number of intermediate drugs in a pathway")

#     if st.button("🔍 Predict Interaction & Analyze Pathways", type="primary"):
#         with st.spinner("Analyzing drug interaction and pathways..."):
#             pair_df = df[((df['Drug_A'] == drug1) & (df['Drug_B'] == drug2)) |
#                          ((df['Drug_A'] == drug2) & (df['Drug_B'] == drug1))]

#             if pair_df.empty:
#                 st.warning(f"⚠️ No interaction data found for {drug1} and {drug2}.")
#                 return

#             row = pair_df.iloc[0]

#             # Assign SMILES + formulas
#             if row["Drug_A"] == drug1:
#                 smiles1, smiles2 = row["DrugA_SMILES"], row["DrugB_SMILES"]
#                 formula1, formula2 = row["DrugA_Formula"], row["DrugB_Formula"]
#             else:
#                 smiles1, smiles2 = row["DrugB_SMILES"], row["DrugA_SMILES"]
#                 formula1, formula2 = row["DrugB_Formula"], row["DrugA_Formula"]

#             # Molecular visualization
#             st.markdown("### 🧪 Molecular Structures")
#             c1, c2 = st.columns(2)
#             with c1:
#                 st.image(smiles_to_image_cached(smiles1), caption=f"{drug1}\nFormula: {formula1}")
#             with c2:
#                 st.image(smiles_to_image_cached(smiles2), caption=f"{drug2}\nFormula: {formula2}")

#             # Predict interaction
#             result, error = predict_interaction(drug1, drug2, model, data, node2idx, label_encoder)
#             if error:
#                 st.error(error)
#                 return

#             predicted_interaction = result['Predicted Interaction']
#             confidence = result['Confidence']

#             st.success(f"💡 **Predicted Interaction:** {predicted_interaction} ({confidence*100:.2f}% confidence)")

#             # ====== PATHWAY EXPLAINER ======
#             st.markdown("---")
#             st.markdown("### Pathway Analysis")
            
#             progress_bar = st.progress(0)
#             status_text = st.empty()
            
#             status_text.text("Finding pathways between drugs...")
#             progress_bar.progress(30)
            
#             explanation = pathway_explainer.explain_pathways(
#                 drug1, drug2,
#                 max_paths=max_paths,
#                 max_path_length=max_length
#             )
            
#             progress_bar.progress(100)
#             status_text.empty()
#             progress_bar.empty()
            
#             if 'error' in explanation:
#                 st.error(f"⚠️ {explanation['error']}")
#                 return
            
#             if 'warning' in explanation:
#                 st.warning(f"⚠️ {explanation['warning']}")
#                 st.info("💡 The drugs may not have direct or indirect connections in the knowledge graph. The prediction is based on learned patterns from similar drug structures.")
#                 return
            
#             # Display summary
#             st.markdown(f"""
#             <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0;'>
#                 <h3 style='color: white; margin: 0;'>📊 Pathway Summary</h3>
#                 <p style='color: white; font-size: 18px; margin: 10px 0;'>
#                     <strong>Query:</strong> {explanation['drug_a']} + {explanation['drug_b']}<br>
#                     <strong>Prediction:</strong> {explanation['prediction']} ({explanation['confidence']*100:.1f}% confidence)<br>
#                     <strong>Total Pathways Found:</strong> {explanation['total_paths_found']}
#                 </p>
#             </div>
#             """, unsafe_allow_html=True)
            
#             # Display pathways
#             st.markdown("#### 🔗 Top Connecting Pathways")
            
#             for pathway in explanation['pathways']:
#                 with st.expander(f"**Pathway {pathway['rank']}** | Importance: {pathway['importance']:.3f} | Length: {pathway['length']} hops", expanded=(pathway['rank'] <= 2)):
#                     st.markdown(f"**Path:** `{pathway['path']}`")
#                     st.markdown(f"**Explanation:** {pathway['explanation']}")
                    
#                     # Color-code intermediate drugs
#                     if pathway['length'] > 1:
#                         st.markdown("**Intermediate Drugs:**")
#                         intermediate = pathway['path_drugs'][1:-1]
#                         cols = st.columns(len(intermediate))
#                         for idx, drug in enumerate(intermediate):
#                             with cols[idx]:
#                                 st.markdown(f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 5px; text-align: center;'>{drug}</div>", unsafe_allow_html=True)
#             layman_chain = create_explanation_bot(OPENROUTER_API_KEY)

#             layman_text = generate_layman_explanation_with_formula(layman_chain, {
#                 "drug_a": drug1,
#                 "drug_b": drug2,
#                 "drug_a_formula": formula1,
#                 "drug_b_formula": formula2,
#                 "prediction": predicted_interaction,
#                 "confidence": confidence,
#                 "pathways": explanation.get('pathways', [])
#             })
            
#             st.markdown("### 📝Explanation")
#             st.info(layman_text)

            
#             # Technical details
#             with st.expander("📊 Technical Details"):
#                 st.write(f"**Drug A SMILES:** {smiles1}")
#                 st.write(f"**Drug B SMILES:** {smiles2}")
#                 st.write(f"**Prediction Confidence:** {confidence*100:.2f}%")
#                 st.write(f"**Total Pathways Discovered:** {explanation['total_paths_found']}")
#                 st.write(f"**Pathways Shown:** {len(explanation['pathways'])}")
                
#                 st.markdown("**Full Pathway Data:**")
#                 st.json(explanation)

# # ==================== NAVIGATION ====================

# # Initialize session state
# if 'page' not in st.session_state:
#     st.session_state.page = "home"
# if 'logged_in' not in st.session_state:
#     st.session_state.logged_in = False

# # Create navbar
# cols = st.columns([2,1,1,1,1])
# with cols[0]:
#     if st.button("🏠 MedPharma AI"):
#         st.session_state.page = "home"
# with cols[1]:
#     if st.button("About"):
#         st.session_state.page = "about"
# with cols[2]:
#     if st.button("Contact"):
#         st.session_state.page = "contact"
# with cols[3]:
#     if st.button("Inference"):
#         st.session_state.page = "inference" if st.session_state.logged_in else "login"
# with cols[4]:
#     if st.session_state.logged_in:
#         if st.button("Logout"):
#             st.session_state.logged_in = False
#             st.session_state.user = None
#             st.session_state.page = "home"
#             st.rerun()
#     else:
#         if st.button("Register"):
#             st.session_state.page = "login"

# # Page routing
# if st.session_state.page == "home":
#     home()
# elif st.session_state.page == "login":
#     login()
# elif st.session_state.page == "about":
#     about()
# elif st.session_state.page == "contact":
#     contact()
# elif st.session_state.page == "inference":
#     inference_with_explanation()

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
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from models.edge_gnn import EdgeGNN
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import pickle
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import requests
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import networkx as nx
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate # New/Updated Import
import pickle
import torch
from torch_geometric.data import Data
import streamlit as st
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
#====================LLM Seetings========================
def create_explanation_bot(openrouter_api_key):
    """Initialize LangChain bot for drug interaction explanations using OpenRouter"""
    
    llm = ChatOpenAI(
        model="arcee-ai/trinity-mini:free",  # Free model on OpenRouter
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.7,
        max_tokens=2000
    )

    # Define the System (AI Persona) Message
    system_template = """
    You are a medical and chemical assistant AI helping explain drug interactions to non-specialists.
    Avoid complex jargon. Use clear, layman-friendly language.
    """
    system_message = SystemMessagePromptTemplate.from_template(system_template)

    # Define the Human (User Input/Data) Message
    human_template = """
        Predicted interaction between:
        - Drug A: {drug_a} (Formula: {drug_a_formula})
        - Drug B: {drug_b} (Formula: {drug_b_formula})
        - Severity: {prediction}
        - Confidence: {confidence}

        Important pathways:
        {pathways}

        Explain in **clear layman-friendly language**:
        1. Clinical meaning
        2. How chemical composition might contribute
        3. Precautions or concerns
        """
    human_message = HumanMessagePromptTemplate.from_template(human_template)
    
    # Create the ChatPromptTemplate using the messages list
    prompt_template = ChatPromptTemplate.from_messages([
        system_message,
        human_message
    ])
    # The chain logic was incorrect in the original code, 
    # as llm() is not the correct way to initialize the runnable
    # Correct LangChain V2 (Runnable) approach:
    layman_chain = prompt_template | llm | StrOutputParser() # Assuming StrOutputParser is used later
    
    return layman_chain

def generate_layman_explanation_with_formula(layman_chain, explanation):
    """Return layman-friendly explanation using LLM"""
    pathway_summary = "\n".join([f"{p['rank']}. {p['path']}" for p in explanation.get("pathways", [])])
    
    return layman_chain.invoke({
        "drug_a": explanation["drug_a"],
        "drug_b": explanation["drug_b"],
        "drug_a_formula": explanation.get("drug_a_formula", "N/A"),
        "drug_b_formula": explanation.get("drug_b_formula", "N/A"),
        "prediction": explanation["prediction"],
        "confidence": round(explanation["confidence"], 4),
        "pathways": pathway_summary or "No pathway information available."
    })



# ==================== PATHWAY EXPLAINER CLASS ====================
class PathwayExplainer:
    """
    Path-based explainer that shows actual connection chains between drugs
    """
    
    def __init__(self, model, data, node2idx, idx2node, label_encoder, device='cpu'):
        self.model = model.to(device)
        self.data = data
        self.data.edge_index = self.data.edge_index.to(device)
        self.node2idx = node2idx
        self.idx2node = idx2node
        self.label_encoder = label_encoder
        self.device = device
        self.model.eval()
        
        # Build NetworkX graph for path finding
        self._build_nx_graph()
        self._cached_embeddings = None
    
    def _build_nx_graph(self):
        """Build NetworkX graph from PyG data"""
        self.nx_graph = nx.DiGraph()
        edge_index = self.data.edge_index.cpu().numpy()
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            src_name = self.idx2node[src]
            dst_name = self.idx2node[dst]
            self.nx_graph.add_edge(src_name, dst_name)
    
    def _get_node_embeddings(self):
        """Get node embeddings from the model"""
        if self._cached_embeddings is not None:
            return self._cached_embeddings
        
        with torch.no_grad():
            x = self.model.node_emb(torch.arange(self.data.num_nodes, device=self.device))
            x = self.model.input_proj(x)
            
            for conv in self.model.convs:
                h = conv(x, self.data.edge_index)
                x = torch.relu(h) + x
        
        self._cached_embeddings = x
        return x
    
    def explain_pathways(self, drug_a: str, drug_b: str, 
                        max_paths: int = 5,
                        max_path_length: int = 4) -> dict:
        """Find and rank pathways between two drugs"""
        
        if drug_a not in self.node2idx or drug_b not in self.node2idx:
            return {'error': f'Drug not found in dataset'}
        
        try:
            src_idx = self.node2idx[drug_a]
            dst_idx = self.node2idx[drug_b]
            
            # Get prediction
            with torch.no_grad():
                x = self._get_node_embeddings()
                edge_feat = torch.cat([x[src_idx].unsqueeze(0), x[dst_idx].unsqueeze(0)], dim=1)
                pred = self.model.edge_mlp(edge_feat)
                probs = torch.softmax(pred, dim=1).cpu().numpy()[0]
                pred_class = pred.argmax(dim=1).item()
                confidence = float(probs[pred_class])
                class_name = self.label_encoder.inverse_transform([pred_class])[0]
            
            # Find all simple paths
            all_paths = []
            try:
                undirected_graph = self.nx_graph.to_undirected()
                for path in nx.all_simple_paths(undirected_graph, drug_a, drug_b, 
                                               cutoff=max_path_length):
                    if len(path) <= max_path_length + 1:
                        all_paths.append(path)
                        if len(all_paths) >= 50:
                            break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
            
            if not all_paths:
                return {
                    'drug_a': drug_a,
                    'drug_b': drug_b,
                    'prediction': class_name,
                    'confidence': confidence,
                    'pathways': [],
                    'warning': 'No connecting paths found - drugs may be disconnected in the graph'
                }
            
            # Score paths
            scored_paths = []
            for path in all_paths:
                importance = self._compute_path_importance(path, src_idx, dst_idx, x)
                scored_paths.append({
                    'path': path,
                    'importance': importance,
                    'length': len(path) - 1
                })
            
            scored_paths.sort(key=lambda x: x['importance'], reverse=True)
            top_paths = scored_paths[:max_paths]
            
            # Generate explanations
            pathway_explanations = []
            for i, path_info in enumerate(top_paths, 1):
                path = path_info['path']
                importance = path_info['importance']
                explanation = self._generate_path_explanation(path, importance, drug_a, drug_b)
                
                pathway_explanations.append({
                    'rank': i,
                    'path': ' → '.join(path),
                    'path_drugs': path,
                    'importance': float(importance),
                    'length': path_info['length'],
                    'explanation': explanation
                })
            
            return {
                'drug_a': drug_a,
                'drug_b': drug_b,
                'prediction': class_name,
                'confidence': confidence,
                'total_paths_found': len(all_paths),
                'pathways': pathway_explanations
            }
            
        except Exception as e:
            return {'error': f'Pathway explanation failed: {str(e)}'}
    
    def _compute_path_importance(self, path, src_idx, dst_idx, embeddings):
        """Compute importance score for a path"""
        path_indices = [self.node2idx[drug] for drug in path]
        path_embs = embeddings[path_indices]
        
        # Path length penalty
        length_penalty = 1.0 / (len(path) ** 0.5)
        
        # Embedding similarity along path
        similarities = []
        for i in range(len(path_embs) - 1):
            sim = torch.cosine_similarity(
                path_embs[i].unsqueeze(0), 
                path_embs[i+1].unsqueeze(0)
            ).item()
            similarities.append(sim)
        avg_similarity = np.mean(similarities) if similarities else 0
        
        # Endpoint relevance
        src_emb = embeddings[src_idx]
        dst_emb = embeddings[dst_idx]
        path_relevance = 0
        for emb in path_embs[1:-1]:
            src_sim = torch.cosine_similarity(src_emb.unsqueeze(0), emb.unsqueeze(0)).item()
            dst_sim = torch.cosine_similarity(dst_emb.unsqueeze(0), emb.unsqueeze(0)).item()
            path_relevance += (src_sim + dst_sim) / 2
        path_relevance = path_relevance / max(len(path) - 2, 1)
        
        importance = (0.3 * length_penalty + 0.4 * avg_similarity + 0.3 * path_relevance)
        return importance
    
    def _generate_path_explanation(self, path, importance, drug_a, drug_b):
        """Generate natural language explanation"""
        path_length = len(path) - 1
        intermediate_drugs = path[1:-1]
        
        if path_length == 1:
            return f"Direct interaction between {drug_a} and {drug_b} in the knowledge graph"
        elif path_length == 2:
            bridge_drug = intermediate_drugs[0]
            return (f"Both drugs interact with {bridge_drug}, creating an indirect "
                   f"connection through shared interaction partner")
        elif path_length == 3:
            bridge1, bridge2 = intermediate_drugs[0], intermediate_drugs[1]
            return (f"Interaction pathway through {bridge1} and {bridge2}, "
                   f"suggesting shared pharmacological mechanisms or metabolic pathways")
        else:
            bridge_drugs = ', '.join(intermediate_drugs[:-1]) + f', and {intermediate_drugs[-1]}'
            return (f"Complex interaction pathway involving {len(intermediate_drugs)} "
                   f"intermediate drugs ({bridge_drugs}), indicating potential "
                   f"indirect pharmacokinetic or pharmacodynamic interactions")
    
    def visualize_pathways_streamlit(self, explanation):
        """Visualize pathways for Streamlit"""
        if 'error' in explanation or not explanation.get('pathways'):
            return None
        
        G = nx.DiGraph()
        drug_a, drug_b = explanation['drug_a'], explanation['drug_b']
        
        G.add_node(drug_a, node_type='target', color='#FF4444')
        G.add_node(drug_b, node_type='target', color='#FF4444')
        
        edge_importance = {}
        for pathway in explanation['pathways'][:3]:
            path_drugs = pathway['path_drugs']
            importance = pathway['importance']
            
            for i in range(len(path_drugs) - 1):
                src, dst = path_drugs[i], path_drugs[i+1]
                if src not in [drug_a, drug_b]:
                    G.add_node(src, node_type='intermediate', color='#4DA6FF')
                if dst not in [drug_a, drug_b]:
                    G.add_node(dst, node_type='intermediate', color='#4DA6FF')
                
                edge_key = (src, dst)
                edge_importance[edge_key] = max(edge_importance.get(edge_key, 0), importance)
                G.add_edge(src, dst, importance=edge_importance[edge_key])
        
        fig, ax = plt.subplots(figsize=(14, 10))
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        node_colors = [G.nodes[node].get('color', '#4DA6FF') for node in G.nodes()]
        node_sizes = [2500 if G.nodes[node].get('node_type') == 'target' else 1500 
                     for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.9, ax=ax)
        
        for (u, v, data) in G.edges(data=True):
            importance = data.get('importance', 0.5)
            width = 1 + importance * 4
            alpha = 0.4 + importance * 0.5
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                                 edge_color='#666666', width=width,
                                 alpha=alpha, arrows=True, arrowsize=20,
                                 arrowstyle='->', ax=ax)
        
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
        
        ax.set_title(
            f"Top Interaction Pathways: {drug_a} ↔ {drug_b}\n"
            f"Prediction: {explanation['prediction']} ({explanation['confidence']*100:.1f}% confidence)",
            fontsize=14, fontweight='bold', pad=20
        )
        ax.axis('off')
        plt.tight_layout()
        return fig

# ==================== GNN EXPLAINER CLASS ====================
class GNNExplainer:
    """
    Fixed Graph Neural Network Explainer for drug interaction predictions.
    Integrated for Streamlit UI
    """
    
    def __init__(self, model, data, node2idx, idx2node, device='cpu'):
        self.model = model.to(device)
        self.data = data
        self.data.edge_index = self.data.edge_index.to(device)
        if hasattr(self.data, 'x') and self.data.x is not None:
            self.data.x = self.data.x.to(device)
        
        self.node2idx = node2idx
        self.idx2node = idx2node
        self.device = device
        self.model.eval()
        self._cached_embeddings = None
    
    def _get_node_embeddings(self, use_cache=True):
        """Get node embeddings from the model"""
        if use_cache and self._cached_embeddings is not None:
            return self._cached_embeddings
        
        with torch.no_grad():
            x = self.model.node_emb(torch.arange(self.data.num_nodes, device=self.device))
            x = self.model.input_proj(x)
            
            for conv in self.model.convs:
                h = conv(x, self.data.edge_index)
                x = torch.relu(h) + x
        
        if use_cache:
            self._cached_embeddings = x
        return x
    
    def explain_edge(self, drug_a: str, drug_b: str, 
                     num_hops: int = 1,
                     epochs: int = 100,  # Increased default
                     lr: float = 0.05,   # Increased learning rate
                     reg_coef: float = 0.05) -> dict:  # Reduced regularization
        """
        Explain prediction for a drug pair by learning important subgraph
        """
        if num_hops < 1 or num_hops > 3:
            return {'error': 'num_hops must be between 1 and 3'}
        
        if drug_a not in self.node2idx or drug_b not in self.node2idx:
            return {'error': f'Drug not found in dataset'}
        
        try:
            src_idx = self.node2idx[drug_a]
            dst_idx = self.node2idx[drug_b]
            
            # Get k-hop subgraph
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                node_idx=[src_idx, dst_idx],
                num_hops=num_hops,
                edge_index=self.data.edge_index,
                relabel_nodes=True
            )
            
            subset = subset.to(self.device)
            edge_index = edge_index.to(self.device)
            
            if edge_index.size(1) == 0:
                return {'error': 'No edges found in subgraph'}
            
            # Initialize learnable edge mask (must be a leaf tensor)
            edge_mask_param = torch.nn.Parameter(
                torch.randn(edge_index.size(1), device=self.device) * 0.1
            )
            optimizer = torch.optim.Adam([edge_mask_param], lr=lr)
            
            # Get original prediction
            with torch.no_grad():
                orig_pred = self._predict_pair(src_idx, dst_idx)
                orig_probs = torch.softmax(orig_pred, dim=1)  # Apply softmax!
                target_class = orig_probs.argmax(dim=1)
                target_prob = orig_probs[0, target_class].item()  # Now between 0-1
            
            # Optimize edge mask
            best_loss = float('inf')
            best_mask = None
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                mask = torch.sigmoid(edge_mask_param)
                pred = self._predict_with_mask(src_idx, dst_idx, subset, edge_index, mask)
                
                # Apply softmax to get probabilities
                probs = torch.softmax(pred, dim=1)
                
                prediction_loss = -torch.log(probs[0, target_class] + 1e-10)  # Log probability
                sparsity_loss = reg_coef * torch.sum(mask)
                entropy_loss = 0.01 * torch.sum(mask * torch.log(mask + 1e-10))  # Encourage diversity
                
                loss = prediction_loss + sparsity_loss + entropy_loss
                
                loss.backward()
                optimizer.step()
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_mask = mask.detach().clone()
            
            final_mask = torch.sigmoid(best_mask).cpu().numpy()
            
            # Better filtering: use top-k instead of percentile for more diverse scores
            top_k = min(15, len(final_mask))
            top_indices = np.argsort(final_mask)[-top_k:][::-1]
            
            # Calculate threshold from top-k
            threshold = final_mask[top_indices[-1]] if len(top_indices) > 0 else 0.5
            
            important_edges = []
            for idx in top_indices:
                src = subset[edge_index[0, idx]].item()
                dst = subset[edge_index[1, idx]].item()
                importance = final_mask[idx]
                
                # Check if this is the target edge
                is_target = (src == src_idx and dst == dst_idx) or \
                           (src == dst_idx and dst == src_idx) or \
                           (dst == src_idx and src == dst_idx) or \
                           (dst == dst_idx and src == src_idx)
                
                important_edges.append({
                    'source': self.idx2node[src],
                    'target': self.idx2node[dst],
                    'importance': float(importance),
                    'is_target_edge': is_target
                })
            
            return {
                'drug_a': drug_a,
                'drug_b': drug_b,
                'original_prediction': target_class.item(),
                'prediction_confidence': target_prob,
                'important_edges': important_edges,
                'subgraph_size': len(subset),
                'num_edges_analyzed': len(final_mask),
                'importance_threshold': float(threshold),
                'avg_importance': float(final_mask.mean()),
                'max_importance': float(final_mask.max()),
                'min_importance': float(final_mask.min()),
                'std_importance': float(final_mask.std()),
                'optimization_loss': float(best_loss)
            }
            
        except Exception as e:
            return {'error': f'Explanation failed: {str(e)}'}
    
    def _predict_pair(self, src_idx: int, dst_idx: int) -> torch.Tensor:
        """Helper: Predict interaction for a drug pair"""
        x = self._get_node_embeddings()
        edge_feat = torch.cat([x[src_idx].unsqueeze(0), x[dst_idx].unsqueeze(0)], dim=1)
        pred = self.model.edge_mlp(edge_feat)
        return pred
    
    def _predict_with_mask(self, src_idx: int, dst_idx: int, 
                          subset: torch.Tensor, edge_index: torch.Tensor, 
                          mask: torch.Tensor) -> torch.Tensor:
        """Predict with masked edges"""
        x = self.model.node_emb(torch.arange(self.data.num_nodes, device=self.device))
        x = self.model.input_proj(x)
        
        for conv in self.model.convs:
            full_edge_weight = torch.zeros(self.data.edge_index.size(1), device=self.device)
            
            for i, (src, dst) in enumerate(edge_index.t()):
                src_global = subset[src]
                dst_global = subset[dst]
                edge_matches = (self.data.edge_index[0] == src_global) & \
                              (self.data.edge_index[1] == dst_global)
                full_edge_weight[edge_matches] = mask[i]
            
            try:
                h = conv(x, self.data.edge_index, edge_weight=full_edge_weight)
            except TypeError:
                h = conv(x, self.data.edge_index)
            
            x = torch.relu(h) + x
        
        edge_feat = torch.cat([x[src_idx].unsqueeze(0), x[dst_idx].unsqueeze(0)], dim=1)
        pred = self.model.edge_mlp(edge_feat)
        return pred
    
    def visualize_explanation_streamlit(self, explanation: dict):
        """
        Create visualization for Streamlit
        Returns matplotlib figure
        """
        if 'error' in explanation:
            return None
        
        G = nx.DiGraph()
        
        # Add target drugs
        G.add_node(explanation['drug_a'], node_type='target')
        G.add_node(explanation['drug_b'], node_type='target')
        
        # Add important edges
        threshold = explanation.get('importance_threshold', 0.5)
        for edge in explanation['important_edges']:
            if edge['importance'] >= threshold:
                G.add_node(edge['source'], node_type='neighbor')
                G.add_node(edge['target'], node_type='neighbor')
                edge_type = 'target' if edge.get('is_target_edge') else 'explanation'
                G.add_edge(edge['source'], edge['target'], 
                          importance=edge['importance'], 
                          edge_type=edge_type)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
        
        # Node colors and sizes
        node_colors = ['#FF4444' if G.nodes[node].get('node_type') == 'target' else "#FFFFFF" 
                      for node in G.nodes()]
        node_sizes = [2000 if G.nodes[node].get('node_type') == 'target' else 1000 
                     for node in G.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.9, ax=ax)
        
        # Draw edges
        target_edges = [(u, v) for u, v, d in G.edges(data=True) 
                       if d.get('edge_type') == 'target']
        explain_edges = [(u, v) for u, v, d in G.edges(data=True) 
                        if d.get('edge_type') == 'explanation']
        
        if target_edges:
            nx.draw_networkx_edges(G, pos, edgelist=target_edges, 
                                  edge_color='#FF4444', width=4, alpha=0.8,
                                  arrows=True, arrowsize=25, arrowstyle='->', ax=ax)
        
        for u, v in explain_edges:
            importance = G[u][v]['importance']
            width = 1 + importance * 4
            alpha = 0.3 + importance * 0.5
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                                 edge_color='gray', width=width,
                                 alpha=alpha, arrows=True, arrowsize=20,
                                 arrowstyle='->', ax=ax)
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
        
        conf = explanation.get('prediction_confidence', 0)
        pred_class = explanation.get('original_prediction', '?')
        ax.set_title(f"GNN Explanation: {explanation['drug_a']} ↔ {explanation['drug_b']}\n"
                    f"Prediction: Class {pred_class} (confidence: {conf:.3f})", 
                    fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        return fig

# ==================== CACHED LOADING ====================

@st.cache_resource
# def load_model_data():
#     with torch.serialization.safe_globals([Data]):
#         data = torch.load(DATA_PT, map_location=DEVICE)
#     with open(META_PATH, "rb") as f:
#         meta = pickle.load(f)
#     node2idx = meta["node2idx"]
#     idx2node = meta["idx2node"]
#     label_encoder = meta["label_encoder"]
#     df = meta["df"]

#     model = EdgeGNN(
#         num_nodes=data.num_nodes,
#         node_embed_dim=128,
#         hidden_dim=256,
#         num_classes=len(label_encoder.classes_)
#     ).to(DEVICE)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#     model.eval()

#     return model, data, node2idx, idx2node, label_encoder, df

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
    .stTextInput input, .stTextArea textarea {{
        color: #333333 !important;
        background-color: rgba(255, 255, 255, 0.9) !important;
    }}
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {{
        color: #666666 !important;
        opacity: 1 !important;
    }}
    .stSelectbox select {{
        color: #333333 !important;
        background-color: rgba(255, 255, 255, 0.9) !important;
    }}
    body, h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {{
        color: white !important;
    }}
    [data-testid="stHeader"] {{
        background-color: rgba(0,0,0,0.5) !important;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_page_background(BACKGROUND)

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
            if error:
                st.error(error)
                return

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
    inference_with_explanation()