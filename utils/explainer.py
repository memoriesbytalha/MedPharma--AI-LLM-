import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch_geometric.utils import k_hop_subgraph

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