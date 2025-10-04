import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class EdgeGNN(nn.Module):
    def __init__(self, num_nodes, node_feat_dim=128, node_embed_dim=256, hidden_dim=512, num_classes=4, dropout=0.3, num_layers=3):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, node_embed_dim)
        self.input_proj = nn.Linear(node_embed_dim, hidden_dim)  # <-- project for residual

        self.convs = nn.ModuleList()
        in_dim = hidden_dim
        for i in range(num_layers):
            self.convs.append(SAGEConv(in_dim, hidden_dim))

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        device = next(self.parameters()).device
        x = self.node_emb(torch.arange(data.num_nodes, device=device))
        x = self.input_proj(x)  # project to hidden_dim

        for conv in self.convs:
            h = conv(x, data.edge_index)
            x = F.relu(h) + x  # residual now works
            x = F.dropout(x, p=0.2, training=self.training)

        src = data.edge_index[0]
        dst = data.edge_index[1]
        edge_feat = torch.cat([x[src], x[dst]], dim=1)
        logits = self.edge_mlp(edge_feat)
        return logits, x

