from typing import Dict

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import global_max_pool, GCNConv, GATConv, GraphNorm, global_mean_pool, GINEConv
import torch.nn.functional as F


class HubDetectionDiscriminator(nn.Module):
    """Enhanced discriminator for hub detection with better architecture"""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()

        self.node_projection = nn.Linear(node_dim, hidden_dim)
        self.edge_projection = nn.Linear(edge_dim, hidden_dim) if edge_dim > 0 else None

        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            else:
                self.convs.append(GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True))
            self.norms.append(GraphNorm(hidden_dim))

        # Hub detection heads
        self.hub_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # binary: clean vs hub-like
        )

        # Node-level hub scoring
        self.node_hub_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Project features
        x = self.node_projection(x)

        # Graph convolutions
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            h = norm(h)
            x = F.relu(h) + x  # Residual connection
            x = F.dropout(x, p=0.3, training=self.training)

        # Node-level hub scores
        node_hub_scores = torch.sigmoid(self.node_hub_scorer(x).squeeze(-1))

        # Graph-level features
        graph_mean = global_mean_pool(x, batch)
        graph_max = global_max_pool(x, batch)
        graph_features = torch.cat([graph_mean, graph_max], dim=1)

        # Graph classification
        logits = self.hub_classifier(graph_features)

        return {
            'logits': logits,
            'node_hub_scores': node_hub_scores,
            'graph_embedding': graph_features,
            'node_embeddings': x
        }