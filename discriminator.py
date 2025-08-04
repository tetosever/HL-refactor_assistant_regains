from typing import Dict

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import (global_max_pool, GCNConv, GATConv, GraphNorm,
                                global_mean_pool, GINEConv, global_add_pool,
                                BatchNorm, LayerNorm)
import torch.nn.functional as F


class HubDetectionDiscriminator(nn.Module):
    """Enhanced discriminator for hub detection with improved architecture.

    By default expects 7 node features and 1 edge feature.
    """

    def __init__(self, node_dim: int = 7, edge_dim: int = 1, hidden_dim: int = 128,
                 num_layers: int = 4, dropout: float = 0.2, heads: int = 8):
        super().__init__()

        # Enhanced input projections with batch norm
        self.node_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        if edge_dim > 0:
            self.edge_projection = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim // 4),
                nn.ReLU()
            )
        else:
            self.edge_projection = None

        # Multi-scale graph convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.attention_weights = nn.ModuleList()

        # First layer - GCN for basic structure
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.norms.append(LayerNorm(hidden_dim))

        # Middle and final layers - All GAT for consistency and robustness
        for i in range(1, num_layers):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads,
                                      heads=heads, concat=True, dropout=dropout))
            self.norms.append(LayerNorm(hidden_dim))

        # Multi-level pooling for richer graph representations
        self.pooling_weights = nn.Parameter(torch.ones(3))  # mean, max, add

        # Enhanced hub detection heads with attention
        self.hub_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )

        # Graph-level classifier with residual connections
        classifier_input_dim = hidden_dim * 3  # multi-pooling
        self.hub_classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(hidden_dim // 2, 2)
        )

        # Enhanced node-level hub scoring with context
        self.node_context_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        self.node_hub_scorer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

        # Feature consistency regularizer
        self.feature_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        self.dropout = dropout
        self._init_weights()

    def _init_weights(self):
        """Improved weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization for better gradient flow
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Enhanced input projection
        x = self.node_projection(x)

        # Store intermediate representations for skip connections
        layer_outputs = [x]

        # Multi-scale graph convolutions with residual connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # All layers now use GCN or GAT - no edge features needed
            h = conv(x, edge_index)

            # Normalization
            h = norm(h)

            # Residual connection with proper dimensionality
            if h.size(-1) == x.size(-1):
                h = h + x  # Skip connection

            # Activation and dropout
            x = F.relu(h)
            x = F.dropout(x, p=self.dropout, training=self.training)

            layer_outputs.append(x)

        # Multi-level graph pooling with learnable weights
        pooling_weights = F.softmax(self.pooling_weights, dim=0)

        graph_mean = global_mean_pool(x, batch) * pooling_weights[0]
        graph_max = global_max_pool(x, batch) * pooling_weights[1]
        graph_add = global_add_pool(x, batch) * pooling_weights[2]

        graph_features = torch.cat([graph_mean, graph_max, graph_add], dim=1)

        # Enhanced node-level hub scoring with context
        # Get graph-level context for each node
        batch_size = batch.max().item() + 1
        graph_context = []

        for i in range(batch_size):
            mask = batch == i
            if mask.sum() > 0:
                ctx = self.node_context_encoder(x[mask].mean(dim=0, keepdim=True))
                graph_context.append(ctx.expand(mask.sum(), -1))

        if graph_context:
            graph_context = torch.cat(graph_context, dim=0)

            # Combine node features with graph context
            node_with_context = torch.cat([x, graph_context], dim=1)
            node_hub_scores = torch.sigmoid(self.node_hub_scorer(node_with_context).squeeze(-1))
        else:
            # Fallback if no valid batches
            node_hub_scores = torch.sigmoid(self.node_hub_scorer(
                torch.cat([x, torch.zeros(x.size(0), x.size(1) // 4, device=x.device)], dim=1)
            ).squeeze(-1))

        # Graph classification with enhanced features
        logits = self.hub_classifier(graph_features)

        # Feature consistency for regularization
        feature_projection = self.feature_projector(x)

        return {
            'logits': logits,
            'node_hub_scores': node_hub_scores,
            'graph_embedding': graph_features,
            'node_embeddings': x,
            'feature_projection': feature_projection,
            'layer_outputs': layer_outputs  # For potential auxiliary losses
        }