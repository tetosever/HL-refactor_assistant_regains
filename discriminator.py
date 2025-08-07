#!/usr/bin/env python3
"""
Clean Hub Detection Discriminator
Simplified version focusing on performance over complexity.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, GraphNorm


class HubDiscriminator(nn.Module):
    """
    Clean discriminator for hub detection using simple GCN architecture.
    Maintains the performance of the original simple version while adding
    useful features like node-level scoring and better error handling.
    """

    def __init__(self,
                 node_dim: int = 7,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 add_node_scoring: bool = True):
        super().__init__()

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.add_node_scoring = add_node_scoring

        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        # GCN backbone with skip connections
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(GraphNorm(hidden_dim))

        # Graph-level classifier (main output)
        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 2)
        )

        # Optional node-level hub scoring
        if add_node_scoring:
            self.node_scorer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        else:
            self.node_scorer = None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Forward pass with comprehensive output

        Args:
            data: PyTorch Geometric Data object with node features and edges

        Returns:
            Dictionary containing:
            - logits: Graph-level classification logits [batch_size, 2]
            - probs: Graph-level classification probabilities [batch_size, 2]
            - node_embeddings: Node representations [num_nodes, hidden_dim]
            - graph_embedding: Graph representations [batch_size, hidden_dim]
            - node_hub_scores: Optional node-level hub scores [num_nodes] if add_node_scoring=True
        """
        x = data.x
        edge_index = data.edge_index

        # Handle batch dimension
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Validate input dimensions
        if x.size(1) != self.node_dim:
            raise ValueError(f"Expected {self.node_dim} node features, got {x.size(1)}")

        # Input projection
        x = self.input_proj(x)

        # GCN layers with residual connections
        for conv, norm in zip(self.convs, self.norms):
            # Apply convolution and normalization
            h = conv(x, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # Skip connection
            x = x + h

        # Global pooling for graph representation
        graph_emb = global_mean_pool(x, batch)

        # Graph-level classification
        logits = self.graph_classifier(graph_emb)
        probs = F.softmax(logits, dim=-1)

        # Prepare output dictionary
        output = {
            'logits': logits,
            'probs': probs,
            'graph_embedding': graph_emb,
            'node_embeddings': x
        }

        # Optional node-level hub scoring
        if self.node_scorer is not None:
            node_hub_scores = self.node_scorer(x).squeeze(-1)
            output['node_hub_scores'] = node_hub_scores

        return output

    def classify_graph(self, data: Data) -> Dict[str, float]:
        """
        Classify a single graph and return interpretable results

        Returns:
            Dictionary with classification results and confidence scores
        """
        with torch.no_grad():
            output = self.forward(data)
            probs = output['probs'][0]  # Assume single graph

            return {
                'is_hub_smell': probs[1].item() > 0.5,
                'confidence': probs.max().item(),
                'hub_smell_prob': probs[1].item(),
                'normal_prob': probs[0].item()
            }

    def get_node_hub_analysis(self, data: Data) -> Optional[Dict[str, any]]:
        """
        Analyze node-level hub characteristics (if node scoring is enabled)

        Returns:
            Dictionary with node-level analysis or None if node scoring is disabled
        """
        if not self.add_node_scoring:
            return None

        with torch.no_grad():
            output = self.forward(data)
            node_scores = output['node_hub_scores']

            # Find top hub candidates
            top_k = min(3, len(node_scores))
            top_values, top_indices = torch.topk(node_scores, top_k)

            return {
                'num_nodes': len(node_scores),
                'avg_hub_score': node_scores.mean().item(),
                'max_hub_score': node_scores.max().item(),
                'min_hub_score': node_scores.min().item(),
                'top_hub_candidates': [
                    {
                        'node_id': idx.item(),
                        'hub_score': score.item(),
                        'node_features': data.x[idx].tolist()
                    }
                    for idx, score in zip(top_indices, top_values)
                ],
                'strong_hubs_count': (node_scores > 0.7).sum().item(),
                'moderate_hubs_count': ((node_scores > 0.4) & (node_scores <= 0.7)).sum().item()
            }


class SimpleHubDiscriminator(nn.Module):
    """
    Ultra-simple version for maximum performance - exactly like the original working version
    """

    def __init__(self,
                 node_dim: int = 7,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.5):
        super().__init__()

        self.dropout = dropout

        # Input projection
        self.input_lin = nn.Linear(node_dim, hidden_dim)

        # GCN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(GraphNorm(hidden_dim))

        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data: Data) -> torch.Tensor:
        """Simple forward pass returning only logits"""
        x = data.x
        edge_index = data.edge_index

        # Handle batch
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Input projection
        x = self.input_lin(x)

        # GCN layers with skip connections
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h  # Skip connection

        # Global pooling and classification
        graph_emb = global_mean_pool(x, batch)
        return self.classifier(graph_emb)


# Factory function for easy instantiation
def create_discriminator(version: str = "clean", **kwargs) -> nn.Module:
    """
    Factory function to create discriminator instances

    Args:
        version: "clean" for full-featured version, "simple" for minimal version
        **kwargs: Additional arguments passed to the discriminator constructor

    Returns:
        Discriminator instance
    """
    if version == "simple":
        return SimpleHubDiscriminator(**kwargs)
    elif version == "clean":
        return HubDiscriminator(**kwargs)
    else:
        raise ValueError(f"Unknown version: {version}. Choose 'clean' or 'simple'")


# Usage examples and testing
if __name__ == "__main__":
    # Create dummy graph data
    num_nodes = 10
    node_features = torch.randn(num_nodes, 7)  # 7 structural features
    edge_index = torch.randint(0, num_nodes, (2, 20))  # Random edges

    test_data = Data(x=node_features, edge_index=edge_index)

    # Test both versions
    print("Testing Simple Discriminator...")
    simple_disc = create_discriminator("simple", node_dim=7)
    simple_out = simple_disc(test_data)
    print(f"Simple output shape: {simple_out.shape}")

    print("\nTesting Clean Discriminator...")
    clean_disc = create_discriminator("clean", node_dim=7, add_node_scoring=True)
    clean_out = clean_disc(test_data)
    print(f"Clean output keys: {list(clean_out.keys())}")

    # Test classification method
    classification = clean_disc.classify_graph(test_data)
    print(f"Classification result: {classification}")

    # Test node analysis
    node_analysis = clean_disc.get_node_hub_analysis(test_data)
    if node_analysis:
        print(f"Node analysis: {node_analysis['strong_hubs_count']} strong hubs found")