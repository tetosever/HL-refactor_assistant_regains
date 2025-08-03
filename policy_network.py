#!/usr/bin/env python3
"""
Fixed Policy Network for Hub Refactoring with Correct Dimensional Handling
"""

from dataclasses import dataclass
from typing import List, Dict, Any

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (GINEConv, GATConv, GraphNorm,
                                global_mean_pool, global_max_pool, GCNConv)
from torch_geometric.utils import softmax


@dataclass
class HubRefactoringAction:
    """Enhanced action structure for hub refactoring"""
    source_node: int
    target_node: int
    pattern: int
    parameters: Dict[str, Any]
    terminate: int
    confidence: float = 0.0


class HubRefactoringPatterns:
    """Catalog of hub-specific refactoring patterns"""

    PATTERNS = {
        0: "EXTRACT_INTERFACE",  # Create interface between hub and dependents
        1: "DEPENDENCY_INJECTION",  # Inject dependencies instead of hub coupling
        2: "SPLIT_BY_RESPONSIBILITY",  # Split hub by different responsibilities
        3: "OBSERVER_PATTERN",  # Replace hub with observer pattern
        4: "STRATEGY_PATTERN",  # Extract strategies from hub
        5: "REMOVE_MIDDLEMAN"  # Remove unnecessary hub intermediary
    }

    @staticmethod
    def get_applicable_patterns(hub_node: int, graph: nx.DiGraph) -> List[int]:
        """Determine which patterns are applicable for a given hub"""
        applicable = []

        in_degree = graph.in_degree(hub_node)
        out_degree = graph.out_degree(hub_node)
        total_degree = in_degree + out_degree

        # Pattern applicability rules
        if out_degree > 3:  # Many dependents
            applicable.extend([0, 3])  # Interface, Observer

        if in_degree > 3:  # Many dependencies
            applicable.extend([1])  # DI

        if total_degree > 6:  # Very connected
            applicable.extend([2])  # Split

        if out_degree > 2 and in_degree > 2:
            applicable.append(4)  # Strategy

        if HubRefactoringPatterns._is_pure_middleman(hub_node, graph):
            applicable.append(5)  # Remove middleman

        return list(set(applicable)) if applicable else [0]  # Default to extract interface

    @staticmethod
    def _is_pure_middleman(node: int, graph: nx.DiGraph) -> bool:
        """Check if node is just passing through connections"""
        predecessors = set(graph.predecessors(node))
        successors = set(graph.successors(node))

        if len(predecessors) == 0 or len(successors) == 0:
            return False

        # Check if there's significant overlap in connections
        for pred in predecessors:
            pred_successors = set(graph.successors(pred))
            if len(pred_successors & successors) > len(successors) * 0.5:
                return True

        return False


class HubRefactoringPolicy(nn.Module):
    """Fixed policy network for hub-reducing refactoring patterns"""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_patterns = len(HubRefactoringPatterns.PATTERNS)

        # Feature embedding
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim) if edge_dim > 0 else None

        # Graph neural network backbone
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            if i % 2 == 0:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            else:
                self.convs.append(GATConv(hidden_dim, hidden_dim // 8, heads=8, concat=True))
            self.norms.append(GraphNorm(hidden_dim))

        # Hub importance computation (fixed dimensions)
        # Input: node features (hidden_dim) + structural features (6)
        self.hub_importance_net = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Action selection heads (fixed dimensions)
        self.hub_selector = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),  # +1 for importance score
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Pattern selector (fixed dimensions)
        # Input: node features (hidden_dim) + graph features (hidden_dim * 2) + structural (6)
        pattern_input_dim = hidden_dim + (hidden_dim * 2) + 6
        self.pattern_selector = nn.Sequential(
            nn.Linear(pattern_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_patterns)
        )

        self.target_selector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # hub + candidate features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.termination_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # continue vs terminate
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compute_structural_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract key structural features for hub identification"""
        # Ensure we have at least 6 features, pad if necessary
        if x.size(1) < 6:
            # Pad with zeros if we don't have enough features
            padding = torch.zeros(x.size(0), 6 - x.size(1), device=x.device)
            structural = torch.cat([x, padding], dim=1)[:, :6]
        else:
            # Take first 6 features: [in_deg, out_deg, total_deg, in_out_ratio, deg_centrality, pagerank]
            structural = x[:, :6]

        return structural

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Extract structural features (fixed size: 6)
        structural_features = self.compute_structural_features(x)

        # Embed features
        x_emb = self.node_embed(x)

        # GNN processing
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x_emb, edge_index)
            h = norm(h)
            x_emb = F.relu(h) + x_emb  # Residual
            x_emb = F.dropout(x_emb, p=0.3, training=self.training)

        # Compute hub importance scores
        hub_input = torch.cat([x_emb, structural_features], dim=-1)
        hub_importance = self.hub_importance_net(hub_input).squeeze(-1)

        # Hub selection (weighted by importance)
        hub_features = torch.cat([x_emb, hub_importance.unsqueeze(-1)], dim=-1)
        hub_logits = self.hub_selector(hub_features).squeeze(-1)
        hub_logits = hub_logits + 2.0 * hub_importance  # Bias toward important hubs
        hub_probs = softmax(hub_logits, batch)

        # Graph-level features for pattern selection
        graph_mean = global_mean_pool(x_emb, batch)
        graph_max = global_max_pool(x_emb, batch)
        graph_features = torch.cat([graph_mean, graph_max], dim=-1)  # Size: hidden_dim * 2

        # Pattern selection (per node, considering graph context)
        batch_size = batch.max().item() + 1
        pattern_logits_list = []

        for b in range(batch_size):
            mask = (batch == b)
            node_features = x_emb[mask]  # Size: [num_nodes_in_graph, hidden_dim]
            node_structural = structural_features[mask]  # Size: [num_nodes_in_graph, 6]
            graph_feat = graph_features[b].unsqueeze(0).expand(mask.sum(),
                                                               -1)  # Size: [num_nodes_in_graph, hidden_dim*2]

            # Combine features (fixed dimensions)
            pattern_input = torch.cat([node_features, graph_feat, node_structural], dim=-1)
            # Size: [num_nodes_in_graph, hidden_dim + hidden_dim*2 + 6]

            pattern_logits = self.pattern_selector(pattern_input)
            pattern_logits_list.append(pattern_logits)

        pattern_logits = torch.cat(pattern_logits_list, dim=0)
        pattern_probs = F.softmax(pattern_logits, dim=-1)

        # Target selection (simplified for now)
        target_logits = torch.zeros_like(hub_logits)

        # Termination decision
        term_logits = self.termination_head(graph_features)
        term_probs = F.softmax(term_logits, dim=-1)

        return {
            'hub_logits': hub_logits,
            'hub_probs': hub_probs,
            'pattern_logits': pattern_logits,
            'pattern_probs': pattern_probs,
            'target_logits': target_logits,
            'term_logits': term_logits,
            'term_probs': term_probs,
            'hub_importance': hub_importance,
            'node_embeddings': x_emb
        }