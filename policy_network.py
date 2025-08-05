#!/usr/bin/env python3
"""
Version 1: Policy Network Adapted for 5 Hub-Focused Refactoring Patterns
- Updated pattern count from 6 to 5
- Maintained existing hub detection logic
- Compatible with current pattern set in rl_gym.py
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
    """Updated catalog of 5 hub-specific refactoring patterns"""

    PATTERNS = {
        0: "SPLIT_RESPONSIBILITY",    # Split hub by responsibilities (Move Class)
        1: "EXTRACT_INTERFACE",      # Extract interface to decouple hub
        2: "DEPENDENCY_INJECTION",   # Break direct dependencies
        3: "EXTRACT_SUPERCLASS",     # Pull up common dependencies
        4: "MOVE_METHOD"             # Move methods from hub to other nodes
    }

    @staticmethod
    def get_applicable_patterns(hub_node: int, graph: nx.DiGraph) -> List[int]:
        """Determine which patterns are applicable for a given hub"""
        applicable = []

        in_degree = graph.in_degree(hub_node)
        out_degree = graph.out_degree(hub_node)
        total_degree = in_degree + out_degree

        # Pattern applicability rules (updated for 5 patterns)
        if out_degree > 3:  # Many outgoing connections
            applicable.extend([0, 1])  # Split Responsibility, Extract Interface

        if in_degree > 3:  # Many incoming dependencies
            applicable.extend([2])  # Dependency Injection

        if total_degree > 6:  # Very connected hub
            applicable.extend([0, 3])  # Split Responsibility, Extract Superclass

        if out_degree > 2 and in_degree > 1:  # Good candidate for method movement
            applicable.append(4)  # Move Method

        if out_degree > 1 and in_degree > 1:  # Can extract common dependencies
            applicable.append(3)  # Extract Superclass

        return list(set(applicable)) if applicable else [0]  # Default to split responsibility


class HubRefactoringPolicy(nn.Module):
    """Version 1: Policy network adapted for 5 hub-focused refactoring patterns"""

    def __init__(self, node_dim: int = 7, edge_dim: int = 1, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_patterns = 5  # UPDATED: Changed from 6 to 5 patterns
        self.node_dim = node_dim

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

        # Hub importance computation
        self.hub_importance_net = nn.Sequential(
            nn.Linear(hidden_dim + 7, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Action selection heads
        self.hub_selector = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Pattern selector - UPDATED for 5 patterns
        pattern_input_dim = hidden_dim + (hidden_dim * 2) + 7
        self.pattern_selector = nn.Sequential(
            nn.Linear(pattern_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_patterns)  # UPDATED: Output 5 patterns instead of 6
        )

        self.target_selector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
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
        """Extract all 7 structural features for hub identification"""
        if x.size(1) >= 7:
            structural = x[:, :7]
        else:
            padding = torch.zeros(x.size(0), 7 - x.size(1), device=x.device)
            structural = torch.cat([x, padding], dim=1)
        return structural

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Validate input dimensions
        if x.size(1) != self.node_dim:
            raise ValueError(f"Expected {self.node_dim} node features, got {x.size(1)}")

        # Extract structural features
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
        graph_features = torch.cat([graph_mean, graph_max], dim=-1)

        # Pattern selection (per node, considering graph context)
        batch_size = batch.max().item() + 1
        pattern_logits_list = []

        for b in range(batch_size):
            mask = (batch == b)
            node_features = x_emb[mask]
            node_structural = structural_features[mask]
            graph_feat = graph_features[b].unsqueeze(0).expand(mask.sum(), -1)

            # Combine features
            pattern_input = torch.cat([node_features, graph_feat, node_structural], dim=-1)
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

    def get_pattern_info(self) -> Dict[int, str]:
        """Get information about the 5 available patterns"""
        return {
            0: "Split Responsibility (Move Class) - Divide hub responsibilities into separate classes",
            1: "Extract Interface - Decouple hub from clients through interfaces",
            2: "Dependency Injection - Remove direct dependencies from hub",
            3: "Extract Superclass (Pull Up) - Factor out common dependencies to superclass",
            4: "Move Method - Redistribute methods from hub to appropriate classes"
        }