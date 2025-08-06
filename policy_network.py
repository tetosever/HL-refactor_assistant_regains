#!/usr/bin/env python3
"""
Version 2: Hub-Focused Policy Network with Smart Hub Selection
- Strongly biases selection toward actual hub nodes
- Uses sophisticated hub detection based on multiple centrality measures
- Maintains exploration capability while preferring hub nodes
- Adaptive exploration based on training progress
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (GINEConv, GATConv, GraphNorm,
                                global_mean_pool, global_max_pool, GCNConv)
from torch_geometric.utils import softmax

import logging
logger = logging.getLogger(__name__)


@dataclass
class HubRefactoringAction:
    """Enhanced action structure for hub refactoring"""
    source_node: int
    target_node: int
    pattern: int
    parameters: Dict[str, Any]
    terminate: int
    confidence: float = 0.0
    is_hub_focused: bool = True  # NEW: Track if action targets actual hub


class AdvancedHubDetector:
    """Advanced hub detection using multiple centrality measures"""

    @staticmethod
    def compute_hub_scores(x: torch.Tensor, edge_index: torch.Tensor,
                           batch: torch.Tensor) -> torch.Tensor:
        """
        Compute comprehensive hub scores using 7 structural features
        Returns scores in [0,1] where higher = more hub-like
        """
        # Extract the 7 structural features
        fan_in = x[:, 0]  # Incoming connections
        fan_out = x[:, 1]  # Outgoing connections
        degree_centrality = x[:, 2]  # Normalized degree
        in_out_ratio = x[:, 3]  # Fan-in/fan-out ratio
        pagerank = x[:, 4]  # PageRank centrality
        betweenness = x[:, 5]  # Betweenness centrality
        closeness = x[:, 6]  # Closeness centrality

        # Multi-criteria hub scoring
        # 1. High total degree (fan_in + fan_out)
        total_degree = fan_in + fan_out
        degree_score = torch.sigmoid(total_degree - 3.0)  # Threshold at 3 connections

        # 2. High outgoing connections (typical for hub classes)
        out_degree_score = torch.sigmoid(fan_out - 2.0)  # Threshold at 2 outgoing

        # 3. Balanced in/out ratio (not just sink or source)
        balanced_ratio = 1.0 - torch.abs(in_out_ratio - 1.0)  # Closer to 1.0 = more balanced
        balance_score = torch.clamp(balanced_ratio, 0.0, 1.0)

        # 4. High centrality measures
        pagerank_score = pagerank / (pagerank.max() + 1e-8)  # Normalize to [0,1]
        betweenness_score = betweenness / (betweenness.max() + 1e-8)
        closeness_score = closeness / (closeness.max() + 1e-8)

        # 5. High degree centrality
        degree_cent_score = degree_centrality

        # Weighted combination of all hub indicators
        hub_score = (
                0.25 * degree_score +  # Total connections
                0.20 * out_degree_score +  # Outgoing connections (hub characteristic)
                0.15 * balance_score +  # Balanced in/out
                0.15 * pagerank_score +  # PageRank importance
                0.10 * betweenness_score +  # Betweenness (bridging)
                0.10 * closeness_score +  # Closeness (centrality)
                0.05 * degree_cent_score  # Normalized degree
        )

        return torch.clamp(hub_score, 0.0, 1.0)

    @staticmethod
    def identify_top_hubs(hub_scores: torch.Tensor, batch: torch.Tensor,
                          top_k: int = 3) -> torch.Tensor:
        """Identify top-k hub nodes per graph"""
        batch_size = batch.max().item() + 1
        top_hub_mask = torch.zeros_like(hub_scores, dtype=torch.bool)

        for b in range(batch_size):
            mask = (batch == b)
            if mask.sum() > 0:
                graph_scores = hub_scores[mask]
                k = min(top_k, mask.sum().item())

                # Get indices of top-k hubs in this graph
                _, top_indices = torch.topk(graph_scores, k)

                # Convert back to global indices
                global_indices = torch.where(mask)[0][top_indices]
                top_hub_mask[global_indices] = True

        return top_hub_mask


class HubRefactoringPatterns:
    """Hub-specific refactoring patterns with applicability scoring"""

    PATTERNS = {
        0: "SPLIT_RESPONSIBILITY",  # Best for large hubs with many responsibilities
        1: "EXTRACT_INTERFACE",  # Best for hubs with many clients
        2: "DEPENDENCY_INJECTION",  # Best for hubs with many dependencies
        3: "EXTRACT_SUPERCLASS",  # Best for hubs with common behavior
        4: "MOVE_METHOD"  # Best for hubs with movable responsibilities
    }

    @staticmethod
    def compute_pattern_applicability(hub_scores: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute applicability scores for each pattern given hub characteristics
        Returns: [num_nodes, num_patterns] tensor with applicability scores
        """
        num_nodes = x.size(0)
        num_patterns = len(HubRefactoringPatterns.PATTERNS)

        fan_in = x[:, 0]
        fan_out = x[:, 1]
        total_degree = fan_in + fan_out

        applicability = torch.zeros(num_nodes, num_patterns, device=x.device)

        # Pattern 0: Split Responsibility - better for highly connected hubs
        applicability[:, 0] = hub_scores * torch.sigmoid(total_degree - 5.0)

        # Pattern 1: Extract Interface - better for hubs with many outgoing connections
        applicability[:, 1] = hub_scores * torch.sigmoid(fan_out - 3.0)

        # Pattern 2: Dependency Injection - better for hubs with many incoming dependencies
        applicability[:, 2] = hub_scores * torch.sigmoid(fan_in - 3.0)

        # Pattern 3: Extract Superclass - better for balanced hubs
        balance_factor = 1.0 - torch.abs((fan_in - fan_out) / (total_degree + 1e-8))
        applicability[:, 3] = hub_scores * balance_factor

        # Pattern 4: Move Method - better for moderately connected hubs
        moderate_connection = torch.sigmoid(total_degree - 2.0) * torch.sigmoid(6.0 - total_degree)
        applicability[:, 4] = hub_scores * moderate_connection

        return applicability


class HubRefactoringPolicy(nn.Module):
    """Version 2: Hub-focused policy network with smart hub selection"""

    def __init__(self, node_dim: int = 7, edge_dim: int = 1, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.3,
                 hub_bias_strength: float = 3.0, exploration_rate: float = 0.15):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_patterns = 5
        self.node_dim = node_dim
        self.hub_bias_strength = hub_bias_strength
        self.exploration_rate = exploration_rate

        # Adaptive exploration
        self.register_buffer('training_progress', torch.tensor(0.0))

        # Feature embedding
        self.node_embed = nn.Linear(node_dim, hidden_dim)

        # Simplified GNN backbone (no attention)
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
            nn.Linear(hidden_dim + 7, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Hub-aware node selector
        self.hub_selector = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim // 2),  # +2 for importance and hub_score
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # PATTERN SELECTOR SEMPLIFICATO - dimensioni fisse
        self.pattern_selector = nn.Sequential(
            nn.Linear(hidden_dim + 7, hidden_dim),  # Solo node features + structural
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_patterns)
        )

        # Target selector semplificato
        self.target_selector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Termination head semplificato
        self.termination_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

        self._init_weights()

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def update_training_progress(self, progress: float):
        self.training_progress.fill_(torch.clamp(torch.tensor(progress), 0.0, 1.0))

    def compute_structural_features(self, x: torch.Tensor) -> torch.Tensor:
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

        if x.size(1) != self.node_dim:
            raise ValueError(f"Expected {self.node_dim} node features, got {x.size(1)}")

        # Extract structural features
        structural_features = self.compute_structural_features(x)

        # Compute hub scores
        hub_scores = AdvancedHubDetector.compute_hub_scores(x, edge_index, batch)
        top_hub_mask = AdvancedHubDetector.identify_top_hubs(hub_scores, batch, top_k=3)

        # Embed features
        x_emb = self.node_embed(x)

        # Simplified GNN processing
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x_emb, edge_index)
            h = norm(h)
            x_emb = F.relu(h) + x_emb
            x_emb = F.dropout(x_emb, p=0.3, training=self.training)

        # Hub importance
        hub_input = torch.cat([x_emb, structural_features], dim=-1)
        learned_importance = self.hub_importance_net(hub_input).squeeze(-1)
        combined_importance = 0.6 * learned_importance + 0.4 * hub_scores

        # Hub selection with bias
        current_exploration = self.exploration_rate * (1.0 - self.training_progress)
        hub_features = torch.cat([x_emb, combined_importance.unsqueeze(-1), hub_scores.unsqueeze(-1)], dim=-1)
        hub_logits = self.hub_selector(hub_features).squeeze(-1)

        hub_bias = self.hub_bias_strength * hub_scores
        exploration_noise = current_exploration * torch.randn_like(hub_logits) * 0.1
        hub_logits = hub_logits + hub_bias + exploration_noise + 2.0 * top_hub_mask.float()
        hub_probs = softmax(hub_logits, batch)

        # Simplified pattern selection
        pattern_input = torch.cat([x_emb, structural_features], dim=-1)
        pattern_logits = self.pattern_selector(pattern_input)

        # Apply hub scores as bias to patterns
        pattern_bias = hub_scores.unsqueeze(-1).expand(-1, self.num_patterns) * 0.5
        pattern_logits = pattern_logits + pattern_bias
        pattern_probs = F.softmax(pattern_logits, dim=-1)

        # Graph-level features for termination
        graph_mean = global_mean_pool(x_emb, batch)
        graph_max = global_max_pool(x_emb, batch)
        graph_features = torch.cat([graph_mean, graph_max], dim=-1)

        # Simple target selection
        target_logits = torch.zeros_like(hub_logits)

        # Termination decision
        term_logits = self.termination_head(graph_features)
        term_probs = F.softmax(term_logits, dim=-1)

        value = self.value_head(graph_features)

        return {
            'value': value,
            'hub_logits': hub_logits,
            'hub_probs': hub_probs,
            'pattern_logits': pattern_logits,
            'pattern_probs': pattern_probs,
            'target_logits': target_logits,
            'term_logits': term_logits,
            'term_probs': term_probs,
            'hub_importance': combined_importance,
            'structural_hub_scores': hub_scores,
            'top_hub_mask': top_hub_mask,
            'node_embeddings': x_emb
        }

    def set_hub_bias_strength(self, strength: float):
        self.hub_bias_strength = max(0.0, strength)

    def set_exploration_rate(self, rate: float):
        self.exploration_rate = torch.clamp(torch.tensor(rate), 0.0, 1.0).item()

    def get_hub_analysis(self, data: Data) -> Dict[str, Any]:
        """Analyze hub characteristics of the current graph"""
        with torch.no_grad():
            output = self.forward(data)
            hub_scores = output['structural_hub_scores']
            top_hubs = output['top_hub_mask']

            # Find top hub nodes
            top_hub_indices = torch.where(top_hubs)[0].tolist()

            # Analyze structural characteristics
            x = data.x
            analysis = {
                'num_nodes': x.size(0),
                'top_hubs': top_hub_indices,
                'hub_scores': hub_scores.tolist(),
                'avg_hub_score': hub_scores.mean().item(),
                'max_hub_score': hub_scores.max().item(),
                'min_hub_score': hub_scores.min().item(),
                'num_strong_hubs': (hub_scores > 0.7).sum().item(),
                'num_moderate_hubs': ((hub_scores > 0.4) & (hub_scores <= 0.7)).sum().item(),
                'num_weak_hubs': (hub_scores <= 0.4).sum().item()
            }

            # Analyze top hubs in detail
            if top_hub_indices:
                top_hub_details = []
                for hub_idx in top_hub_indices:
                    hub_features = x[hub_idx]
                    hub_detail = {
                        'node_id': hub_idx,
                        'hub_score': hub_scores[hub_idx].item(),
                        'fan_in': hub_features[0].item(),
                        'fan_out': hub_features[1].item(),
                        'degree_centrality': hub_features[2].item(),
                        'pagerank': hub_features[4].item(),
                        'betweenness': hub_features[5].item(),
                        'total_degree': (hub_features[0] + hub_features[1]).item()
                    }
                    top_hub_details.append(hub_detail)

                analysis['top_hub_details'] = top_hub_details

            return analysis

    def get_hub_selection_stats(self, data: Data) -> Dict[str, float]:
        """Get statistics about hub selection probabilities"""
        with torch.no_grad():
            output = self.forward(data)
            hub_probs = output['hub_probs']
            hub_scores = output['structural_hub_scores']
            top_hubs = output['top_hub_mask']

            # Calculate selection probabilities for different hub categories
            strong_hubs = hub_scores > 0.7
            moderate_hubs = (hub_scores > 0.4) & (hub_scores <= 0.7)
            weak_hubs = hub_scores <= 0.4

            stats = {
                'prob_select_strong_hub': hub_probs[strong_hubs].sum().item() if strong_hubs.any() else 0.0,
                'prob_select_moderate_hub': hub_probs[moderate_hubs].sum().item() if moderate_hubs.any() else 0.0,
                'prob_select_weak_hub': hub_probs[weak_hubs].sum().item() if weak_hubs.any() else 0.0,
                'prob_select_top_hub': hub_probs[top_hubs].sum().item() if top_hubs.any() else 0.0,
                'entropy': -torch.sum(hub_probs * torch.log(hub_probs + 1e-8)).item(),
                'max_prob': hub_probs.max().item(),
                'min_prob': hub_probs.min().item(),
                'concentration_ratio': (hub_probs > hub_probs.mean()).sum().item() / hub_probs.size(0)
            }

            return stats

    def log_hub_focus_info(self, data: Data) -> str:
        """Generate detailed logging information about hub focus"""
        analysis = self.get_hub_analysis(data)
        stats = self.get_hub_selection_stats(data)

        log_info = []
        log_info.append(f"🎯 HUB ANALYSIS - Graph with {analysis['num_nodes']} nodes:")
        log_info.append(f"   Strong hubs (>0.7): {analysis['num_strong_hubs']}")
        log_info.append(f"   Moderate hubs (0.4-0.7): {analysis['num_moderate_hubs']}")
        log_info.append(f"   Weak hubs (<0.4): {analysis['num_weak_hubs']}")
        log_info.append(f"   Top identified hubs: {analysis['top_hubs']}")

        log_info.append(f"📊 SELECTION PROBABILITIES:")
        log_info.append(f"   Strong hubs: {stats['prob_select_strong_hub']:.3f}")
        log_info.append(f"   Top hubs: {stats['prob_select_top_hub']:.3f}")
        log_info.append(f"   Selection entropy: {stats['entropy']:.3f}")
        log_info.append(f"   Bias strength: {self.hub_bias_strength}")
        log_info.append(f"   Exploration rate: {self.exploration_rate:.3f}")

        return "\n".join(log_info)

    def get_pattern_info(self) -> Dict[int, str]:
        """Get information about the 5 available patterns"""
        return {
            0: "Split Responsibility (Move Class) - Divide hub responsibilities into separate classes",
            1: "Extract Interface - Decouple hub from clients through interfaces",
            2: "Dependency Injection - Remove direct dependencies from hub",
            3: "Extract Superclass (Pull Up) - Factor out common dependencies to superclass",
            4: "Move Method - Redistribute methods from hub to appropriate classes"
        }