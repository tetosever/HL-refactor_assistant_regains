#!/usr/bin/env python3
"""
Actor-Critic GNN Models for Graph Refactoring - CORRECTED for PPO
Maintains original GNN architecture but fixes PPO compatibility issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, GraphNorm
from typing import Dict, Tuple, Optional
import numpy as np


class GraphEncoder(nn.Module):
    """
    Shared GCN encoder - MAINTAINS ORIGINAL ARCHITECTURE
    """

    def __init__(self,
                 node_dim: int = 7,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Input projection - UNCHANGED
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        # GCN layers with skip connections - UNCHANGED
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(GraphNorm(hidden_dim))

        # Global pooling layers - UNCHANGED
        self.global_pool_mean = global_mean_pool
        self.global_pool_max = global_max_pool

        # CORRECTED: Proper initialization for PPO
        self._init_weights()

    def _init_weights(self):
        """CORRECTED: Orthogonal initialization for better PPO performance"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Forward pass - MAINTAINS ORIGINAL STRUCTURE
        """
        x = data.x
        edge_index = data.edge_index

        # Handle batch dimension - UNCHANGED
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Input projection - UNCHANGED
        x = self.input_proj(x)
        x = F.relu(x)

        # GCN layers with skip connections - UNCHANGED
        for conv, norm in zip(self.convs, self.norms):
            h = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + h  # Skip connection

        # Global pooling - UNCHANGED
        graph_emb_mean = self.global_pool_mean(x, batch)
        graph_emb_max = self.global_pool_max(x, batch)

        # Concatenate mean and max pooling - UNCHANGED
        graph_emb = torch.cat([graph_emb_mean, graph_emb_max], dim=-1)

        return {
            'node_embeddings': x,
            'graph_embedding': graph_emb
        }


class ActorCritic(nn.Module):
    """
    CORRECTED: Combined Actor-Critic with FORCED shared encoder for PPO
    Maintains all original functionality but fixes PPO compatibility
    """

    def __init__(self,
                 node_dim: int = 7,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_actions: int = 7,
                 global_features_dim: int = 4,  # CORRECTED: From original 10 to 4
                 dropout: float = 0.2,
                 shared_encoder: bool = True):  # CORRECTED: Always True for PPO
        super().__init__()

        self.num_actions = num_actions
        # CORRECTED: Force shared encoder for proper PPO implementation
        self.shared_encoder = True

        # CORRECTED: Single shared encoder (no separate encoders)
        self.encoder = GraphEncoder(node_dim, hidden_dim, num_layers, dropout)

        # Graph embedding dimension after concatenating mean and max pooling
        graph_emb_dim = hidden_dim * 2
        combined_dim = graph_emb_dim + global_features_dim

        # Actor head - MAINTAINS ORIGINAL ARCHITECTURE
        self.actor_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_actions)
        )

        # Critic head - MAINTAINS ORIGINAL ARCHITECTURE
        self.critic_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1)  # Only state value output
        )

        # CORRECTED: Proper initialization for PPO
        self._init_weights()

    def _init_weights(self):
        """CORRECTED: Orthogonal initialization for PPO"""

        def init_layer(layer):
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Initialize both heads
        for layer in self.actor_head:
            init_layer(layer)
        for layer in self.critic_head:
            init_layer(layer)

        # Special initialization for output layers
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)  # Smaller init for policy
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)

    def forward(self, data: Data, global_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        CORRECTED: Forward pass using ONLY shared encoder
        """
        # CORRECTED: Use single shared encoder
        encoder_out = self.encoder(data)
        graph_emb = encoder_out['graph_embedding']

        # Combine graph embeddings with global features - UNCHANGED
        combined_features = torch.cat([graph_emb, global_features], dim=-1)

        # Actor forward pass - UNCHANGED
        action_logits = self.actor_head(combined_features)
        action_probs = F.softmax(action_logits, dim=-1)

        # Critic forward pass - UNCHANGED
        state_value = self.critic_head(combined_features).squeeze(-1)

        return {
            'action_logits': action_logits,
            'action_probs': action_probs,
            'state_value': state_value,
            'graph_embedding': graph_emb,
            'node_embeddings': encoder_out['node_embeddings']
        }

    def get_action_and_value(self, data: Data, global_features: torch.Tensor) -> Tuple[int, float, float]:
        """
        Sample action and get state value - MAINTAINS ORIGINAL INTERFACE
        """
        with torch.no_grad():
            output = self.forward(data, global_features)

            # Sample action
            action_dist = torch.distributions.Categorical(output['action_probs'])
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            return action.item(), log_prob.item(), output['state_value'].item()

    def evaluate_actions(self, data: Data, global_features: torch.Tensor,
                         actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        CORRECTED: Key method for PPO training
        Evaluates actions for PPO update
        """
        output = self.forward(data, global_features)

        # Create action distribution
        action_dist = torch.distributions.Categorical(output['action_probs'])

        # Get log probabilities of taken actions
        log_probs = action_dist.log_prob(actions)

        # Calculate entropy for regularization
        entropy = action_dist.entropy()

        return log_probs, output['state_value'], entropy

    # MAINTAIN ORIGINAL METHODS for compatibility
    def sample_action(self, data: Data, global_features: torch.Tensor) -> Tuple[int, float]:
        """MAINTAINED: Original sampling method"""
        with torch.no_grad():
            output = self.forward(data, global_features)
            action_probs = output['action_probs']

            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            return action.item(), log_prob.item()

    def get_action_log_probs(self, data: Data, global_features: torch.Tensor,
                             actions: torch.Tensor) -> torch.Tensor:
        """MAINTAINED: Original log prob method"""
        output = self.forward(data, global_features)
        action_probs = output['action_probs']

        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)

        return log_probs


# MAINTAINED: Original separate classes for compatibility
class Actor(nn.Module):
    """
    MAINTAINED: Original Actor class for compatibility
    """

    def __init__(self,
                 node_dim: int = 7,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_actions: int = 7,
                 global_features_dim: int = 4,  # CORRECTED: From 10 to 4
                 dropout: float = 0.2):
        super().__init__()

        self.num_actions = num_actions
        self.encoder = GraphEncoder(node_dim, hidden_dim, num_layers, dropout)

        graph_emb_dim = hidden_dim * 2
        combined_dim = graph_emb_dim + global_features_dim

        self.policy_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_actions)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.policy_head:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data: Data, global_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoder_out = self.encoder(data)
        graph_emb = encoder_out['graph_embedding']

        combined_features = torch.cat([graph_emb, global_features], dim=-1)
        action_logits = self.policy_head(combined_features)
        action_probs = F.softmax(action_logits, dim=-1)

        return {
            'action_logits': action_logits,
            'action_probs': action_probs,
            'graph_embedding': graph_emb,
            'node_embeddings': encoder_out['node_embeddings']
        }


class Critic(nn.Module):
    """
    MAINTAINED: Original Critic class for compatibility
    """

    def __init__(self,
                 node_dim: int = 7,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 global_features_dim: int = 4,  # CORRECTED: From 10 to 4
                 dropout: float = 0.2):
        super().__init__()

        self.encoder = GraphEncoder(node_dim, hidden_dim, num_layers, dropout)

        graph_emb_dim = hidden_dim * 2
        combined_dim = graph_emb_dim + global_features_dim

        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.value_head:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data: Data, global_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoder_out = self.encoder(data)
        graph_emb = encoder_out['graph_embedding']

        combined_features = torch.cat([graph_emb, global_features], dim=-1)
        state_value = self.value_head(combined_features)

        return {
            'state_value': state_value.squeeze(-1),
            'graph_embedding': graph_emb,
            'node_embeddings': encoder_out['node_embeddings']
        }


def create_actor_critic(config: Dict) -> ActorCritic:
    """
    CORRECTED: Factory function that FORCES shared encoder for PPO
    """
    return ActorCritic(
        node_dim=config.get('node_dim', 7),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 3),
        num_actions=config.get('num_actions', 7),
        global_features_dim=config.get('global_features_dim', 4),  # CORRECTED: Default to 4
        dropout=config.get('dropout', 0.2),
        shared_encoder=True  # CORRECTED: Always True for PPO
    )


# Testing code - MAINTAINED and UPDATED
if __name__ == "__main__":
    print("Testing corrected Actor-Critic models for PPO...")

    # Create dummy data
    num_nodes = 10
    node_features = torch.randn(num_nodes, 7)
    edge_index = torch.randint(0, num_nodes, (2, 20))
    global_features = torch.randn(1, 4)  # CORRECTED: 4 features instead of 10

    test_data = Data(x=node_features, edge_index=edge_index)

    # Test ActorCritic with corrected config
    config = {
        'node_dim': 7,
        'hidden_dim': 128,  # Increased for better performance
        'num_layers': 3,
        'num_actions': 7,
        'global_features_dim': 4,  # CORRECTED
        'dropout': 0.2,
        'shared_encoder': True  # CORRECTED
    }

    model = create_actor_critic(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Using shared encoder: {model.shared_encoder}")

    # Test forward pass
    output = model(test_data, global_features)
    print(f"Action logits shape: {output['action_logits'].shape}")
    print(f"Action probs shape: {output['action_probs'].shape}")
    print(f"Action probs sum: {output['action_probs'].sum().item():.6f}")
    print(f"State value shape: {output['state_value'].shape}")

    # Test action sampling
    action, log_prob, value = model.get_action_and_value(test_data, global_features)
    print(f"Sampled action: {action}, log_prob: {log_prob:.4f}, value: {value:.4f}")

    # Test PPO key method
    actions = torch.tensor([action])
    log_probs, values, entropy = model.evaluate_actions(test_data, global_features, actions)
    print(f"PPO evaluate_actions - log_probs: {log_probs.item():.4f}, "
          f"values: {values.item():.4f}, entropy: {entropy.item():.4f}")

    # Test with batch
    batch_data = Batch.from_data_list([test_data, test_data])
    batch_global_features = torch.cat([global_features, global_features], dim=0)
    batch_output = model(batch_data, batch_global_features)
    print(f"Batch test - Action probs shape: {batch_output['action_probs'].shape}")
    print(f"Batch test - State value shape: {batch_output['state_value'].shape}")

    print("All tests passed! Model is ready for PPO training.")