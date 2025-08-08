#!/usr/bin/env python3
"""
Actor-Critic GNN Models for Graph Refactoring
Implements GCN-based Actor and Critic networks for reinforcement learning.
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
    Shared GCN encoder for both Actor and Critic networks.
    Uses 3-layer GCN with skip connections and graph normalization.
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

        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        # GCN layers with skip connections
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(GraphNorm(hidden_dim))

        # Global pooling layers
        self.global_pool_mean = global_mean_pool
        self.global_pool_max = global_max_pool

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the graph encoder.

        Args:
            data: PyG Data object with node features and edges

        Returns:
            Dictionary containing node embeddings and graph embeddings
        """
        x = data.x
        edge_index = data.edge_index

        # Handle batch dimension
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        # GCN layers with skip connections
        for conv, norm in zip(self.convs, self.norms):
            # Store input for skip connection
            h = x

            # Apply convolution and normalization
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Skip connection
            x = x + h

        # Global pooling for graph-level representation
        graph_emb_mean = self.global_pool_mean(x, batch)
        graph_emb_max = self.global_pool_max(x, batch)

        # Concatenate mean and max pooling
        graph_emb = torch.cat([graph_emb_mean, graph_emb_max], dim=-1)

        return {
            'node_embeddings': x,
            'graph_embedding': graph_emb
        }


class Actor(nn.Module):
    """
    Actor network that outputs action probabilities.
    Uses shared graph encoder + action-specific MLP.
    """

    def __init__(self,
                 node_dim: int = 7,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_actions: int = 5,
                 global_features_dim: int = 10,
                 dropout: float = 0.2):
        super().__init__()

        self.num_actions = num_actions

        # Shared graph encoder
        self.encoder = GraphEncoder(node_dim, hidden_dim, num_layers, dropout)

        # Graph embedding dimension after concatenating mean and max pooling
        graph_emb_dim = hidden_dim * 2

        # Combined feature dimension (graph + global metrics)
        combined_dim = graph_emb_dim + global_features_dim

        # Action policy head
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
        """Initialize policy head weights"""
        for module in self.policy_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data: Data, global_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the actor network.

        Args:
            data: PyG Data object
            global_features: Global graph metrics [batch_size, global_features_dim]

        Returns:
            Dictionary with action logits and probabilities
        """
        # Get graph embeddings
        encoder_out = self.encoder(data)
        graph_emb = encoder_out['graph_embedding']

        # Combine graph embeddings with global features
        combined_features = torch.cat([graph_emb, global_features], dim=-1)

        # Compute action logits
        action_logits = self.policy_head(combined_features)
        action_probs = F.softmax(action_logits, dim=-1)

        return {
            'action_logits': action_logits,
            'action_probs': action_probs,
            'graph_embedding': graph_emb,
            'node_embeddings': encoder_out['node_embeddings']
        }

    def sample_action(self, data: Data, global_features: torch.Tensor) -> Tuple[int, float]:
        """
        Sample an action from the policy distribution.

        Returns:
            Tuple of (action, log_prob)
        """
        with torch.no_grad():
            output = self.forward(data, global_features)
            action_probs = output['action_probs']

            # Sample action
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            return action.item(), log_prob.item()

    def get_action_log_probs(self, data: Data, global_features: torch.Tensor,
                             actions: torch.Tensor) -> torch.Tensor:
        """
        Get log probabilities for given actions.

        Args:
            data: PyG Data object
            global_features: Global graph metrics
            actions: Actions taken [batch_size]

        Returns:
            Log probabilities of the actions [batch_size]
        """
        output = self.forward(data, global_features)
        action_probs = output['action_probs']

        # Create categorical distribution
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)

        return log_probs


class Critic(nn.Module):
    """
    Critic network that estimates state values V(s).
    Uses shared graph encoder + value estimation MLP.
    """

    def __init__(self,
                 node_dim: int = 7,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 global_features_dim: int = 10,
                 dropout: float = 0.2):
        super().__init__()

        # Shared graph encoder
        self.encoder = GraphEncoder(node_dim, hidden_dim, num_layers, dropout)

        # Graph embedding dimension after concatenating mean and max pooling
        graph_emb_dim = hidden_dim * 2

        # Combined feature dimension (graph + global metrics)
        combined_dim = graph_emb_dim + global_features_dim

        # Value estimation head
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
        """Initialize value head weights"""
        for module in self.value_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data: Data, global_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the critic network.

        Args:
            data: PyG Data object
            global_features: Global graph metrics [batch_size, global_features_dim]

        Returns:
            Dictionary with state value estimate
        """
        # Get graph embeddings
        encoder_out = self.encoder(data)
        graph_emb = encoder_out['graph_embedding']

        # Combine graph embeddings with global features
        combined_features = torch.cat([graph_emb, global_features], dim=-1)

        # Compute state value
        state_value = self.value_head(combined_features)

        return {
            'state_value': state_value.squeeze(-1),
            'graph_embedding': graph_emb,
            'node_embeddings': encoder_out['node_embeddings']
        }


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network with separate encoders.
    """

    def __init__(self,
                 node_dim: int = 7,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_actions: int = 7,  # Aggiornato per includere STOP
                 global_features_dim: int = 10,
                 dropout: float = 0.2,
                 shared_encoder: bool = False):  # Modificato default a False
        super().__init__()

        self.shared_encoder = shared_encoder
        self.num_actions = num_actions

        if shared_encoder:
            # Single shared encoder (mantieni logica esistente)
            self.encoder = GraphEncoder(node_dim, hidden_dim, num_layers, dropout)
            graph_emb_dim = hidden_dim * 2
            combined_dim = graph_emb_dim + global_features_dim

            self.actor_head = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 2, num_actions)
            )

            self.critic_head = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 2, 1)
            )
        else:
            # --- Modifica: encoder separati per Actor e Critic ---
            self.actor_encoder = GraphEncoder(node_dim, hidden_dim, num_layers, dropout)
            self.critic_encoder = GraphEncoder(node_dim, hidden_dim, num_layers, dropout)

            graph_emb_dim = hidden_dim * 2
            combined_dim = graph_emb_dim + global_features_dim

            # Actor head con supporto per azione STOP
            self.actor_head = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 2, num_actions)
            )

            # Critic head con possibile output esteso (valore + terminazione)
            self.critic_head = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 2, 2)  # valore + prob terminazione
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        if self.shared_encoder:
            for module in [self.actor_head, self.critic_head]:
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)

    def forward(self, data: Data, global_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through both actor and critic.
        """
        if self.shared_encoder:
            # Shared encoder path (mantieni logica esistente)
            encoder_out = self.encoder(data)
            graph_emb = encoder_out['graph_embedding']
            combined_features = torch.cat([graph_emb, global_features], dim=-1)

            action_logits = self.actor_head(combined_features)
            action_probs = F.softmax(action_logits, dim=-1)
            state_value = self.critic_head(combined_features).squeeze(-1)

            return {
                'action_logits': action_logits,
                'action_probs': action_probs,
                'state_value': state_value,
                'graph_embedding': graph_emb,
                'node_embeddings': encoder_out['node_embeddings']
            }
        else:
            # --- Modifica: encoder separati ---
            # Actor path
            actor_encoder_out = self.actor_encoder(data)
            actor_graph_emb = actor_encoder_out['graph_embedding']
            actor_combined = torch.cat([actor_graph_emb, global_features], dim=-1)

            action_logits = self.actor_head(actor_combined)
            action_probs = F.softmax(action_logits, dim=-1)

            # Critic path
            critic_encoder_out = self.critic_encoder(data)
            critic_graph_emb = critic_encoder_out['graph_embedding']
            critic_combined = torch.cat([critic_graph_emb, global_features], dim=-1)

            critic_output = self.critic_head(critic_combined)
            state_value = critic_output[:, 0]  # Primo output: valore
            termination_prob = torch.sigmoid(critic_output[:, 1])  # Secondo output: prob terminazione

            return {
                'action_logits': action_logits,
                'action_probs': action_probs,
                'state_value': state_value,
                'termination_prob': termination_prob,
                'actor_graph_embedding': actor_graph_emb,
                'critic_graph_embedding': critic_graph_emb,
                'actor_node_embeddings': actor_encoder_out['node_embeddings'],
                'critic_node_embeddings': critic_encoder_out['node_embeddings']
            }

    def get_action_and_value(self, data: Data, global_features: torch.Tensor) -> Tuple[int, float, float]:
        """
        Sample action and get state value for a single step.
        Supporta il campionamento dell'azione STOP (indice = num_actions - 1).
        """
        with torch.no_grad():
            output = self.forward(data, global_features)

            # Sample action con supporto per STOP
            action_dist = torch.distributions.Categorical(output['action_probs'])
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            return action.item(), log_prob.item(), output['state_value'].item()

    def evaluate_actions(self, data: Data, global_features: torch.Tensor,
                         actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.

        Returns:
            Tuple of (log_probs, state_values, entropy)
        """
        output = self.forward(data, global_features)

        # Action distribution
        action_dist = torch.distributions.Categorical(output['action_probs'])
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()

        return log_probs, output['state_value'], entropy

# Factory function for creating models
def create_actor_critic(config: Dict) -> ActorCritic:
    """
    Factory function to create ActorCritic model from config.
    """
    return ActorCritic(
        node_dim=config.get('node_dim', 7),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 3),
        num_actions=config.get('num_actions', 7),  # Aggiornato default
        global_features_dim=config.get('global_features_dim', 10),
        dropout=config.get('dropout', 0.2),
        shared_encoder=config.get('shared_encoder', False)  # Nuovo default
    )

# Testing code
if __name__ == "__main__":
    # Test the models
    print("Testing Actor-Critic models...")

    # Create dummy data
    num_nodes = 10
    node_features = torch.randn(num_nodes, 7)
    edge_index = torch.randint(0, num_nodes, (2, 20))
    global_features = torch.randn(1, 10)

    test_data = Data(x=node_features, edge_index=edge_index)

    # Test ActorCritic
    config = {
        'node_dim': 7,
        'hidden_dim': 64,
        'num_layers': 3,
        'num_actions': 5,
        'global_features_dim': 10,
        'dropout': 0.2,
        'shared_encoder': True
    }

    model = create_actor_critic(config)

    # Test forward pass
    output = model(test_data, global_features)
    print(f"Action logits shape: {output['action_logits'].shape}")
    print(f"State value shape: {output['state_value'].shape}")

    # Test action sampling
    action, log_prob, value = model.get_action_and_value(test_data, global_features)
    print(f"Sampled action: {action}, log_prob: {log_prob:.4f}, value: {value:.4f}")

    print("✅ All tests passed!")