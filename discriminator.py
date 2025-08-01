from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, GraphNorm


class HubAwareLoss(nn.Module):
    """
    Loss function specializzata per hub-like dependencies che considera:
    - Importance weighting basato su metriche topologiche
    - Focal loss per hard examples
    - Penalità per falsi negativi (più critico non rilevare un hub problematico)
    """

    def __init__(self,
                 alpha: float = 0.25,  # weight per classe positiva
                 gamma: float = 2.0,  # focusing parameter
                 hub_weight: float = 2.0):  # extra weight per nodi hub
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.hub_weight = hub_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                hub_importance: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, 2] - output del discriminator
            targets: [batch_size] - labels (0/1)
            hub_importance: [batch_size] - peso basato su metriche topologiche
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Focal loss component
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # Hub importance weighting
        if hub_importance is not None:
            focal_loss = focal_loss * (1 + self.hub_weight * hub_importance)

        return focal_loss.mean()


class AttentionPooling(nn.Module):
    """
    Attention-based pooling che si concentra sui nodi più importanti
    invece del semplice mean pooling
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Compute attention weights
        attention_weights = self.attention(x)  # [num_nodes, 1]
        attention_weights = F.softmax(attention_weights, dim=0)

        # Apply attention pooling per batch
        batch_size = batch.max().item() + 1
        pooled = []

        for i in range(batch_size):
            mask = (batch == i)
            if mask.any():
                node_features = x[mask]  # [nodes_in_graph_i, hidden_dim]
                node_weights = attention_weights[mask]  # [nodes_in_graph_i, 1]
                weighted_sum = (node_features * node_weights).sum(dim=0)
                pooled.append(weighted_sum)
            else:
                pooled.append(torch.zeros(x.size(1), device=x.device))

        return torch.stack(pooled)


class MultiScaleGCNDiscriminator(nn.Module):
    """
    Discriminator avanzato per hub-like dependencies con:
    - Multi-scale feature extraction (diversi hop neighborhoods)
    - Attention mechanism per identificare nodi critici
    - Hub-specific features integration
    - Architectural pattern recognition
    """

    def __init__(self,
                 node_in_dim: int,
                 edge_in_dim: int = 0,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 use_attention: bool = True,
                 use_gat: bool = True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_attention = use_attention
        self.use_gat = use_gat

        # Input projection
        self.node_projection = nn.Linear(node_in_dim, hidden_dim)
        if edge_in_dim > 0:
            self.edge_projection = nn.Linear(edge_in_dim, hidden_dim)

        # Multi-scale GCN layers
        self.gcn_layers = nn.ModuleList()
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            # Standard GCN per struttura locale
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

            # GAT per catturare relazioni importanti
            if use_gat:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
                )

            self.norms.append(GraphNorm(hidden_dim))

        # Hub-specific feature extractor
        self.hub_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4)  # [is_hub, coupling, cohesion, centrality]
        )

        # Pooling strategy
        if use_attention:
            self.pooling = AttentionPooling(hidden_dim)
        else:
            self.pooling = None

        # Pattern recognition module
        self.pattern_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # concat mean + max pool
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 8)  # common architectural patterns
        )

        # Final classifier
        classifier_input_dim = hidden_dim + 4 + 8  # graph_emb + hub_features + patterns
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier/He initialization for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if 'classifier' in str(m):
                    nn.init.xavier_normal_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compute_hub_importance(self, x: torch.Tensor, edge_index: torch.Tensor,
                               batch: torch.Tensor) -> torch.Tensor:
        """
        Calcola l'importanza dei nodi basata su metriche topologiche
        per essere usata nella loss function
        """
        batch_size = batch.max().item() + 1
        importance_scores = []

        for i in range(batch_size):
            mask = (batch == i)
            if not mask.any():
                importance_scores.append(0.0)
                continue

            # Estrai sottografo
            node_indices = torch.where(mask)[0]

            # Calcola degree centrality (semplificato)
            degrees = torch.zeros(mask.sum(), device=x.device)
            for j, node_idx in enumerate(node_indices):
                in_degree = (edge_index[1] == node_idx).sum().float()
                out_degree = (edge_index[0] == node_idx).sum().float()
                degrees[j] = in_degree + out_degree

            # Normalizza e calcola importanza
            if degrees.max() > 0:
                normalized_degrees = degrees / degrees.max()
                # Nodi con alto grado sono più importanti
                graph_importance = normalized_degrees.max().item()
            else:
                graph_importance = 0.0

            importance_scores.append(graph_importance)

        return torch.tensor(importance_scores, device=x.device)

    def extract_graph_patterns(self, x: torch.Tensor, edge_index: torch.Tensor,
                               batch: torch.Tensor) -> torch.Tensor:
        """
        Estrae pattern architetturali comuni che indicano problemi di design
        """
        batch_size = batch.max().item() + 1
        patterns = []

        for i in range(batch_size):
            mask = (batch == i)
            if not mask.any():
                patterns.append(torch.zeros(8, device=x.device))
                continue

            # Calcola metriche strutturali
            node_count = mask.sum().float()

            # Estrai edges per questo grafo
            edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
            edge_count = edge_mask.sum().float()

            # Pattern features
            density = edge_count / (node_count * (node_count - 1) + 1e-6)

            # Degree distribution analysis
            degrees = torch.zeros(mask.sum(), device=x.device)
            node_indices = torch.where(mask)[0]
            for j, node_idx in enumerate(node_indices):
                degree = ((edge_index[0] == node_idx) | (edge_index[1] == node_idx)).sum()
                degrees[j] = degree.float()

            if len(degrees) > 0:
                max_degree = degrees.max()
                mean_degree = degrees.mean()
                std_degree = degrees.std() if len(degrees) > 1 else torch.tensor(0.0, device=x.device)
                degree_centralization = (max_degree - mean_degree) / (node_count - 1 + 1e-6)
            else:
                max_degree = mean_degree = std_degree = degree_centralization = torch.tensor(0.0, device=x.device)

            pattern_vector = torch.tensor([
                density,
                max_degree / (node_count + 1e-6),  # normalized max degree
                mean_degree,
                std_degree,
                degree_centralization,
                (max_degree > 0.7 * node_count).float(),  # star pattern detection
                (density > 0.8).float(),  # clique pattern detection
                (std_degree > mean_degree).float()  # irregular pattern detection
            ], device=x.device)

            patterns.append(pattern_vector)

        return torch.stack(patterns)

    def forward(self, data) -> Dict[str, torch.Tensor]:
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        batch = getattr(data, 'batch', None)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Input projection
        x = self.node_projection(x)

        # Multi-scale feature extraction
        layer_outputs = []
        for i, (gcn, norm) in enumerate(zip(self.gcn_layers, self.norms)):
            # GCN pass
            h_gcn = gcn(x, edge_index)

            # GAT pass (if enabled)
            if self.use_gat and i < len(self.gat_layers):
                h_gat = self.gat_layers[i](x, edge_index)
                h = h_gcn + h_gat  # Combine GCN and GAT
            else:
                h = h_gcn

            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # Skip connection
            if i > 0:
                x = x + h
            else:
                x = h

            layer_outputs.append(x)

        # Hub importance per loss weighting
        hub_importance = self.compute_hub_importance(x, edge_index, batch)

        # Hub-specific features
        hub_features = self.hub_analyzer(x)
        hub_features_pooled = global_mean_pool(hub_features, batch)

        # Graph-level embedding
        if self.use_attention:
            graph_emb = self.pooling(x, batch)
        else:
            graph_emb = global_mean_pool(x, batch)

        # Pattern detection
        pattern_emb = torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch)
        ], dim=1)
        pattern_features = self.pattern_detector(pattern_emb)

        # Combine all features
        combined_features = torch.cat([
            graph_emb,
            hub_features_pooled,
            pattern_features
        ], dim=1)

        # Final classification
        logits = self.classifier(combined_features)

        return {
            'logits': logits,
            'hub_importance': hub_importance,
            'graph_embedding': graph_emb,
            'hub_features': hub_features_pooled,
            'pattern_features': pattern_features
        }


class AdvancedTrainer:
    """
    Trainer con features avanzate per il discriminator migliorato
    """

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.hub_loss = HubAwareLoss(alpha=0.25, gamma=2.0, hub_weight=2.0)

    def compute_loss(self, outputs, targets):
        """Compute the specialized hub-aware loss"""
        return self.hub_loss(
            outputs['logits'],
            targets,
            outputs['hub_importance']
        )

    def evaluate_detailed(self, loader):
        """
        Valutazione dettagliata con metriche specifiche per hub detection
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_hub_scores = []

        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                outputs = self.model(data)

                probs = F.softmax(outputs['logits'], dim=1)
                predictions = probs.argmax(dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(data.is_smelly.cpu().numpy())
                all_hub_scores.extend(outputs['hub_importance'].cpu().numpy())

        return {
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets),
            'hub_scores': np.array(all_hub_scores)
        }


# Esempio di utilizzo migliorato
def create_improved_discriminator(node_dim: int, edge_dim: int = 0) -> MultiScaleGCNDiscriminator:
    """Factory function per creare il discriminator migliorato"""
    return MultiScaleGCNDiscriminator(
        node_in_dim=node_dim,
        edge_in_dim=edge_dim,
        hidden_dim=128,  # Aumentato per catturare più pattern
        num_layers=4,  # Più layer per pattern complessi
        dropout=0.3,
        use_attention=True,
        use_gat=True
    )