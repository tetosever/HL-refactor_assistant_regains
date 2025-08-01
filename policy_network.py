from __future__ import annotations

from typing import NamedTuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_add_pool, GraphNorm
from torch_geometric.utils import softmax


class EnhancedGCPNOutput(NamedTuple):
    pi1: torch.Tensor  # Node selection (source)
    pi2: torch.Tensor  # Node selection (target/affected)
    pi3: torch.Tensor  # Refactoring pattern
    pi4: torch.Tensor  # Termination decision
    logits1: torch.Tensor
    logits2: torch.Tensor
    logits3: torch.Tensor
    logits4: torch.Tensor
    hub_scores: torch.Tensor  # Hub probability per node
    pattern_applicability: torch.Tensor  # Pattern scores per node
    attention_weights: torch.Tensor  # Attention weights per debugging


class HubAwareAttention(nn.Module):
    """
    Attention layer che si concentra sui nodi hub usando le features esistenti
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_proj = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),  # +1 per hub_score
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, node_embeddings: torch.Tensor, hub_scores: torch.Tensor) -> torch.Tensor:
        # Concatena embeddings con hub scores
        attended_input = torch.cat([node_embeddings, hub_scores.unsqueeze(-1)], dim=-1)
        attention_weights = self.attention_proj(attended_input)

        # Applica attention con bias verso hub nodes
        attended_embeddings = node_embeddings * (1.0 + attention_weights)

        return attended_embeddings, attention_weights.squeeze(-1)


class ActionSequenceMemory(nn.Module):
    """
    Modulo per tracciare e apprendere da sequenze di azioni precedenti
    """

    def __init__(self, hidden_dim: int, max_sequence_length: int = 10):
        super().__init__()
        self.max_length = max_sequence_length
        self.hidden_dim = hidden_dim

        # 6 pattern types + terminate
        self.action_embedding = nn.Embedding(7, hidden_dim // 4)
        self.sequence_lstm = nn.LSTM(
            hidden_dim // 4,
            hidden_dim // 2,
            batch_first=True,
            num_layers=1
        )
        self.context_proj = nn.Linear(hidden_dim // 2, hidden_dim)

    def forward(self, action_history: List[int]) -> torch.Tensor:
        if len(action_history) == 0:
            return torch.zeros(1, self.hidden_dim, device=next(self.parameters()).device)

        # Prendi solo le ultime azioni
        recent_actions = action_history[-self.max_length:]
        actions_tensor = torch.tensor(recent_actions, device=next(self.parameters()).device).unsqueeze(0)

        # Embedding delle azioni
        action_embs = self.action_embedding(actions_tensor)

        # LSTM per sequence encoding
        _, (hidden, _) = self.sequence_lstm(action_embs)

        # Proietta a dimensione corretta
        sequence_context = self.context_proj(hidden.squeeze(0))

        return sequence_context


class HierarchicalPooling(nn.Module):
    """
    Pooling gerarchico che da più peso ai nodi hub
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hub_importance_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, hub_scores: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Calcola importance weights
        importance_weights = self.hub_importance_net(x).squeeze(-1)

        # Combina con hub scores (bias verso nodi hub)
        combined_weights = importance_weights * (1.0 + 2.0 * hub_scores)

        # Weighted pooling
        weighted_x = x * combined_weights.unsqueeze(-1)
        graph_emb = global_add_pool(weighted_x, batch)

        return graph_emb


class EnhancedGCPNPolicyNetwork(nn.Module):
    """
    Enhanced Policy Network per hub-like dependency refactoring con:
    - Hub-aware attention
    - Pattern-based actions
    - Sequence memory
    - Hierarchical pooling
    """

    # Mapping delle azioni ai pattern di refactoring
    ACTION_PATTERNS = {
        0: "EXTRACT_INTERFACE",  # Per hub con alto fan-out
        1: "SPLIT_COMPONENT",  # Per componenti troppo grandi
        2: "INTRODUCE_MEDIATOR",  # Per hub con alta interconnessione
        3: "INTRODUCE_FACADE",  # Per semplificare interfacce complesse
        4: "APPLY_DEPENDENCY_INJECTION",  # Per ridurre accoppiamento
        5: "REMOVE_UNNECESSARY_DEPENDENCY",  # Cleanup semplice
    }

    def __init__(
            self,
            node_in_dim: int,
            edge_in_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 3,
            dropout: float = 0.3,
            device: torch.device = torch.device('cpu'),
            enable_memory: bool = True,
            max_sequence_length: int = 10
    ):
        super().__init__()
        self.device = device
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.enable_memory = enable_memory

        # Embedding layers
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_in_dim, hidden_dim)

        # GNN backbone con residual connections
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim, train_eps=True))
            self.norms.append(GraphNorm(hidden_dim))

        # Enhanced modules
        self.hub_attention = HubAwareAttention(hidden_dim)
        self.hierarchical_pool = HierarchicalPooling(hidden_dim)

        if enable_memory:
            self.action_memory = ActionSequenceMemory(hidden_dim, max_sequence_length)

        # Policy heads
        mid_dim = hidden_dim // 2

        # Node selection heads (con hub bias)
        self.node1_head = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1)
        )

        self.node2_head = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1)
        )

        # Pattern action head (6 patterns)
        self.pattern_head = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, len(self.ACTION_PATTERNS))
        )

        # Termination head
        self.terminate_head = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 2)
        )

        # Weight initialization
        self._init_weights()
        self.to(self.device)

    def _init_weights(self):
        """Inizializzazione dei pesi per stabilità del training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compute_hub_indicators(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Calcola hub scores usando le features esistenti:
        - FanIn, FanOut, PageRank, LinesOfCode, NumberOfChildren,
          InstabilityMetric, AbstractnessMetric, TotalAmountOfChanges
        """
        # Estrai features (assumendo ordine dal config)
        fan_in = node_features[:, 0]  # FanIn
        fan_out = node_features[:, 1]  # FanOut
        pagerank = node_features[:, 2]  # PageRank
        loc = node_features[:, 3]  # LinesOfCode
        n_children = node_features[:, 4]  # NumberOfChildren
        instability = node_features[:, 5]  # InstabilityMetric
        abstractness = node_features[:, 6]  # AbstractnessMetric
        changes = node_features[:, 7]  # TotalAmountOfChanges

        # Normalizza le features
        eps = 1e-8
        fan_in_norm = fan_in / (fan_in.max() + eps)
        fan_out_norm = fan_out / (fan_out.max() + eps)
        pagerank_norm = pagerank / (pagerank.max() + eps)
        loc_norm = loc / (loc.max() + eps)
        children_norm = n_children / (n_children.max() + eps)
        changes_norm = changes / (changes.max() + eps)

        # Calcola hub score composito
        # Peso maggiore a fan-in/out e pagerank (indicatori primari di hub)
        hub_score = (
                0.25 * fan_in_norm +  # Molte dipendenze in entrata
                0.25 * fan_out_norm +  # Molte dipendenze in uscita
                0.20 * pagerank_norm +  # Alta centralità
                0.15 * loc_norm +  # Componente grande
                0.10 * children_norm +  # Molti sotto-componenti
                0.05 * instability  # Instabilità (0-1 già)
            # Abstractness può essere bassa (implementazione) o alta (interfaccia)
            # Changes indica attività/problematicità
        )

        return torch.clamp(hub_score, 0.0, 1.0)

    def compute_pattern_applicability(self, node_features: torch.Tensor, hub_scores: torch.Tensor) -> torch.Tensor:
        """
        Calcola applicabilità dei pattern di refactoring per ogni nodo
        """
        fan_in = node_features[:, 0]
        fan_out = node_features[:, 1]
        pagerank = node_features[:, 2]
        loc = node_features[:, 3]
        instability = node_features[:, 5]
        abstractness = node_features[:, 6]

        # Soglie dinamiche basate sulla distribuzione
        fan_out_threshold = fan_out.median() + fan_out.std()
        loc_threshold = loc.quantile(0.75)
        fan_in_threshold = fan_in.median() + fan_in.std()

        # Pattern applicability scores
        extract_interface = (
                (fan_out > fan_out_threshold) &
                (instability > 0.7) &
                (hub_scores > 0.6)
        ).float()

        split_component = (
                (loc > loc_threshold) &
                (hub_scores > 0.5)
        ).float()

        introduce_mediator = (
                (fan_in > fan_in_threshold) &
                (fan_out > fan_out_threshold) &
                (hub_scores > 0.7)
        ).float()

        introduce_facade = (
                (fan_out > fan_out_threshold) &
                (abstractness < 0.3) &  # Implementazione concreta
                (hub_scores > 0.6)
        ).float()

        apply_di = (
                (fan_out > fan_out.median()) &
                (instability > 0.5) &
                (hub_scores > 0.4)
        ).float()

        remove_dependency = (
                (fan_out > 0) |  # Ha almeno una dipendenza
                (fan_in > 0)
        ).float()

        return torch.stack([
            extract_interface, split_component, introduce_mediator,
            introduce_facade, apply_di, remove_dependency
        ], dim=-1)

    def create_action_mask(self, node_features: torch.Tensor, hub_scores: torch.Tensor,
                           selected_node_idx: int) -> torch.Tensor:
        """
        Crea mask per azioni valide basate sulle features del nodo selezionato
        """
        if selected_node_idx >= len(node_features):
            # Se indice non valido, permetti solo terminate
            return torch.tensor([False] * len(self.ACTION_PATTERNS), dtype=torch.bool, device=self.device)

        node_feats = node_features[selected_node_idx]
        hub_score = hub_scores[selected_node_idx]

        # Regole di validità basate sulle features
        can_extract_interface = node_feats[1] > 2 and hub_score > 0.5  # FanOut > 2 + hub
        can_split = node_feats[3] > node_features[:, 3].median()  # LOC above median
        can_mediator = node_feats[0] > 1 and node_feats[1] > 1  # Fan-in > 1 AND Fan-out > 1
        can_facade = node_feats[1] > 2 and node_feats[6] < 0.5  # FanOut > 2 + low abstractness
        can_di = node_feats[1] > 1 and node_feats[5] > 0.3  # FanOut > 1 + some instability
        can_remove = node_feats[0] > 0 or node_feats[1] > 0  # Ha almeno una dipendenza

        mask = torch.tensor([
            can_extract_interface,
            can_split,
            can_mediator,
            can_facade,
            can_di,
            can_remove
        ], dtype=torch.bool, device=self.device)

        return mask

    def forward(self, data: Data, action_history: Optional[List[int]] = None) -> EnhancedGCPNOutput:
        """
        Forward pass con enhanced features
        """
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_attr = data.edge_attr.to(self.device)
        batch = data.batch.to(self.device) if data.batch is not None else torch.zeros(
            x.size(0), dtype=torch.long, device=self.device
        )

        # Calcola hub indicators dalle features esistenti
        hub_scores = self.compute_hub_indicators(x)

        # Pattern applicability
        pattern_applicability = self.compute_pattern_applicability(x, hub_scores)

        # Embedding iniziale
        x_emb = self.node_embed(x)
        edge_attr_emb = self.edge_embed(edge_attr)

        # GNN layers con hub-aware attention
        attention_weights_list = []
        for conv, norm in zip(self.convs, self.norms):
            out = conv(x_emb, edge_index, edge_attr_emb)
            out = norm(out)
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)

            # Applica hub-aware attention
            out, att_weights = self.hub_attention(out, hub_scores)
            attention_weights_list.append(att_weights)

            # Skip connection
            x_emb = x_emb + out

        # Usa l'ultima layer di attention per il debugging
        final_attention_weights = attention_weights_list[-1]

        # Hierarchical pooling per graph-level embedding
        graph_emb = self.hierarchical_pool(x_emb, hub_scores, batch)

        # Sequence memory (se abilitata)
        if self.enable_memory and action_history:
            sequence_context = self.action_memory(action_history)
            graph_emb = graph_emb + sequence_context

        # Policy heads

        # π1: Node selection (source) - bias verso hub nodes
        logits1 = self.node1_head(x_emb).squeeze(-1)
        # Aggiungi bias per hub nodes
        hub_bias = hub_scores * 2.0  # Aumenta probabilità di selezione per hub
        logits1_biased = logits1 + hub_bias
        pi1 = softmax(logits1_biased, batch)

        # π2: Node selection (target/affected)
        logits2 = self.node2_head(x_emb).squeeze(-1)
        pi2 = softmax(logits2, batch)

        # π3: Pattern selection con applicability weighting
        logits3 = self.pattern_head(graph_emb)

        # Weight logits based on pattern applicability
        # Prendi la media dell'applicability per tutti i nodi nel grafo
        avg_pattern_applicability = global_add_pool(pattern_applicability, batch)
        pattern_weighted_logits = logits3 + avg_pattern_applicability * 2.0

        pi3 = F.softmax(pattern_weighted_logits, dim=-1)

        # π4: Termination decision
        logits4 = self.terminate_head(graph_emb)
        pi4 = F.softmax(logits4, dim=-1)

        return EnhancedGCPNOutput(
            pi1=pi1, pi2=pi2, pi3=pi3, pi4=pi4,
            logits1=logits1_biased, logits2=logits2,
            logits3=pattern_weighted_logits, logits4=logits4,
            hub_scores=hub_scores,
            pattern_applicability=pattern_applicability,
            attention_weights=final_attention_weights
        )

    def get_action_name(self, action_idx: int) -> str:
        """Ottieni nome leggibile dell'azione"""
        return self.ACTION_PATTERNS.get(action_idx, f"UNKNOWN_{action_idx}")

    def compute_action_mask_for_graph(self, data: Data) -> torch.Tensor:
        """
        Calcola action mask per l'intero grafo (per debugging/analysis)
        """
        hub_scores = self.compute_hub_indicators(data.x)

        # Calcola mask per ogni nodo
        masks = []
        for i in range(len(data.x)):
            node_mask = self.create_action_mask(data.x, hub_scores, i)
            masks.append(node_mask)

        return torch.stack(masks, dim=0)  # [num_nodes, num_actions]