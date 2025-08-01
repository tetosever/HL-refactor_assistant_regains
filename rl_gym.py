import copy
from typing import Dict, Tuple, Any, Optional

import networkx as nx
import torch
import torch.nn.functional as F
from gymnasium import Env
from gymnasium.spaces import MultiDiscrete
from torch_geometric.data import Data


class EnhancedDependencyRefactorEnv(Env):
    """
    Enhanced RL Environment per hub-like dependency refactoring con:
    - Pattern-based actions invece di azioni atomiche
    - Reward shaping sofisticato
    - Validazione semantica delle azioni
    - Tracking dettagliato delle metriche
    """

    # Pattern di refactoring supportati
    REFACTORING_PATTERNS = {
        0: "EXTRACT_INTERFACE",
        1: "SPLIT_COMPONENT",
        2: "INTRODUCE_MEDIATOR",
        3: "INTRODUCE_FACADE",
        4: "APPLY_DEPENDENCY_INJECTION",
        5: "REMOVE_UNNECESSARY_DEPENDENCY"
    }

    def __init__(
            self,
            initial_data: Data,
            discriminator: torch.nn.Module,
            max_steps: int = 50,
            device: torch.device = torch.device('cpu'),

            # Reward parameters
            pos_terminal: float = 3.0,
            neg_terminal: float = -1.5,
            step_penalty: float = -0.02,

            # Pattern-specific rewards
            successful_pattern_bonus: float = 1.5,
            failed_pattern_penalty: float = -0.5,
            pattern_appropriateness_weight: float = 0.3,

            # Hub-specific rewards
            hub_reduction_bonus: float = 2.0,
            centrality_improvement_weight: float = 0.4,
            coupling_reduction_weight: float = 0.3,

            # Constraint violations
            invalid_action_penalty: float = -0.3,
            architectural_violation_penalty: float = -0.8,

            # Environment limits
            max_nodes_multiplier: float = 1.8,
            min_meaningful_change_threshold: float = 0.05,

            # Advanced features
            enable_progress_tracking: bool = True,
            enable_pattern_memory: bool = True,
            diversity_bonus_weight: float = 0.1
    ):
        super().__init__()

        self.device = device
        self.discriminator = discriminator
        self.max_steps = max_steps

        # Environment state
        self.initial_data = initial_data
        self.current_data = None
        self.steps = 0

        # Graph constraints
        self.initial_n_nodes = initial_data.x.size(0)
        self.max_allowed_nodes = int(self.initial_n_nodes * max_nodes_multiplier)

        # Action space: [source_node, target_node, pattern_type, terminate]
        self.action_space = MultiDiscrete([
            self.initial_n_nodes,  # Source node
            self.initial_n_nodes,  # Target node
            len(self.REFACTORING_PATTERNS),  # Pattern type
            2  # Terminate (0/1)
        ])

        # Reward parameters
        self.REW_POS_TERM = pos_terminal
        self.REW_NEG_TERM = neg_terminal
        self.REW_STEP_PENALTY = step_penalty
        self.REW_PATTERN_SUCCESS = successful_pattern_bonus
        self.REW_PATTERN_FAIL = failed_pattern_penalty
        self.REW_PATTERN_APPROPRIATENESS = pattern_appropriateness_weight
        self.REW_HUB_REDUCTION = hub_reduction_bonus
        self.REW_CENTRALITY_WEIGHT = centrality_improvement_weight
        self.REW_COUPLING_WEIGHT = coupling_reduction_weight
        self.REW_INVALID_ACTION = invalid_action_penalty
        self.REW_ARCH_VIOLATION = architectural_violation_penalty
        self.MIN_CHANGE_THRESHOLD = min_meaningful_change_threshold
        self.DIVERSITY_WEIGHT = diversity_bonus_weight

        # State tracking
        self.enable_progress_tracking = enable_progress_tracking
        self.enable_pattern_memory = enable_pattern_memory

        if enable_progress_tracking:
            self.initial_metrics = self._compute_comprehensive_metrics(initial_data)
            self.best_metrics = copy.deepcopy(self.initial_metrics)
            self.steps_without_improvement = 0

        if enable_pattern_memory:
            self.applied_patterns = []
            self.pattern_success_history = []

        # Action tracking
        self.action_history = []
        self.invalid_action_count = 0
        self.valid_action_count = 0

        # Statistics
        self.episode_stats = {
            'patterns_applied': 0,
            'successful_patterns': 0,
            'hub_nodes_addressed': set(),
            'total_node_changes': 0,
            'total_edge_changes': 0
        }

    def reset(self) -> Data:
        """Reset environment to initial state"""
        # Deep copy initial data
        d0 = self.initial_data
        self.current_data = Data(
            x=d0.x.clone(),
            edge_index=d0.edge_index.clone(),
            edge_attr=d0.edge_attr.clone(),
        ).to(self.device)
        self.current_data.num_nodes = self.current_data.x.size(0)

        # Reset counters
        self.steps = 0
        self.invalid_action_count = 0
        self.valid_action_count = 0

        # Reset tracking
        if self.enable_progress_tracking:
            self.initial_metrics = self._compute_comprehensive_metrics(self.current_data)
            self.best_metrics = copy.deepcopy(self.initial_metrics)
            self.steps_without_improvement = 0

        if self.enable_pattern_memory:
            self.applied_patterns.clear()
            self.pattern_success_history.clear()

        self.action_history.clear()

        # Reset statistics
        self.episode_stats = {
            'patterns_applied': 0,
            'successful_patterns': 0,
            'hub_nodes_addressed': set(),
            'total_node_changes': 0,
            'total_edge_changes': 0
        }

        # Update action space if needed
        current_n_nodes = self.current_data.x.size(0)
        self.action_space = MultiDiscrete([
            current_n_nodes, current_n_nodes,
            len(self.REFACTORING_PATTERNS), 2
        ])

        return self.current_data

    def step(self, action: Tuple[int, int, int, int]) -> Tuple[Data, float, bool, Dict]:
        """Execute action and return new state, reward, done, info"""
        source_node, target_node, pattern_type, terminate = action
        self.steps += 1

        old_data = copy.deepcopy(self.current_data)
        old_metrics = self._compute_comprehensive_metrics(old_data)

        reward = 0.0
        done = False
        info = {
            'valid_action': False,
            'pattern_applied': None,
            'pattern_success': False,
            'hub_improvement': 0.0,
            'early_termination_reason': None
        }

        # Handle termination
        if terminate == 1:
            done = True
            reward += self._compute_termination_reward(old_metrics)
            info['termination_type'] = 'requested'
            self._update_episode_stats(info)
            return self.current_data, reward, done, info

        # Validate action parameters
        current_n_nodes = self.current_data.x.size(0)
        if (source_node >= current_n_nodes or target_node >= current_n_nodes or
                pattern_type >= len(self.REFACTORING_PATTERNS)):
            reward += self.REW_INVALID_ACTION
            self.invalid_action_count += 1
            info['early_termination_reason'] = 'invalid_parameters'
        else:
            # Apply refactoring pattern
            pattern_name = self.REFACTORING_PATTERNS[pattern_type]
            success, new_data, pattern_info = self._apply_refactoring_pattern(
                pattern_name, source_node, target_node, old_data
            )

            if success:
                self.current_data = new_data
                self.valid_action_count += 1
                info['valid_action'] = True
                info['pattern_applied'] = pattern_name
                info['pattern_success'] = True

                # Compute pattern-specific reward
                pattern_reward = self._compute_pattern_reward(
                    old_data, new_data, pattern_name, pattern_info
                )
                reward += pattern_reward

                # Track applied pattern
                if self.enable_pattern_memory:
                    self.applied_patterns.append(pattern_type)
                    self.pattern_success_history.append(True)

                self.episode_stats['patterns_applied'] += 1
                self.episode_stats['successful_patterns'] += 1
                self.episode_stats['hub_nodes_addressed'].add(source_node)

            else:
                # Pattern application failed
                reward += self.REW_PATTERN_FAIL
                self.invalid_action_count += 1
                info['pattern_applied'] = pattern_name
                info['pattern_success'] = False
                info['failure_reason'] = pattern_info.get('failure_reason', 'unknown')

                if self.enable_pattern_memory:
                    self.pattern_success_history.append(False)

        # Compute progress-based rewards
        if info['valid_action']:
            new_metrics = self._compute_comprehensive_metrics(self.current_data)
            progress_reward = self._compute_progress_reward(old_metrics, new_metrics)
            reward += progress_reward
            info['hub_improvement'] = old_metrics['hub_score'] - new_metrics['hub_score']

            # Update best metrics
            if self.enable_progress_tracking:
                if new_metrics['hub_score'] < self.best_metrics['hub_score']:
                    self.best_metrics = copy.deepcopy(new_metrics)
                    self.steps_without_improvement = 0
                    reward += 0.5  # Bonus for new best
                else:
                    self.steps_without_improvement += 1

        # Diversity bonus (encourage trying different patterns)
        if self.enable_pattern_memory and len(self.applied_patterns) > 1:
            recent_patterns = self.applied_patterns[-5:]  # Last 5 patterns
            pattern_diversity = len(set(recent_patterns)) / len(recent_patterns)
            reward += self.DIVERSITY_WEIGHT * pattern_diversity

        # Step penalty
        reward += self.REW_STEP_PENALTY

        # Track action
        self.action_history.append((source_node, target_node, pattern_type))

        # Check termination conditions
        if not done:
            done, termination_reason = self._check_early_termination()
            if done:
                info['early_termination_reason'] = termination_reason
                if termination_reason == 'no_progress':
                    reward += self.REW_NEG_TERM * 0.5
                elif termination_reason == 'too_many_invalid':
                    reward += self.REW_NEG_TERM * 0.7
                elif termination_reason == 'max_steps':
                    reward += self.REW_NEG_TERM * 0.3

        # Update episode statistics
        self._update_episode_stats(info)

        return self.current_data, reward, done, info

    def _apply_refactoring_pattern(self, pattern_name: str, source_node: int,
                                   target_node: int, data: Data) -> Tuple[bool, Data, Dict]:
        """Apply specific refactoring pattern"""
        try:
            if pattern_name == "EXTRACT_INTERFACE":
                return self._extract_interface(source_node, target_node, data)
            elif pattern_name == "SPLIT_COMPONENT":
                return self._split_component(source_node, data)
            elif pattern_name == "INTRODUCE_MEDIATOR":
                return self._introduce_mediator(source_node, target_node, data)
            elif pattern_name == "INTRODUCE_FACADE":
                return self._introduce_facade(source_node, data)
            elif pattern_name == "APPLY_DEPENDENCY_INJECTION":
                return self._apply_dependency_injection(source_node, target_node, data)
            elif pattern_name == "REMOVE_UNNECESSARY_DEPENDENCY":
                return self._remove_unnecessary_dependency(source_node, target_node, data)
            else:
                return False, data, {'failure_reason': 'unknown_pattern'}
        except Exception as e:
            return False, data, {'failure_reason': f'exception: {str(e)}'}

    def _extract_interface(self, hub_node: int, client_node: int, data: Data) -> Tuple[bool, Data, Dict]:
        """
        Extract Interface Pattern: Crea interfaccia tra hub e client
        Hub -> Interface -> Client
        """
        # Validazione prerequisiti
        hub_features = data.x[hub_node]
        fan_out = hub_features[1].item()  # FanOut

        if fan_out < 2:
            return False, data, {'failure_reason': 'insufficient_fan_out'}

        # Verifica che esista connessione hub -> client
        edge_exists = ((data.edge_index[0] == hub_node) &
                       (data.edge_index[1] == client_node)).any().item()

        if not edge_exists:
            return False, data, {'failure_reason': 'no_direct_connection'}

        # Crea nuovo nodo interfaccia
        new_data = copy.deepcopy(data)
        interface_node = new_data.x.size(0)

        # Features per nodo interfaccia (media tra hub e client)
        hub_feats = new_data.x[hub_node]
        client_feats = new_data.x[client_node]
        interface_feats = (hub_feats + client_feats) / 2.0

        # L'interfaccia dovrebbe avere alta abstractness, bassa instability
        interface_feats[5] = 0.2  # InstabilityMetric bassa
        interface_feats[6] = 0.8  # AbstractnessMetric alta
        interface_feats[3] = hub_feats[3] * 0.1  # LOC molto ridotte

        # Aggiungi nodo interfaccia
        new_data.x = torch.cat([new_data.x, interface_feats.unsqueeze(0)], dim=0)
        new_data.num_nodes += 1

        # Ridireziona connessione: hub -> interface -> client
        edge_mask = ~((new_data.edge_index[0] == hub_node) &
                      (new_data.edge_index[1] == client_node))

        # Rimuovi edge diretto hub -> client
        new_edge_index = new_data.edge_index[:, edge_mask]
        new_edge_attr = new_data.edge_attr[edge_mask]

        # Aggiungi hub -> interface
        hub_to_interface = torch.tensor([[hub_node], [interface_node]], device=self.device)
        # Aggiungi interface -> client
        interface_to_client = torch.tensor([[interface_node], [client_node]], device=self.device)

        new_edge_index = torch.cat([new_edge_index, hub_to_interface, interface_to_client], dim=1)

        # Crea attributi per nuovi edge (media degli attributi esistenti)
        if new_edge_attr.size(0) > 0:
            avg_edge_attr = new_edge_attr.mean(dim=0, keepdim=True)
            new_edge_attrs = avg_edge_attr.repeat(2, 1)  # 2 nuovi edge
            new_edge_attr = torch.cat([new_edge_attr, new_edge_attrs], dim=0)

        new_data.edge_index = new_edge_index
        new_data.edge_attr = new_edge_attr

        # Aggiorna statistiche
        self.episode_stats['total_node_changes'] += 1
        self.episode_stats['total_edge_changes'] += 1  # +2 edges, -1 edge = net +1

        return True, new_data, {
            'interface_node': interface_node,
            'redirected_connection': (hub_node, client_node)
        }

    def _split_component(self, large_node: int, data: Data) -> Tuple[bool, Data, Dict]:
        """
        Split Component Pattern: Divide componente grande in parti più piccole
        """
        node_features = data.x[large_node]
        loc = node_features[3].item()  # LinesOfCode

        # Validazione: deve essere abbastanza grande
        median_loc = data.x[:, 3].median().item()
        if loc <= median_loc * 1.5:
            return False, data, {'failure_reason': 'component_too_small'}

        new_data = copy.deepcopy(data)

        # Crea due nuovi nodi (split in 2 parti)
        node1_idx = new_data.x.size(0)
        node2_idx = new_data.x.size(0) + 1

        # Distribuisci features tra i nuovi nodi
        original_feats = new_data.x[large_node]

        # Primo nodo: prende ~60% delle features
        node1_feats = original_feats.clone()
        node1_feats[3] = original_feats[3] * 0.6  # 60% LOC
        node1_feats[1] = torch.ceil(original_feats[1] * 0.6)  # 60% FanOut
        node1_feats[4] = torch.ceil(original_feats[4] * 0.6)  # 60% Children

        # Secondo nodo: prende ~40% delle features
        node2_feats = original_feats.clone()
        node2_feats[3] = original_feats[3] * 0.4  # 40% LOC
        node2_feats[1] = torch.floor(original_feats[1] * 0.4)  # 40% FanOut
        node2_feats[4] = torch.floor(original_feats[4] * 0.4)  # 40% Children

        # Aggiorna nodo originale (diventa coordinatore)
        new_data.x[large_node, 3] = original_feats[3] * 0.1  # Mantieni solo 10% LOC
        new_data.x[large_node, 6] = 0.9  # Alta abstractness (diventa coordinatore)
        new_data.x[large_node, 5] = 0.3  # Bassa instability

        # Aggiungi nuovi nodi
        new_data.x = torch.cat([new_data.x, node1_feats.unsqueeze(0), node2_feats.unsqueeze(0)], dim=0)
        new_data.num_nodes += 2

        # Ridireziona alcune connessioni uscenti ai nuovi nodi
        outgoing_edges = (new_data.edge_index[0] == large_node)
        outgoing_targets = new_data.edge_index[1, outgoing_edges].unique()

        if len(outgoing_targets) > 0:
            # Distribuisci connessioni tra i nuovi nodi
            mid_point = len(outgoing_targets) // 2

            new_edges = []
            for i, target in enumerate(outgoing_targets):
                if i < mid_point:
                    # Collega node1 a questo target
                    new_edges.append([node1_idx, target.item()])
                else:
                    # Collega node2 a questo target
                    new_edges.append([node2_idx, target.item()])

            # Aggiungi connessioni coordinate: original -> node1, original -> node2
            new_edges.append([large_node, node1_idx])
            new_edges.append([large_node, node2_idx])

            if new_edges:
                new_edge_tensor = torch.tensor(new_edges, device=self.device).t()
                new_data.edge_index = torch.cat([new_data.edge_index, new_edge_tensor], dim=1)

                # Aggiungi attributi per nuovi edge
                if new_data.edge_attr.size(0) > 0:
                    avg_attr = new_data.edge_attr.mean(dim=0, keepdim=True)
                    new_attrs = avg_attr.repeat(len(new_edges), 1)
                    new_data.edge_attr = torch.cat([new_data.edge_attr, new_attrs], dim=0)

        self.episode_stats['total_node_changes'] += 2
        self.episode_stats['total_edge_changes'] += len(new_edges) if 'new_edges' in locals() else 0

        return True, new_data, {
            'split_nodes': [node1_idx, node2_idx],
            'original_loc': loc,
            'new_total_loc': node1_feats[3].item() + node2_feats[3].item()
        }

    def _introduce_mediator(self, hub_node: int, participant_node: int, data: Data) -> Tuple[bool, Data, Dict]:
        """
        Introduce Mediator Pattern: Crea mediatore per comunicazioni complesse
        """
        hub_features = data.x[hub_node]
        fan_in, fan_out = hub_features[0].item(), hub_features[1].item()

        # Validazione: deve avere alta interconnessione
        if fan_in < 2 or fan_out < 2:
            return False, data, {'failure_reason': 'insufficient_interconnection'}

        new_data = copy.deepcopy(data)
        mediator_node = new_data.x.size(0)

        # Features per mediatore (alta abstractness, moderata instability)
        mediator_feats = data.x[hub_node].clone()
        mediator_feats[3] = hub_features[3] * 0.3  # 30% LOC del hub
        mediator_feats[5] = 0.5  # Instability moderata
        mediator_feats[6] = 0.8  # Alta abstractness
        mediator_feats[0] = fan_in  # Mantieni fan-in
        mediator_feats[1] = fan_out  # Mantieni fan-out

        # Aggiungi mediatore
        new_data.x = torch.cat([new_data.x, mediator_feats.unsqueeze(0)], dim=0)
        new_data.num_nodes += 1

        # Trova nodi collegati al hub
        connected_nodes = []
        # Nodi con edge in entrata dal hub
        incoming = new_data.edge_index[1, new_data.edge_index[0] == hub_node].unique()
        # Nodi con edge in uscita verso il hub
        outgoing = new_data.edge_index[0, new_data.edge_index[1] == hub_node].unique()
        connected_nodes = torch.cat([incoming, outgoing]).unique()

        # Rimuovi self-connections
        connected_nodes = connected_nodes[connected_nodes != hub_node]

        if len(connected_nodes) > 0:
            # Ridireziona alcune connessioni attraverso il mediatore
            num_to_redirect = min(len(connected_nodes), 3)  # Max 3 redirections
            redirect_nodes = connected_nodes[:num_to_redirect]

            new_edges = []
            edges_to_remove = []

            for node in redirect_nodes:
                # Trova edge hub <-> node
                hub_to_node = ((new_data.edge_index[0] == hub_node) &
                               (new_data.edge_index[1] == node)).nonzero(as_tuple=False)
                node_to_hub = ((new_data.edge_index[0] == node) &
                               (new_data.edge_index[1] == hub_node)).nonzero(as_tuple=False)

                # Marca per rimozione
                edges_to_remove.extend(hub_to_node.squeeze(-1).tolist())
                edges_to_remove.extend(node_to_hub.squeeze(-1).tolist())

                # Aggiungi attraverso mediatore: node -> mediator -> hub
                new_edges.extend([[node.item(), mediator_node], [mediator_node, hub_node]])

            # Rimuovi edge diretti
            if edges_to_remove:
                keep_mask = torch.ones(new_data.edge_index.size(1), dtype=torch.bool)
                keep_mask[edges_to_remove] = False
                new_data.edge_index = new_data.edge_index[:, keep_mask]
                new_data.edge_attr = new_data.edge_attr[keep_mask]

            # Aggiungi nuovi edge
            if new_edges:
                new_edge_tensor = torch.tensor(new_edges, device=self.device).t()
                new_data.edge_index = torch.cat([new_data.edge_index, new_edge_tensor], dim=1)

                if new_data.edge_attr.size(0) > 0:
                    avg_attr = new_data.edge_attr.mean(dim=0, keepdim=True)
                    new_attrs = avg_attr.repeat(len(new_edges), 1)
                    new_data.edge_attr = torch.cat([new_data.edge_attr, new_attrs], dim=0)

        self.episode_stats['total_node_changes'] += 1
        self.episode_stats['total_edge_changes'] += len(new_edges) - len(
            edges_to_remove) if 'new_edges' in locals() else 0

        return True, new_data, {
            'mediator_node': mediator_node,
            'redirected_connections': len(redirect_nodes) if 'redirect_nodes' in locals() else 0
        }

    def _introduce_facade(self, complex_node: int, data: Data) -> Tuple[bool, Data, Dict]:
        """
        Introduce Facade Pattern: Semplifica interfaccia complessa
        """
        node_features = data.x[complex_node]
        fan_out = node_features[1].item()
        abstractness = node_features[6].item()

        # Validazione: alta complessità (alto fan-out, bassa abstractness)
        if fan_out < 3 or abstractness > 0.5:
            return False, data, {'failure_reason': 'not_complex_enough'}

        new_data = copy.deepcopy(data)
        facade_node = new_data.x.size(0)

        # Features per facade (alta abstractness, bassa instability)
        facade_feats = data.x[complex_node].clone()
        facade_feats[3] = node_features[3] * 0.2  # 20% LOC
        facade_feats[5] = 0.2  # Bassa instability
        facade_feats[6] = 0.9  # Molto alta abstractness
        facade_feats[1] = min(3, fan_out)  # Fan-out limitato

        # Aggiungi facade
        new_data.x = torch.cat([new_data.x, facade_feats.unsqueeze(0)], dim=0)
        new_data.num_nodes += 1

        # Trova nodi che dipendono dal nodo complesso
        dependent_nodes = new_data.edge_index[1, new_data.edge_index[0] == complex_node].unique()

        if len(dependent_nodes) > 0:
            # Ridireziona alcune dipendenze attraverso facade
            num_to_facade = min(len(dependent_nodes), max(2, len(dependent_nodes) // 2))
            facade_clients = dependent_nodes[:num_to_facade]

            edges_to_remove = []
            new_edges = []

            for client in facade_clients:
                # Trova edge complex -> client
                edge_idx = ((new_data.edge_index[0] == complex_node) &
                            (new_data.edge_index[1] == client)).nonzero(as_tuple=False)

                if len(edge_idx) > 0:
                    edges_to_remove.extend(edge_idx.squeeze(-1).tolist())
                    # Ridireziona: facade -> client, complex -> facade
                    new_edges.extend([[facade_node, client.item()]])

            # Aggiungi connessione complex -> facade
            new_edges.append([complex_node, facade_node])

            # Rimuovi edge diretti
            if edges_to_remove:
                keep_mask = torch.ones(new_data.edge_index.size(1), dtype=torch.bool)
                keep_mask[edges_to_remove] = False
                new_data.edge_index = new_data.edge_index[:, keep_mask]
                new_data.edge_attr = new_data.edge_attr[keep_mask]

            # Aggiungi nuovi edge
            if new_edges:
                new_edge_tensor = torch.tensor(new_edges, device=self.device).t()
                new_data.edge_index = torch.cat([new_data.edge_index, new_edge_tensor], dim=1)

                if new_data.edge_attr.size(0) > 0:
                    avg_attr = new_data.edge_attr.mean(dim=0, keepdim=True)
                    new_attrs = avg_attr.repeat(len(new_edges), 1)
                    new_data.edge_attr = torch.cat([new_data.edge_attr, new_attrs], dim=0)

        self.episode_stats['total_node_changes'] += 1
        self.episode_stats['total_edge_changes'] += len(new_edges) - len(
            edges_to_remove) if 'new_edges' in locals() else 0

        return True, new_data, {
            'facade_node': facade_node,
            'simplified_connections': len(facade_clients) if 'facade_clients' in locals() else 0
        }

    def _apply_dependency_injection(self, dependent_node: int, dependency_node: int, data: Data) -> Tuple[
        bool, Data, Dict]:
        """
        Apply Dependency Injection Pattern: Inverte controllo delle dipendenze
        """
        # Verifica che esista dipendenza diretta
        edge_exists = ((data.edge_index[0] == dependent_node) &
                       (data.edge_index[1] == dependency_node)).any().item()

        if not edge_exists:
            return False, data, {'failure_reason': 'no_direct_dependency'}

        dependent_features = data.x[dependent_node]
        instability = dependent_features[5].item()

        # Validazione: il nodo dipendente deve avere alta instability
        if instability < 0.3:
            return False, data, {'failure_reason': 'low_instability'}

        new_data = copy.deepcopy(data)
        container_node = new_data.x.size(0)

        # Features per DI container (alta abstractness, bassa instability)
        container_feats = data.x[dependent_node].clone()
        container_feats[3] = dependent_features[3] * 0.1  # 10% LOC
        container_feats[5] = 0.1  # Molto bassa instability
        container_feats[6] = 0.95  # Molto alta abstractness
        container_feats[0] = 1  # Fan-in basso
        container_feats[1] = 2  # Fan-out controllato

        # Aggiungi container
        new_data.x = torch.cat([new_data.x, container_feats.unsqueeze(0)], dim=0)
        new_data.num_nodes += 1

        # Ridireziona dipendenza attraverso container
        # Rimuovi edge diretto dependent -> dependency
        edge_mask = ~((new_data.edge_index[0] == dependent_node) &
                      (new_data.edge_index[1] == dependency_node))
        new_data.edge_index = new_data.edge_index[:, edge_mask]
        new_data.edge_attr = new_data.edge_attr[edge_mask]

        # Aggiungi: dependent -> container -> dependency
        new_edges = [
            [dependent_node, container_node],
            [container_node, dependency_node]
        ]

        new_edge_tensor = torch.tensor(new_edges, device=self.device).t()
        new_data.edge_index = torch.cat([new_data.edge_index, new_edge_tensor], dim=1)

        if new_data.edge_attr.size(0) > 0:
            avg_attr = new_data.edge_attr.mean(dim=0, keepdim=True)
            new_attrs = avg_attr.repeat(2, 1)
            new_data.edge_attr = torch.cat([new_data.edge_attr, new_attrs], dim=0)

        # Aggiorna features del nodo dipendente (riduce instability)
        new_data.x[dependent_node, 5] = max(0.1, instability * 0.7)  # Riduce instability

        self.episode_stats['total_node_changes'] += 1
        self.episode_stats['total_edge_changes'] += 1  # +2 edges, -1 edge = net +1

        return True, new_data, {
            'container_node': container_node,
            'injected_dependency': (dependent_node, dependency_node)
        }

    def _remove_unnecessary_dependency(self, source_node: int, target_node: int, data: Data) -> Tuple[bool, Data, Dict]:
        """
        Remove Unnecessary Dependency: Rimuove dipendenze non essenziali
        """
        # Verifica che esista l'edge
        edge_exists = ((data.edge_index[0] == source_node) &
                       (data.edge_index[1] == target_node)).any().item()

        if not edge_exists:
            return False, data, {'failure_reason': 'edge_not_found'}

        new_data = copy.deepcopy(data)

        # Rimuovi edge
        edge_mask = ~((new_data.edge_index[0] == source_node) &
                      (new_data.edge_index[1] == target_node))
        new_data.edge_index = new_data.edge_index[:, edge_mask]
        new_data.edge_attr = new_data.edge_attr[edge_mask]

        # Aggiorna features dei nodi coinvolti
        # Riduci fan-out del source
        if new_data.x[source_node, 1] > 0:
            new_data.x[source_node, 1] -= 1

        # Riduci fan-in del target
        if new_data.x[target_node, 0] > 0:
            new_data.x[target_node, 0] -= 1

        # Controlla se il grafo rimane connesso
        edge_list = new_data.edge_index.t().tolist()
        if edge_list:
            G = nx.DiGraph()
            G.add_edges_from(edge_list)
            if not nx.is_weakly_connected(G):
                return False, data, {'failure_reason': 'would_disconnect_graph'}

        self.episode_stats['total_edge_changes'] -= 1

        return True, new_data, {
            'removed_edge': (source_node, target_node),
            'remaining_edges': new_data.edge_index.size(1)
        }

    def _compute_comprehensive_metrics(self, data: Data) -> Dict[str, float]:
        """Calcola metriche complete per valutazione del progresso"""
        try:
            # Hub score dal discriminator
            with torch.no_grad():
                hub_score = self._eval_discriminator(data).item()

            # Metriche dalle features
            features = data.x
            fan_in = features[:, 0]
            fan_out = features[:, 1]
            pagerank = features[:, 2]
            loc = features[:, 3]
            instability = features[:, 5]

            # Metriche aggregate
            max_fan_out = fan_out.max().item()
            avg_fan_out = fan_out.mean().item()
            max_fan_in = fan_in.max().item()
            avg_instability = instability.mean().item()
            total_loc = loc.sum().item()

            # Distribuzione delle dimensioni
            loc_std = loc.std().item() if len(loc) > 1 else 0.0

            # Centralità
            max_pagerank = pagerank.max().item()
            avg_pagerank = pagerank.mean().item()

            # Densità del grafo
            n_nodes = data.x.size(0)
            n_edges = data.edge_index.size(1)
            density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

            return {
                'hub_score': hub_score,
                'max_fan_out': max_fan_out,
                'avg_fan_out': avg_fan_out,
                'max_fan_in': max_fan_in,
                'avg_instability': avg_instability,
                'total_loc': total_loc,
                'loc_std': loc_std,
                'max_pagerank': max_pagerank,
                'avg_pagerank': avg_pagerank,
                'density': density,
                'n_nodes': n_nodes,
                'n_edges': n_edges
            }
        except Exception:
            return {'hub_score': 0.5}  # Default fallback

    def _compute_pattern_reward(self, old_data: Data, new_data: Data,
                                pattern_name: str, pattern_info: Dict) -> float:
        """Calcola reward specifico per il pattern applicato"""
        reward = self.REW_PATTERN_SUCCESS

        # Bonus/penalty basati su appropriatezza del pattern
        old_metrics = self._compute_comprehensive_metrics(old_data)
        new_metrics = self._compute_comprehensive_metrics(new_data)

        if pattern_name == "EXTRACT_INTERFACE":
            # Dovrebbe ridurre fan-out e migliorare abstractness
            fan_out_reduction = old_metrics['max_fan_out'] - new_metrics['max_fan_out']
            reward += fan_out_reduction * 0.3

        elif pattern_name == "SPLIT_COMPONENT":
            # Dovrebbe migliorare distribuzione LOC
            if new_metrics['loc_std'] < old_metrics['loc_std']:
                reward += 0.5  # Bonus per migliore distribuzione

        elif pattern_name == "INTRODUCE_MEDIATOR":
            # Dovrebbe ridurre connessioni dirette
            density_reduction = old_metrics['density'] - new_metrics['density']
            reward += density_reduction * 1.0

        elif pattern_name == "REMOVE_UNNECESSARY_DEPENDENCY":
            # Dovrebbe ridurre complessità senza rompere funzionalità
            if new_metrics['hub_score'] < old_metrics['hub_score']:
                reward += 0.8  # Bonus per riduzione hub score

        return reward

    def _compute_progress_reward(self, old_metrics: Dict, new_metrics: Dict) -> float:
        """Calcola reward basato sul progresso generale"""
        reward = 0.0

        # Hub score improvement (principale obiettivo)
        hub_improvement = old_metrics['hub_score'] - new_metrics['hub_score']
        reward += hub_improvement * self.REW_HUB_REDUCTION

        # Centrality improvements
        centrality_improvement = (
                                         (old_metrics['max_pagerank'] - new_metrics['max_pagerank']) +
                                         (old_metrics['avg_pagerank'] - new_metrics['avg_pagerank'])
                                 ) / 2.0
        reward += centrality_improvement * self.REW_CENTRALITY_WEIGHT

        # Coupling reduction (fan-out reduction)
        coupling_improvement = (
                old_metrics['max_fan_out'] - new_metrics['max_fan_out']
        )
        reward += coupling_improvement * self.REW_COUPLING_WEIGHT

        # Architectural improvements
        instability_improvement = old_metrics['avg_instability'] - new_metrics['avg_instability']
        reward += instability_improvement * 0.2

        return reward

    def _compute_termination_reward(self, final_metrics: Dict) -> float:
        """Calcola reward per terminazione basato sullo stato finale"""
        final_hub_score = final_metrics['hub_score']
        initial_hub_score = self.initial_metrics['hub_score']

        improvement = initial_hub_score - final_hub_score

        if final_hub_score < 0.3:  # Soglia per "successo"
            # Terminazione positiva
            reward = self.REW_POS_TERM * (1.0 + improvement)

            # Bonus per efficienza (meno step = meglio)
            efficiency_bonus = max(0, (self.max_steps - self.steps) / self.max_steps) * 0.5
            reward += efficiency_bonus

        elif improvement > self.MIN_CHANGE_THRESHOLD:
            # Miglioramento parziale
            reward = self.REW_POS_TERM * 0.5 * improvement

        else:
            # Nessun miglioramento significativo
            reward = self.REW_NEG_TERM * (0.5 + final_hub_score)

        return reward

    def _check_early_termination(self) -> Tuple[bool, str]:
        """Controlla condizioni per terminazione anticipata"""

        # Max steps raggiunto
        if self.steps >= self.max_steps:
            return True, 'max_steps'

        # Troppe azioni invalide consecutive
        if self.invalid_action_count > max(5, self.steps * 0.7):
            return True, 'too_many_invalid'

        # Nessun progresso per troppo tempo
        if (self.enable_progress_tracking and
                self.steps_without_improvement > 15):
            return True, 'no_progress'

        # Grafo troppo grande
        if self.current_data.x.size(0) > self.max_allowed_nodes:
            return True, 'too_many_nodes'

        # Grafo disconnesso
        if self.current_data.edge_index.size(1) == 0:
            return True, 'disconnected_graph'

        return False, 'none'

    def _eval_discriminator(self, data: Data) -> torch.Tensor:
        """Valuta discriminator sul grafo corrente"""
        try:
            with torch.no_grad():
                logits = self.discriminator(data)
                probs = F.softmax(logits, dim=1)[:, 1]  # Probabilità di essere smelly
                return probs.mean()
        except Exception:
            return torch.tensor(0.5, device=self.device)

    def _update_episode_stats(self, info: Dict):
        """Aggiorna statistiche dell'episodio"""
        if info.get('valid_action'):
            pattern_name = info.get('pattern_applied')
            if pattern_name and info.get('pattern_success'):
                self.episode_stats['successful_patterns'] += 1

    def get_episode_summary(self) -> Dict[str, Any]:
        """Ottieni riassunto dell'episodio corrente"""
        current_metrics = self._compute_comprehensive_metrics(self.current_data)

        summary = {
            'steps_taken': self.steps,
            'valid_actions': self.valid_action_count,
            'invalid_actions': self.invalid_action_count,
            'success_rate': self.valid_action_count / max(1, self.steps),

            # Metriche di progresso
            'initial_hub_score': self.initial_metrics['hub_score'],
            'final_hub_score': current_metrics['hub_score'],
            'hub_improvement': self.initial_metrics['hub_score'] - current_metrics['hub_score'],

            # Statistiche pattern
            'patterns_applied': self.episode_stats['patterns_applied'],
            'successful_patterns': self.episode_stats['successful_patterns'],
            'pattern_success_rate': (
                    self.episode_stats['successful_patterns'] /
                    max(1, self.episode_stats['patterns_applied'])
            ),

            # Cambiamenti strutturali
            'nodes_added': self.current_data.x.size(0) - self.initial_n_nodes,
            'initial_edges': self.initial_data.edge_index.size(1),
            'final_edges': self.current_data.edge_index.size(1),
            'edge_changes': self.episode_stats['total_edge_changes'],

            # Pattern diversity
            'unique_patterns_tried': len(set(self.applied_patterns)) if self.enable_pattern_memory else 0,
            'hub_nodes_addressed': len(self.episode_stats['hub_nodes_addressed']),
        }

        # Aggiungi dettagli metriche
        summary.update({
            'metrics_improvement': {
                'max_fan_out': self.initial_metrics.get('max_fan_out', 0) - current_metrics.get('max_fan_out', 0),
                'avg_instability': self.initial_metrics.get('avg_instability', 0) - current_metrics.get(
                    'avg_instability', 0),
                'density': self.initial_metrics.get('density', 0) - current_metrics.get('density', 0),
            }
        })

        return summary

    def render(self, mode='human') -> Optional[str]:
        """Rendering per debugging (opzionale)"""
        if mode == 'human':
            summary = self.get_episode_summary()
            output = f"""
=== Dependency Refactor Environment ===
Step: {self.steps}/{self.max_steps}
Hub Score: {summary['final_hub_score']:.3f} (Δ{summary['hub_improvement']:+.3f})
Valid Actions: {summary['valid_actions']}/{self.steps} ({summary['success_rate']:.1%})
Patterns Applied: {summary['patterns_applied']} ({summary['pattern_success_rate']:.1%} success)
Graph: {self.current_data.x.size(0)} nodes, {self.current_data.edge_index.size(1)} edges
            """
            return output.strip()
        return None

    def close(self):
        """Cleanup resources"""
        # Reset tracking structures
        if hasattr(self, 'applied_patterns'):
            self.applied_patterns.clear()
        if hasattr(self, 'action_history'):
            self.action_history.clear()

        # Clear large objects
        self.current_data = None
        self.initial_data = None

    def get_action_space_info(self) -> Dict[str, Any]:
        """Informazioni sullo spazio delle azioni per debugging"""
        return {
            'action_space_shape': self.action_space.nvec.tolist(),
            'current_nodes': self.current_data.x.size(0) if self.current_data is not None else 0,
            'supported_patterns': list(self.REFACTORING_PATTERNS.values()),
            'pattern_descriptions': {
                'EXTRACT_INTERFACE': 'Creates interface between hub and clients',
                'SPLIT_COMPONENT': 'Divides large component into smaller parts',
                'INTRODUCE_MEDIATOR': 'Creates mediator for complex communications',
                'INTRODUCE_FACADE': 'Simplifies complex interface',
                'APPLY_DEPENDENCY_INJECTION': 'Inverts dependency control',
                'REMOVE_UNNECESSARY_DEPENDENCY': 'Removes non-essential dependencies'
            }
        }

    def validate_action_semantics(self, action: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Valida semantica dell'azione senza applicarla (per debugging/analisi)
        """
        source_node, target_node, pattern_type, terminate = action

        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'expected_outcome': None
        }

        if terminate == 1:
            validation['expected_outcome'] = 'termination'
            return validation

        # Controlla bounds
        current_n_nodes = self.current_data.x.size(0)
        if source_node >= current_n_nodes:
            validation['valid'] = False
            validation['errors'].append(f'source_node {source_node} >= {current_n_nodes}')

        if target_node >= current_n_nodes:
            validation['valid'] = False
            validation['errors'].append(f'target_node {target_node} >= {current_n_nodes}')

        if pattern_type >= len(self.REFACTORING_PATTERNS):
            validation['valid'] = False
            validation['errors'].append(f'pattern_type {pattern_type} >= {len(self.REFACTORING_PATTERNS)}')

        if not validation['valid']:
            return validation

        # Validazione pattern-specifica
        pattern_name = self.REFACTORING_PATTERNS[pattern_type]
        node_features = self.current_data.x[source_node]

        if pattern_name == "EXTRACT_INTERFACE":
            fan_out = node_features[1].item()
            if fan_out < 2:
                validation['warnings'].append('Low fan-out for interface extraction')

            edge_exists = ((self.current_data.edge_index[0] == source_node) &
                           (self.current_data.edge_index[1] == target_node)).any().item()
            if not edge_exists:
                validation['valid'] = False
                validation['errors'].append('No direct connection for interface extraction')

        elif pattern_name == "SPLIT_COMPONENT":
            loc = node_features[3].item()
            median_loc = self.current_data.x[:, 3].median().item()
            if loc <= median_loc * 1.5:
                validation['valid'] = False
                validation['errors'].append('Component too small for splitting')

        # Aggiungi più validazioni per altri pattern...

        if validation['valid']:
            validation['expected_outcome'] = f'apply_{pattern_name.lower()}'

        return validation