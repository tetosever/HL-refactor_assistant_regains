"""
Ambiente di reinforcement learning per la rifattorizzazione automatica
di sub-graph 1-hop di dependency graph usando PyTorch Geometric.

CORRECTED VERSION for PPO - Maintains all original environment logic
but fixes PPO interface compatibility issues.
"""

import gym
from gym import spaces
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from functools import lru_cache

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Standard features per node (UNCHANGED)
HUB_FEATURES = [
    'fan_in', 'fan_out', 'degree_centrality', 'in_out_ratio',
    'pagerank', 'betweenness_centrality', 'closeness_centrality'
]


class HubTracker:
    """
    MAINTAINED: Original robust hub tracking system
    """

    def __init__(self, initial_hub_idx: int):
        self.original_hub_idx = initial_hub_idx
        self.original_hub_id = f"hub_original_{initial_hub_idx}"
        self.current_hub_idx = initial_hub_idx
        self.hub_lost = False

        self.node_id_mapping = {}
        self.reverse_id_mapping = {}
        self.next_node_id = 0

    def initialize_tracking(self, num_nodes: int):
        """UNCHANGED: Initialize tracking system"""
        self.node_id_mapping.clear()
        self.reverse_id_mapping.clear()
        self.next_node_id = 0

        for current_index in range(num_nodes):
            stable_id = f"node_{self.next_node_id}"
            self.node_id_mapping[stable_id] = current_index
            self.reverse_id_mapping[current_index] = stable_id
            self.next_node_id += 1

        self.original_hub_id = self.reverse_id_mapping[self.original_hub_idx]

    def get_current_hub_index(self, data: Data) -> int:
        """UNCHANGED: Find current hub index"""
        if self.original_hub_id is None:
            return self._find_fallback_hub(data)

        current_index = self.node_id_mapping.get(self.original_hub_id, None)

        if current_index is None or current_index >= data.num_nodes:
            print(f"Warning: Hub {self.original_hub_id} lost! Using fallback.")
            self.hub_lost = True
            return self._find_fallback_hub(data)

        return current_index

    def _find_fallback_hub(self, data: Data) -> int:
        """UNCHANGED: Find fallback hub"""
        if data.edge_index.size(1) > 0:
            degrees = torch.bincount(data.edge_index[0], minlength=data.num_nodes)
            return degrees.argmax().item()
        else:
            return 0

    def update_after_node_addition(self, num_new_nodes: int, start_index: int):
        """UNCHANGED: Update mapping after adding nodes"""
        for i in range(num_new_nodes):
            new_index = start_index + i
            stable_id = f"node_{self.next_node_id}"
            self.node_id_mapping[stable_id] = new_index
            self.reverse_id_mapping[new_index] = stable_id
            self.next_node_id += 1

    def rebuild_mapping(self, old_num_nodes: int, new_num_nodes: int):
        """UNCHANGED: Rebuild mapping after graph modifications"""
        new_mapping = {}
        new_reverse_mapping = {}

        for old_index in range(min(old_num_nodes, new_num_nodes)):
            if old_index in self.reverse_id_mapping:
                stable_id = self.reverse_id_mapping[old_index]
                new_mapping[stable_id] = old_index
                new_reverse_mapping[old_index] = stable_id

        if new_num_nodes > old_num_nodes:
            self.update_after_node_addition(
                new_num_nodes - old_num_nodes,
                old_num_nodes
            )
            for new_index in range(old_num_nodes, new_num_nodes):
                if new_index in self.reverse_id_mapping:
                    stable_id = self.reverse_id_mapping[new_index]
                    new_mapping[stable_id] = new_index
                    new_reverse_mapping[new_index] = stable_id

        self.node_id_mapping = new_mapping
        self.reverse_id_mapping = new_reverse_mapping


class RefactorEnv(gym.Env):
    """
    CORRECTED: Graph Refactoring Environment for PPO

    MAIN CHANGES:
    - reset() returns Data object instead of observation array
    - step() returns Data object as next_state
    - Simplified reward function for PPO
    - Removed observation space conversion (not needed for PPO)
    - Maintains all original environment logic and capabilities
    """

    def __init__(self,
                 data_path: str,
                 discriminator=None,
                 max_steps: int = 20,
                 reward_weights: Optional[Dict[str, float]] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        MAINTAINED: Original initialization with all parameters
        """
        super(RefactorEnv, self).__init__()

        self.device = device
        self.max_steps = max_steps
        self.discriminator = discriminator

        # MAINTAINED: Original reward weights structure
        self.reward_weights = reward_weights or {
            'hub_weight': 10.0,
            'step_valid': 0.0,
            'step_invalid': -0.1,
            'time_penalty': -0.02,
            'early_stop_penalty': -0.5,
            'cycle_penalty': -0.2,
            'duplicate_penalty': -0.1,
            'adversarial_weight': 2.0,
            'patience': 15
        }

        # MAINTAINED: Original performance tracking
        self.best_hub_score = 0.0
        self.no_improve_steps = 0
        self.disc_start = 0.5
        self.prev_disc_score = None

        # MAINTAINED: Original data loading and preprocessing
        print("Loading and preprocessing data...")
        self.original_data_list = self._load_and_preprocess_data(data_path)

        # MAINTAINED: Original state variables
        self.current_data = None
        self.current_step = 0
        self.initial_metrics = {}
        self.prev_hub_score = 0.0
        self.hub_tracker = None

        # MAINTAINED: Original action space
        self.num_actions = 7
        self.action_space = spaces.Discrete(self.num_actions)

        # CORRECTED: Simplified observation space (not used by PPO but kept for compatibility)
        max_nodes = max([data.num_nodes for data in self.original_data_list])
        self.max_nodes = max_nodes
        obs_dim = max_nodes * 7 + max_nodes * max_nodes + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        # MAINTAINED: Original feature scaler
        self.feature_scaler = None
        self._fit_feature_scaler()

        print(f"Environment initialized: {len(self.original_data_list)} graphs, max_nodes={max_nodes}")

    # MAINTAINED: All original computational methods
    def _get_discriminator_score(self) -> Optional[float]:
        """UNCHANGED: Get discriminator score"""
        if not hasattr(self, 'discriminator') or self.discriminator is None:
            return None

        try:
            with torch.no_grad():
                disc_output = self.discriminator(self.current_data)
                if isinstance(disc_output, dict):
                    p_smelly = torch.softmax(disc_output['logits'], dim=1)[0, 1].item()
                else:
                    p_smelly = torch.softmax(disc_output, dim=1)[0, 1].item()
                return p_smelly
        except Exception:
            return None

    def _fit_feature_scaler(self):
        """UNCHANGED: Fit feature scaler"""
        print("Training feature scaler...")

        all_features = []
        sample_size = min(100, len(self.original_data_list))
        sampled_indices = np.random.choice(len(self.original_data_list), sample_size, replace=False)

        for idx in sampled_indices:
            data = self.original_data_list[idx]
            G = to_networkx(data, to_undirected=False)
            node_features = self._compute_node_features(G)

            for node_feats in node_features.values():
                feature_vector = [node_feats[feat] for feat in HUB_FEATURES]
                all_features.append(feature_vector)

        if all_features:
            self.feature_scaler = StandardScaler()
            self.feature_scaler.fit(np.array(all_features))
            print(f"Scaler trained on {len(all_features)} samples")
        else:
            print("No features found for training scaler")

    @staticmethod
    def _compute_centrality_metrics(G: nx.DiGraph) -> Tuple[Dict, Dict, Dict]:
        """UNCHANGED: Compute centrality metrics efficiently"""
        num_nodes = len(G)

        if num_nodes <= 100:
            try:
                pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-4)
                betweenness = nx.betweenness_centrality(G, normalized=True)
                closeness = nx.closeness_centrality(G)
            except:
                pagerank = {n: 1.0 / num_nodes for n in G.nodes()}
                betweenness = {n: 0.0 for n in G.nodes()}
                closeness = {n: 1.0 / max(num_nodes - 1, 1) for n in G.nodes()}
        else:
            total_edges = G.number_of_edges()
            pagerank = {n: float(G.degree(n)) / (2 * total_edges + 1e-8) for n in G.nodes()}
            betweenness = {n: 0.0 for n in G.nodes()}
            closeness = {n: 1.0 / max(num_nodes - 1, 1) for n in G.nodes()}

        return pagerank, betweenness, closeness

    @staticmethod
    def _compute_node_features(G: nx.DiGraph) -> Dict[str, Dict[str, float]]:
        """UNCHANGED: Compute features for all nodes"""
        if not G.is_directed():
            raise ValueError("Graph must be directed")

        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        pagerank, betweenness, closeness = RefactorEnv._compute_centrality_metrics(G)

        node_features = {}
        num_nodes = len(G)

        for node in G.nodes():
            fan_in = float(in_degrees.get(node, 0))
            fan_out = float(out_degrees.get(node, 0))
            total_degree = fan_in + fan_out

            node_features[str(node)] = {
                'fan_in': fan_in,
                'fan_out': fan_out,
                'degree_centrality': total_degree / max(num_nodes - 1, 1),
                'in_out_ratio': fan_in / (fan_out + 1e-8),
                'pagerank': float(pagerank.get(node, 0)),
                'betweenness_centrality': float(betweenness.get(node, 0)),
                'closeness_centrality': float(closeness.get(node, 0))
            }

        return node_features

    def compute_hub_score_from_tensor(self, data: Data, hub_index: int) -> float:
        """UNCHANGED: Optimized hub score calculation"""
        if hub_index >= data.num_nodes or hub_index < 0:
            return 0.0

        hub_features = data.x[hub_index]
        fan_in = hub_features[0].item()
        fan_out = hub_features[1].item()
        degree_centrality = hub_features[2].item()
        pagerank_hub = hub_features[4].item()
        closeness_centrality = hub_features[6].item()

        total_degree = fan_in + fan_out
        max_possible_degree = 2 * max(data.num_nodes - 1, 1)
        normalized_total_degree = total_degree / max_possible_degree

        hub_score = (
            0.30 * normalized_total_degree +
            0.35 * degree_centrality +
            0.25 * pagerank_hub +
            0.10 * closeness_centrality
        )

        return float(np.clip(hub_score, 0.0, 1.0))

    def _calculate_metrics(self, data: Data) -> Dict[str, float]:
        """UNCHANGED: Calculate essential metrics"""
        try:
            current_hub = self.hub_tracker.get_current_hub_index(data)
            hub_score = self.compute_hub_score_from_tensor(data, current_hub)

            num_nodes = int(data.num_nodes)
            num_edges = int(data.edge_index.shape[1])

            try:
                G = to_networkx(data, to_undirected=True)
                connected = float(nx.is_connected(G))
            except:
                connected = 0.0

            return {
                'hub_score': float(hub_score),
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'connected': connected
            }

        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {
                'hub_score': 0.0,
                'num_nodes': 0,
                'num_edges': 0,
                'connected': 0.0
            }

    def _create_fresh_data_object(self, x: torch.Tensor, edge_index: torch.Tensor) -> Data:
        """UNCHANGED: Create Data object with fresh features"""
        try:
            num_nodes = x.size(0)

            G = nx.DiGraph()
            G.add_nodes_from(range(num_nodes))

            if edge_index.numel() > 0:
                valid_edges = []
                for i in range(edge_index.size(1)):
                    src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                    if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                        valid_edges.append((src, dst))

                if valid_edges:
                    G.add_edges_from(valid_edges)

            node_features = self._compute_node_features(G)

            feature_matrix = []
            for node_id in range(num_nodes):
                if str(node_id) in node_features:
                    feature_vector = [node_features[str(node_id)][feat] for feat in HUB_FEATURES]
                else:
                    feature_vector = [0.0] * len(HUB_FEATURES)
                feature_matrix.append(feature_vector)

            feature_matrix = np.array(feature_matrix)

            if self.feature_scaler is not None:
                try:
                    feature_matrix = self.feature_scaler.transform(feature_matrix)
                except Exception as e:
                    print(f"Normalization failed: {e}")
                    feature_matrix = (feature_matrix - feature_matrix.mean(axis=0)) / (
                        feature_matrix.std(axis=0) + 1e-8)

            new_data = Data(
                x=torch.tensor(feature_matrix, dtype=torch.float32, device=self.device),
                edge_index=edge_index.clone(),
                num_nodes=num_nodes
            )

            return new_data

        except Exception as e:
            print(f"Error creating Data object: {e}")
            return Data(
                x=torch.zeros((x.size(0), len(HUB_FEATURES)), dtype=torch.float32, device=self.device),
                edge_index=torch.empty((2, 0), dtype=torch.long, device=self.device),
                num_nodes=x.size(0)
            )

    def _rebuild_graph_with_fresh_data(self, new_x: torch.Tensor, new_edge_index: torch.Tensor) -> None:
        """UNCHANGED: Rebuild graph with hub tracking"""
        old_num_nodes = self.current_data.num_nodes
        num_nodes = new_x.size(0)
        self.max_nodes = max(self.max_nodes, num_nodes)

        valid_edges = []
        for i in range(new_edge_index.size(1)):
            src, dst = new_edge_index[0, i].item(), new_edge_index[1, i].item()
            if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                valid_edges.append([src, dst])

        if valid_edges:
            filtered_edge_index = torch.tensor(valid_edges, dtype=torch.long, device=self.device).t().contiguous()
        else:
            filtered_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

        self.current_data = self._create_fresh_data_object(new_x, filtered_edge_index)
        self.hub_tracker.rebuild_mapping(old_num_nodes, self.current_data.num_nodes)

    def _load_and_preprocess_data(self, data_path: str) -> List[Data]:
        """UNCHANGED: Load and preprocess data"""
        data_dir = Path(data_path)
        if not data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {data_path}")

        pt_files = list(data_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {data_path}")

        print(f"Found {len(pt_files)} .pt files")

        data_list = []
        for pt_file in pt_files:
            try:
                data = torch.load(pt_file, map_location=self.device)

                if isinstance(data, dict):
                    if 'data' in data:
                        graph_data = data['data']
                    else:
                        graph_data = Data(x=data['x'], edge_index=data['edge_index'])
                elif isinstance(data, Data):
                    graph_data = data
                else:
                    print(f"Unrecognized format in {pt_file}")
                    continue

                if hasattr(graph_data, 'x') and hasattr(graph_data, 'edge_index'):
                    if graph_data.x.size(1) == 7:
                        data_list.append(graph_data)
                    else:
                        print(f"{pt_file}: {graph_data.x.size(1)} features instead of 7")
                else:
                    print(f"{pt_file}: missing x or edge_index attributes")

            except Exception as e:
                print(f"Error loading {pt_file}: {e}")
                continue

        if not data_list:
            raise ValueError("No valid data loaded")

        print("Normalizing features...")
        scaler = StandardScaler()
        all_features = torch.cat([data.x for data in data_list], dim=0)
        scaler.fit(all_features.cpu().numpy())

        processed_data = []
        for data in data_list:
            normalized_features = scaler.transform(data.x.cpu().numpy())

            clean_data = Data(
                x=torch.tensor(normalized_features, dtype=torch.float32, device=self.device),
                edge_index=data.edge_index.to(self.device),
                num_nodes=data.x.size(0)
            )
            processed_data.append(clean_data)

        print(f"Processed {len(processed_data)} sub-graphs")
        return processed_data

    def reset(self, graph_idx: Optional[int] = None) -> Data:
        """
        CORRECTED: Reset environment returning Data object for PPO
        """
        if graph_idx is None:
            graph_idx = np.random.randint(0, len(self.original_data_list))

        original_data = self.original_data_list[graph_idx]

        self.current_data = Data(
            x=original_data.x.clone(),
            edge_index=original_data.edge_index.clone(),
            num_nodes=original_data.x.size(0)
        )

        self.current_step = 0

        # Find initial hub
        if self.current_data.edge_index.size(1) > 0:
            degrees = torch.bincount(self.current_data.edge_index[0], minlength=self.current_data.num_nodes)
            initial_hub = degrees.argmax().item()
        else:
            initial_hub = 0

        # Initialize hub tracker
        self.hub_tracker = HubTracker(initial_hub)
        self.hub_tracker.initialize_tracking(self.current_data.num_nodes)

        # Calculate initial metrics
        self.initial_metrics = self._calculate_metrics(self.current_data)
        self.prev_hub_score = self.initial_metrics['hub_score']
        self.best_hub_score = self.prev_hub_score
        self.no_improve_steps = 0

        # Initialize discriminator baseline
        self.prev_disc_score = self._get_discriminator_score()
        self.disc_start = self.prev_disc_score if self.prev_disc_score is not None else 0.5

        # CORRECTED: Return Data object instead of observation array
        return self.current_data

    def step(self, action: int) -> Tuple[Data, float, bool, Dict]:
        """
        CORRECTED: Step function returning Data object for PPO
        """
        if self.current_data is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Save previous state for reward calculation
        prev_hub_score = self._calculate_metrics(self.current_data)['hub_score']
        prev_disc_score = self._get_discriminator_score()

        self.current_step += 1

        # Apply action
        success = self._apply_action(action)

        # Calculate current state
        current_metrics = self._calculate_metrics(self.current_data)
        current_hub_score = current_metrics['hub_score']
        current_disc_score = self._get_discriminator_score()

        # CORRECTED: Simplified reward function for PPO
        # Main reward: hub score improvement
        hub_improvement = prev_hub_score - current_hub_score
        reward = self.reward_weights['hub_weight'] * hub_improvement

        # Action validity penalty
        if not success:
            reward += self.reward_weights['step_invalid']

        # Time penalty
        reward += self.reward_weights['time_penalty']

        # Adversarial reward (if discriminator available)
        if prev_disc_score is not None and current_disc_score is not None:
            disc_improvement = prev_disc_score - current_disc_score
            reward += self.reward_weights['adversarial_weight'] * disc_improvement

        # Structural penalties
        if success:
            reward += self._check_structural_penalties()

        # Anti early STOP penalty
        if action == 6 and self.current_step <= 2:
            reward += self.reward_weights['early_stop_penalty']

        # Update tracking
        if current_hub_score < self.best_hub_score:
            self.best_hub_score = current_hub_score
            self.no_improve_steps = 0
        else:
            self.no_improve_steps += 1

        # Determine termination
        done = (action == 6) or (self.current_step >= self.max_steps)

        # Early stopping with patience
        patience = self.reward_weights.get('patience', 15)
        if self.no_improve_steps >= patience:
            done = True

        # Info for monitoring
        info = {
            'action_success': success,
            'hub_improvement_step': hub_improvement,
            'hub_improvement_total': self.initial_metrics['hub_score'] - current_hub_score,
            'current_hub_score': current_hub_score,
            'best_hub_score': self.best_hub_score,
            'no_improve_steps': self.no_improve_steps,
            'current_step': self.current_step,
            'hub_lost': self.hub_tracker.hub_lost if self.hub_tracker else False
        }

        # CORRECTED: Return current_data (Data object) for PPO
        return self.current_data, reward, done, info

    # CORRECTED: Add helper method for PPO trainer
    def get_global_features(self) -> torch.Tensor:
        """Helper method to extract global features for PPO"""
        return self._extract_global_features(self.current_data)

    def _extract_global_features(self, data: Data) -> torch.Tensor:
        """CORRECTED: Extract global features (4 features for PPO)"""
        metrics = self._calculate_metrics(data)

        global_features = torch.tensor([
            metrics['hub_score'],
            metrics['num_nodes'],
            metrics['num_edges'],
            metrics['connected']
        ], dtype=torch.float32, device=self.device)

        if global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)

        return global_features

    # MAINTAINED: All original action methods
    def _apply_action(self, action: int) -> bool:
        """UNCHANGED: Apply action to graph"""
        try:
            if action == 0:
                return self._remove_edge()
            elif action == 1:
                return self._add_edge()
            elif action == 2:
                return self._move_edge()
            elif action == 3:
                return self._extract_method()
            elif action == 4:
                return self._extract_abstract_unit()
            elif action == 5:
                return self._extract_unit()
            elif action == 6:
                return True  # STOP
            else:
                return False
        except Exception as e:
            print(f"Error applying action {action}: {e}")
            return False

    def _remove_edge(self) -> bool:
        """UNCHANGED: Remove edge from hub"""
        current_hub = self.hub_tracker.get_current_hub_index(self.current_data)
        edge_index = self.current_data.edge_index

        hub_edges = []
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if u == current_hub and u != v:
                hub_edges.append(i)

        if not hub_edges:
            return False

        edge_to_remove = np.random.choice(hub_edges)
        mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=self.device)
        mask[edge_to_remove] = False
        new_edge_index = edge_index[:, mask]

        if self._is_connected(new_edge_index):
            self._rebuild_graph_with_fresh_data(self.current_data.x, new_edge_index)
            return True

        return False

    def _add_edge(self) -> bool:
        """UNCHANGED: Add edge from hub"""
        current_hub = self.hub_tracker.get_current_hub_index(self.current_data)
        edge_index = self.current_data.edge_index

        connected_nodes = set()
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if u == current_hub:
                connected_nodes.add(v)

        possible_targets = []
        for node in range(self.current_data.num_nodes):
            if node not in connected_nodes and node != current_hub:
                possible_targets.append(node)

        if not possible_targets:
            return False

        target = np.random.choice(possible_targets)
        new_edge = torch.tensor([[current_hub], [target]], dtype=torch.long, device=self.device)
        new_edge_index = torch.cat([edge_index, new_edge], dim=1)

        self._rebuild_graph_with_fresh_data(self.current_data.x, new_edge_index)
        return True

    def _move_edge(self) -> bool:
        """UNCHANGED: Move edge"""
        return self._remove_edge() and self._add_edge()

    def _extract_method(self) -> bool:
        """UNCHANGED: Extract method refactoring"""
        edge_index = self.current_data.edge_index

        if edge_index.shape[1] == 0:
            return False

        edge_idx = np.random.randint(0, edge_index.shape[1])
        u, v = edge_index[0, edge_idx].item(), edge_index[1, edge_idx].item()

        if u == v:
            return False

        u_features = self.current_data.x[u]
        v_features = self.current_data.x[v]
        method_features = ((u_features + v_features) / 2).unsqueeze(0)

        method_idx = self.current_data.x.size(0)
        new_edges = []

        for i in range(edge_index.shape[1]):
            if i != edge_idx:
                new_edges.append((edge_index[0, i].item(), edge_index[1, i].item()))

        new_edges.append((u, method_idx))
        new_edges.append((method_idx, v))

        v_incoming = [(src, dst) for src, dst in new_edges if dst == v and src != method_idx]
        if len(v_incoming) > 1:
            num_reassign = min(2, len(v_incoming) // 2)
            to_reassign = np.random.choice(len(v_incoming), num_reassign, replace=False)
            for idx in to_reassign:
                src, _ = v_incoming[idx]
                new_edges.remove((src, v))
                new_edges.append((src, method_idx))

        new_x = torch.cat([self.current_data.x, method_features], dim=0)
        new_edge_index = torch.tensor(new_edges, dtype=torch.long, device=self.device).t().contiguous()
        self._rebuild_graph_with_fresh_data(new_x, new_edge_index)

        return True

    def _extract_abstract_unit(self) -> bool:
        """UNCHANGED: Extract abstract unit"""
        edge_index = self.current_data.edge_index

        if edge_index.shape[1] < 3:
            return False

        targets = {}
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if dst not in targets:
                targets[dst] = []
            targets[dst].append(src)

        common_targets = [(dst, srcs) for dst, srcs in targets.items()
                         if len(set(srcs)) >= 2]

        if not common_targets:
            return False

        target_dst, source_nodes = common_targets[np.random.randint(len(common_targets))]
        unique_sources = list(set(source_nodes))

        if len(unique_sources) < 2:
            return False

        num_to_abstract = min(3, len(unique_sources))
        selected_sources = np.random.choice(unique_sources, num_to_abstract, replace=False)

        abstract_idx = self.current_data.x.size(0)
        selected_features = self.current_data.x[selected_sources]
        abstract_features = selected_features.mean(dim=0, keepdim=True)

        new_edges = []
        removed_edges = set()

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in selected_sources and dst == target_dst:
                removed_edges.add(i)

        for i in range(edge_index.shape[1]):
            if i not in removed_edges:
                new_edges.append((edge_index[0, i].item(), edge_index[1, i].item()))

        new_edges.append((abstract_idx, target_dst))
        for src in selected_sources:
            new_edges.append((src, abstract_idx))

        new_x = torch.cat([self.current_data.x, abstract_features], dim=0)
        new_edge_index = torch.tensor(new_edges, dtype=torch.long, device=self.device).t().contiguous()
        self._rebuild_graph_with_fresh_data(new_x, new_edge_index)

        return True

    def _extract_unit(self) -> bool:
        """UNCHANGED: Extract unit"""
        current_hub = self.hub_tracker.get_current_hub_index(self.current_data)

        if current_hub >= self.current_data.num_nodes:
            return False

        edge_index = self.current_data.edge_index

        hub_neighbors = []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src == current_hub and dst != current_hub:
                hub_neighbors.append(dst)

        hub_neighbors = list(set(hub_neighbors))

        if len(hub_neighbors) < 2:
            return False

        mid_point = len(hub_neighbors) // 2
        group1 = hub_neighbors[:mid_point]
        group2 = hub_neighbors[mid_point:]

        if not group1 or not group2:
            return False

        unit1_idx = self.current_data.x.size(0)
        unit2_idx = unit1_idx + 1

        hub_features = self.current_data.x[current_hub]
        group1_features = self.current_data.x[group1].mean(dim=0) if group1 else hub_features
        group2_features = self.current_data.x[group2].mean(dim=0) if group2 else hub_features

        unit1_features = ((hub_features + group1_features) / 2).unsqueeze(0)
        unit2_features = ((hub_features + group2_features) / 2).unsqueeze(0)

        new_edges = []

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src != current_hub and dst != current_hub:
                new_edges.append((src, dst))
            elif src == current_hub and dst in group1:
                new_edges.append((unit1_idx, dst))
            elif src == current_hub and dst in group2:
                new_edges.append((unit2_idx, dst))
            elif dst == current_hub:
                new_edges.append((src, dst))

        new_edges.append((current_hub, unit1_idx))
        new_edges.append((current_hub, unit2_idx))

        new_features = torch.cat([unit1_features, unit2_features], dim=0)
        new_x = torch.cat([self.current_data.x, new_features], dim=0)
        new_edge_index = torch.tensor(new_edges, dtype=torch.long, device=self.device).t().contiguous()
        self._rebuild_graph_with_fresh_data(new_x, new_edge_index)

        return True

    def _check_structural_penalties(self) -> float:
        """UNCHANGED: Check structural penalties"""
        penalty = 0.0

        try:
            G = to_networkx(self.current_data, to_undirected=False)

            try:
                next(nx.simple_cycles(G))
                penalty += self.reward_weights.get('cycle_penalty', -0.2)
            except StopIteration:
                pass

            seen_edges = set()
            for u, v in self.current_data.edge_index.t().tolist():
                edge = (u, v)
                if edge in seen_edges:
                    penalty += self.reward_weights.get('duplicate_penalty', -0.1)
                    break
                seen_edges.add(edge)

        except Exception:
            penalty += self.reward_weights.get('cycle_penalty', -0.2)

        return penalty

    def _is_connected(self, edge_index: torch.Tensor) -> bool:
        """UNCHANGED: Check graph connectivity"""
        try:
            if edge_index.size(1) == 0:
                return self.current_data.num_nodes <= 1

            temp_data = Data(edge_index=edge_index, num_nodes=self.current_data.num_nodes)
            G = to_networkx(temp_data, to_undirected=False)
            return nx.is_weakly_connected(G)
        except:
            return False

    # MAINTAINED: All original utility methods
    def render(self, mode: str = 'human'):
        """UNCHANGED: Render environment state"""
        if self.current_data is None:
            print("Environment not initialized")
            return

        metrics = self._calculate_metrics(self.current_data)
        current_hub = self.hub_tracker.get_current_hub_index(self.current_data) if self.hub_tracker else 0

        print(f"\n{'='*50}")
        print(f"ENVIRONMENT STATE - Step {self.current_step}")
        print(f"{'='*50}")
        print(f"Hub: ID {self.hub_tracker.original_hub_id if self.hub_tracker else 'N/A'} -> index {current_hub}")
        print(f"Hub Score: {metrics['hub_score']:.4f}")
        print(f"Graph: {metrics['num_nodes']} nodes, {metrics['num_edges']} edges")
        print(f"Connected: {'Yes' if metrics['connected'] else 'No'}")

        if self.hub_tracker:
            print(f"Nodes tracked: {len(self.hub_tracker.node_id_mapping)}")
            print(f"Hub lost: {'Yes' if self.hub_tracker.hub_lost else 'No'}")

        print(f"Best hub score: {self.best_hub_score:.4f}")
        print(f"Steps without improvement: {self.no_improve_steps}")
        print(f"{'='*50}")

    def get_hub_info(self) -> Dict:
        """UNCHANGED: Get hub information"""
        if not self.hub_tracker or self.current_data is None:
            return {}

        current_hub_idx = self.hub_tracker.get_current_hub_index(self.current_data)

        info = {
            'original_hub_id': self.hub_tracker.original_hub_id,
            'current_hub_index': current_hub_idx,
            'hub_lost': self.hub_tracker.hub_lost,
            'hub_score': self.compute_hub_score_from_tensor(self.current_data, current_hub_idx),
            'total_nodes_tracked': len(self.hub_tracker.node_id_mapping)
        }

        if current_hub_idx < self.current_data.num_nodes:
            hub_features = self.current_data.x[current_hub_idx]
            info.update({
                'hub_fan_in': hub_features[0].item(),
                'hub_fan_out': hub_features[1].item(),
                'hub_degree_centrality': hub_features[2].item(),
                'hub_pagerank': hub_features[4].item(),
                'hub_closeness_centrality': hub_features[6].item()
            })

        return info

    def get_action_mask(self) -> np.ndarray:
        """UNCHANGED: Get valid action mask"""
        if self.current_data is None:
            return np.ones(self.num_actions, dtype=bool)

        mask = np.ones(self.num_actions, dtype=bool)

        try:
            current_hub = self.hub_tracker.get_current_hub_index(self.current_data)
            edge_index = self.current_data.edge_index

            # Action 0 (RemoveEdge): requires outgoing edges from hub
            hub_outgoing = 0
            for i in range(edge_index.shape[1]):
                u, v = edge_index[0, i].item(), edge_index[1, i].item()
                if u == current_hub and u != v:
                    hub_outgoing += 1
                    break
            mask[0] = (hub_outgoing > 0)

            # Action 1 (AddEdge): requires unconnected nodes
            connected_nodes = set()
            for i in range(edge_index.shape[1]):
                u, v = edge_index[0, i].item(), edge_index[1, i].item()
                if u == current_hub:
                    connected_nodes.add(v)

            available_targets = self.current_data.num_nodes - len(connected_nodes) - 1
            mask[1] = (available_targets > 0)

            # Action 2 (MoveEdge): requires both remove and add possible
            mask[2] = mask[0] and mask[1]

            # Action 3 (ExtractMethod): requires at least one edge
            mask[3] = (edge_index.shape[1] > 0)

            # Action 4 (ExtractAbstractUnit): requires at least 3 edges
            mask[4] = (edge_index.shape[1] >= 3)

            # Action 5 (ExtractUnit): requires at least 2 hub successors
            hub_successors = set()
            for i in range(edge_index.shape[1]):
                u, v = edge_index[0, i].item(), edge_index[1, i].item()
                if u == current_hub and u != v:
                    hub_successors.add(v)
            mask[5] = (len(hub_successors) >= 2)

            # Action 6 (STOP): always available
            mask[6] = True

        except Exception as e:
            print(f"Error calculating action mask: {e}")
            mask = np.ones(self.num_actions, dtype=bool)

        return mask

    def get_performance_stats(self) -> Dict:
        """UNCHANGED: Get performance statistics"""
        if self.current_data is None:
            return {}

        current_metrics = self._calculate_metrics(self.current_data)

        stats = {
            'current_hub_score': current_metrics['hub_score'],
            'initial_hub_score': self.initial_metrics.get('hub_score', 0.0),
            'best_hub_score': self.best_hub_score,
            'hub_improvement': self.initial_metrics.get('hub_score', 0.0) - self.best_hub_score,
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'progress': self.current_step / self.max_steps,
            'no_improve_steps': self.no_improve_steps,
            'num_nodes': current_metrics['num_nodes'],
            'num_edges': current_metrics['num_edges'],
            'graph_connected': bool(current_metrics['connected']),
            'hub_lost': self.hub_tracker.hub_lost if self.hub_tracker else False,
            'nodes_tracked': len(self.hub_tracker.node_id_mapping) if self.hub_tracker else 0,
            'discriminator_available': hasattr(self, 'discriminator') and self.discriminator is not None,
            'disc_start': self.disc_start
        }

        if self.hub_tracker and not self.hub_tracker.hub_lost:
            hub_info = self.get_hub_info()
            stats.update({
                'hub_fan_in': hub_info.get('hub_fan_in', 0),
                'hub_fan_out': hub_info.get('hub_fan_out', 0),
                'hub_total_degree': hub_info.get('hub_fan_in', 0) + hub_info.get('hub_fan_out', 0)
            })

        return stats

    def close(self):
        """UNCHANGED: Cleanup resources"""
        if hasattr(self, 'hub_tracker') and self.hub_tracker:
            self.hub_tracker.node_id_mapping.clear()
            self.hub_tracker.reverse_id_mapping.clear()

        self.current_data = None
        self.original_data_list = None

        print("Environment closed and resources freed")


# MAINTAINED: Original testing and helper classes
class RefactorEnvTester:
    """UNCHANGED: Testing and debugging helper"""

    def __init__(self, env: RefactorEnv):
        self.env = env
        self.test_results = []

    def test_basic_functionality(self) -> Dict:
        """Test basic environment functionality"""
        results = {
            'reset_test': False,
            'step_test': False,
            'state_consistency': False,
            'hub_tracking': False,
            'action_validity': False
        }

        try:
            # Test reset - CORRECTED: expects Data object
            initial_state = self.env.reset()
            results['reset_test'] = isinstance(initial_state, Data)

            # Test step - CORRECTED: expects Data object as next_state
            next_state, reward, done, info = self.env.step(6)  # STOP action
            results['step_test'] = isinstance(next_state, Data) and done

            # Test hub tracking
            hub_info = self.env.get_hub_info()
            results['hub_tracking'] = (hub_info is not None and
                                     'current_hub_index' in hub_info)

            # Test action mask
            action_mask = self.env.get_action_mask()
            results['action_validity'] = (len(action_mask) == self.env.num_actions and
                                        action_mask[6])  # STOP always valid

        except Exception as e:
            print(f"Error during basic tests: {e}")

        return results

    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        basic_results = self.test_basic_functionality()

        report = []
        report.append("=" * 60)
        report.append("PPO-CORRECTED REFACTOR ENVIRONMENT TEST REPORT")
        report.append("=" * 60)

        report.append("\nBASIC FUNCTIONALITY TESTS:")
        for test_name, passed in basic_results.items():
            status = "PASS" if passed else "FAIL"
            report.append(f"   {test_name}: {status}")

        # Environment info
        stats = self.env.get_performance_stats()
        report.append("\nENVIRONMENT STATUS:")
        report.append(f"   Dataset size: {len(self.env.original_data_list)} graphs")
        report.append(f"   Max nodes: {self.env.max_nodes}")
        report.append(f"   Action space: {self.env.num_actions} actions")
        report.append(f"   PPO Compatible: reset() and step() return Data objects")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


# CORRECTED: PPO-compatible wrapper
class PPORefactorEnv(RefactorEnv):
    """
    PPO-specific wrapper that ensures full compatibility
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("PPO-compatible RefactorEnv initialized")

    def reset(self) -> Data:
        """Ensure reset returns Data object"""
        return super().reset()

    def step(self, action: int) -> Tuple[Data, float, bool, Dict]:
        """Ensure step returns Data object as next_state"""
        return super().step(action)


# CORRECTED: Example usage
def example_usage():
    """Demonstrate corrected PPO-compatible usage"""
    print("Demonstrating PPO-corrected RefactorEnv usage")

    # This would be the actual usage in PPO trainer:
    # env = PPORefactorEnv(
    #     data_path="/path/to/your/data",
    #     max_steps=20,
    #     reward_weights={
    #         'hub_weight': 10.0,
    #         'step_invalid': -0.1,
    #         'time_penalty': -0.02,
    #         'adversarial_weight': 2.0,
    #         'patience': 15
    #     }
    # )
    #
    # # PPO rollout collection:
    # current_data = env.reset()  # Returns Data object
    # assert isinstance(current_data, Data)
    #
    # for step in range(10):
    #     # Extract global features for model
    #     global_features = env.get_global_features()
    #     assert global_features.shape[1] == 4
    #
    #     # Get action from PPO model (not shown)
    #     action = 0  # Example action
    #
    #     # Take step
    #     next_data, reward, done, info = env.step(action)
    #     assert isinstance(next_data, Data)
    #
    #     print(f"Step {step}: reward={reward:.3f}, done={done}")
    #
    #     if done:
    #         print(f"Episode ended. Hub improvement: {info['hub_improvement_total']:.4f}")
    #         break
    #
    #     current_data = next_data
    #
    # env.close()

    pass


if __name__ == "__main__":
    example_usage()