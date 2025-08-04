#!/usr/bin/env python3
"""
Fixed RL Environment for Hub Refactoring with Enhanced Error Handling and 7-Feature Compatibility
"""

import copy
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

import logging

logger = logging.getLogger(__name__)


@dataclass
class RefactoringResult:
    """Result of applying a refactoring pattern"""
    success: bool
    new_data: Optional[Data]
    info: Dict[str, Any]


class HubRefactoringEnv:
    """Fixed environment for hub refactoring with 7-feature compatibility"""

    def __init__(self, initial_data: Data, discriminator: nn.Module,
                 max_steps: int = 20, device: torch.device = torch.device('cpu'),
                 lazy_feature_update: bool = True):
        self.device = device
        self.discriminator = discriminator
        self.max_steps = max_steps
        self.lazy_feature_update = lazy_feature_update

        # Store initial graph
        self.initial_data = initial_data.to(device)
        self.current_data = None
        self.steps = 0

        # Caching for expensive computations
        self._cached_hub_score = None
        self._graph_hash = None

        # Reward parameters
        self.REWARD_SUCCESS = 15.0
        self.REWARD_PARTIAL_SUCCESS = 5.0
        self.REWARD_FAILURE = -3.0
        self.REWARD_STEP = -0.1
        self.REWARD_HUB_REDUCTION = 8.0
        self.REWARD_INVALID = -2.0

        # Tracking
        self.action_history = []
        self.initial_hub_score = None

    def reset(self) -> Data:
        """Reset environment"""
        self.current_data = copy.deepcopy(self.initial_data)
        self.steps = 0
        self.action_history.clear()

        # Reset cache
        self._cached_hub_score = None
        self._graph_hash = None

        # Compute initial hub score
        with torch.no_grad():
            try:
                disc_out = self.discriminator(self.current_data)
                self.initial_hub_score = F.softmax(disc_out['logits'], dim=1)[0, 1].item()
            except Exception as e:
                logger.warning(f"Failed to compute initial hub score: {e}")
                self.initial_hub_score = 0.5  # Fallback

        return self.current_data

    def _get_graph_hash(self, data: Data) -> str:
        """Generate hash for graph structure to detect changes"""
        try:
            edge_hash = hash(tuple(data.edge_index.flatten().tolist()))
            node_hash = hash(tuple(data.x.flatten().tolist()))
            return f"{edge_hash}_{node_hash}"
        except Exception as e:
            logger.warning(f"Failed to compute graph hash: {e}")
            return str(hash(str(data)))

    def _get_current_hub_score(self) -> float:
        """Get current hub-like score from discriminator with caching"""
        try:
            current_hash = self._get_graph_hash(self.current_data)

            # Return cached score if graph hasn't changed
            if self._cached_hub_score is not None and self._graph_hash == current_hash:
                return self._cached_hub_score

            # Compute new score
            with torch.no_grad():
                disc_out = self.discriminator(self.current_data)
                score = F.softmax(disc_out['logits'], dim=1)[0, 1].item()

            # Cache the result
            self._cached_hub_score = score
            self._graph_hash = current_hash

            return score
        except Exception as e:
            logger.warning(f"Failed to compute hub score: {e}")
            return 0.5  # Fallback to neutral score

    def _update_graph_features_7_unified(self, data: Data) -> Data:
        """Update graph features for 7-feature unified format"""
        try:
            # Convert to NetworkX for metric computation
            G = to_networkx(data, to_undirected=False)

            # Recompute the 7 unified structural features
            in_degrees = dict(G.in_degree())
            out_degrees = dict(G.out_degree())

            # Recompute centrality metrics efficiently
            try:
                if len(G) > 2 and len(G) <= 100:  # Only for reasonable sizes
                    pagerank = nx.pagerank(G, alpha=0.85, max_iter=50)
                    betweenness = nx.betweenness_centrality(G) if len(G) < 50 else {n: 0.0 for n in G.nodes()}
                    closeness = nx.closeness_centrality(G) if len(G) < 50 else {n: 1.0 for n in G.nodes()}
                else:
                    # Fallback for disconnected or very large graphs
                    pagerank = {n: 1.0 / max(len(G), 1) for n in G.nodes()}
                    betweenness = {n: 0.0 for n in G.nodes()}
                    closeness = {n: 1.0 for n in G.nodes()}
            except:
                # Ultimate fallback
                pagerank = {n: 1.0 / max(len(G), 1) for n in G.nodes()}
                betweenness = {n: 0.0 for n in G.nodes()}
                closeness = {n: 1.0 for n in G.nodes()}

            # Update node features - EXACTLY 7 features as in normalizer
            num_nodes = data.x.size(0)
            new_features = torch.zeros(num_nodes, 7, device=data.x.device, dtype=torch.float32)

            for i in range(num_nodes):
                if i in G.nodes():
                    # Extract the 7 unified structural features
                    fan_in = float(in_degrees.get(i, 0))
                    fan_out = float(out_degrees.get(i, 0))
                    total_degree = fan_in + fan_out

                    # Compute exactly the same 7 features as in EnhancedStructuralNormalizer
                    eps = 1e-8
                    new_features[i, 0] = fan_in  # fan_in
                    new_features[i, 1] = fan_out  # fan_out
                    new_features[i, 2] = total_degree / (len(G) - 1 + eps)  # degree_centrality
                    new_features[i, 3] = fan_in / (fan_out + eps)  # in_out_ratio
                    new_features[i, 4] = float(pagerank.get(i, 0))  # pagerank
                    new_features[i, 5] = float(betweenness.get(i, 0))  # betweenness_centrality
                    new_features[i, 6] = float(closeness.get(i, 0))  # closeness_centrality
                else:
                    # Node was removed, keep minimal features
                    new_features[i] = torch.zeros(7, device=data.x.device)

            data.x = new_features
            return data

        except Exception as e:
            logger.warning(f"Failed to update graph features: {e}")
            # Return original data if update fails
            return data

    def step(self, action: Tuple[int, int, int, bool]) -> Tuple[Data, float, bool, Dict]:
        """Execute refactoring action with robust error handling"""
        source, target, pattern, terminate = action
        self.steps += 1

        reward = self.REWARD_STEP
        done = False
        info = {'valid': False, 'pattern': pattern, 'success': False}

        try:
            # Check termination
            if terminate or self.steps >= self.max_steps:
                done = True
                final_reward = self._evaluate_final_state()
                reward += final_reward
                info['termination'] = 'requested' if terminate else 'max_steps'
                return self.current_data, reward, done, info

            # Validate action
            if source >= self.current_data.x.size(0) or target >= self.current_data.x.size(0):
                reward += self.REWARD_INVALID
                info['error'] = 'invalid_node_index'
                return self.current_data, reward, done, info

            # Apply refactoring pattern
            old_hub_score = self._get_current_hub_score()
            success, new_data, pattern_info = self._apply_pattern(pattern, source, target)

            if success and new_data is not None:
                # Update graph features after modification with 7-feature format
                if self.lazy_feature_update:
                    # Only update if the structural change is significant
                    structural_change = (new_data.edge_index.size(1) != self.current_data.edge_index.size(1) or
                                         new_data.x.size(0) != self.current_data.x.size(0))
                    if structural_change:
                        self.current_data = self._update_graph_features_7_unified(new_data)
                    else:
                        self.current_data = new_data
                else:
                    self.current_data = self._update_graph_features_7_unified(new_data)

                # Invalidate cache
                self._cached_hub_score = None
                self._graph_hash = None

                new_hub_score = self._get_current_hub_score()

                # Calculate improvement
                hub_improvement = old_hub_score - new_hub_score

                if hub_improvement > 0.1:
                    reward += self.REWARD_HUB_REDUCTION * hub_improvement
                elif hub_improvement > 0:
                    reward += self.REWARD_PARTIAL_SUCCESS * hub_improvement

                # Record action
                self.action_history.append((source, target, pattern, hub_improvement))

                info.update({
                    'valid': True,
                    'success': True,
                    'hub_improvement': hub_improvement,
                    'pattern_info': pattern_info
                })
            else:
                reward += self.REWARD_INVALID
                info['error'] = pattern_info.get('error', 'pattern_failed')

        except Exception as e:
            logger.warning(f"Error in step: {e}")
            reward += self.REWARD_INVALID
            info['error'] = f'step_error: {str(e)}'

        return self.current_data, reward, done, info

    def _apply_pattern(self, pattern: int, source: int, target: int) -> Tuple[bool, Optional[Data], Dict]:
        """Apply specific refactoring pattern with error handling"""
        patterns = {
            0: self._extract_interface,
            1: self._dependency_injection,
            2: self._split_by_responsibility,
            3: self._observer_pattern,
            4: self._strategy_pattern,
            5: self._remove_middleman
        }

        if pattern not in patterns:
            return False, None, {'error': 'unknown_pattern'}

        try:
            return patterns[pattern](source, target)
        except Exception as e:
            logger.warning(f"Pattern {pattern} failed: {e}")
            return False, None, {'error': f'pattern_exception: {str(e)}'}

    def _extract_interface(self, hub: int, client: int) -> Tuple[bool, Optional[Data], Dict]:
        """Extract interface to decouple hub from clients - 7-feature compatible"""
        try:
            data = copy.deepcopy(self.current_data)

            # Check if edge exists
            edge_mask = (data.edge_index[0] == hub) & (data.edge_index[1] == client)
            if not edge_mask.any():
                return False, None, {'error': 'no_edge_to_decouple'}

            # Add interface node with 7 features
            n_nodes = data.x.size(0)
            interface_features = (data.x[hub] + data.x[client]) / 2.0

            # Ensure exactly 7 features for interface
            if interface_features.size(0) != 7:
                interface_features = torch.zeros(7, device=data.x.device, dtype=torch.float32)
                # Set reasonable defaults for interface node
                interface_features[0] = 1.0  # fan_in
                interface_features[1] = 2.0  # fan_out
                interface_features[2] = 0.1  # degree_centrality
                interface_features[3] = 0.5  # in_out_ratio
                interface_features[4] = 0.1  # pagerank
                interface_features[5] = 0.0  # betweenness_centrality
                interface_features[6] = 0.5  # closeness_centrality

            data.x = torch.cat([data.x, interface_features.unsqueeze(0)], dim=0)

            # Redirect edge through interface
            edge_mask = ~((data.edge_index[0] == hub) & (data.edge_index[1] == client))
            new_edges = torch.tensor([[hub, n_nodes], [n_nodes, client]], device=self.device).t()
            data.edge_index = torch.cat([data.edge_index[:, edge_mask], new_edges], dim=1)

            return True, data, {'interface_node': n_nodes, 'decoupled': (hub, client)}
        except Exception as e:
            return False, None, {'error': f'extract_interface_failed: {str(e)}'}

    def _dependency_injection(self, dependent: int, dependency: int) -> Tuple[bool, Optional[Data], Dict]:
        """Apply dependency injection to reduce coupling"""
        try:
            data = copy.deepcopy(self.current_data)

            edge_mask = (data.edge_index[0] == dependent) & (data.edge_index[1] == dependency)
            if not edge_mask.any():
                return False, None, {'error': 'no_dependency_to_inject'}

            # Remove direct dependency (simulating injection)
            edge_mask = ~edge_mask
            data.edge_index = data.edge_index[:, edge_mask]

            # Reduce coupling metrics in node features (fan_out and fan_in)
            if data.x.size(1) >= 2:
                data.x[dependent, 1] = max(0, data.x[dependent, 1] - 1)  # Reduce fan_out
                data.x[dependency, 0] = max(0, data.x[dependency, 0] - 1)  # Reduce fan_in

            return True, data, {'injected_dependency': (dependent, dependency)}
        except Exception as e:
            return False, None, {'error': f'dependency_injection_failed: {str(e)}'}

    def _split_by_responsibility(self, large_node: int, _: int) -> Tuple[bool, Optional[Data], Dict]:
        """Split node by responsibilities - 7-feature compatible"""
        try:
            data = copy.deepcopy(self.current_data)

            # Check if node has enough connections to split
            out_mask = data.edge_index[0] == large_node
            in_mask = data.edge_index[1] == large_node

            out_degree = out_mask.sum().item()
            in_degree = in_mask.sum().item()

            if out_degree + in_degree < 4:
                return False, None, {'error': 'insufficient_connections_to_split'}

            # Add two responsibility nodes with 7 features each
            n_nodes = data.x.size(0)

            # Create features for split nodes (reduced responsibility)
            resp1_features = data.x[large_node] * 0.6
            resp2_features = data.x[large_node] * 0.4

            # Ensure exactly 7 features
            if resp1_features.size(0) != 7:
                resp1_features = torch.zeros(7, device=data.x.device, dtype=torch.float32)
                resp2_features = torch.zeros(7, device=data.x.device, dtype=torch.float32)

                # Copy original features if available
                orig_features = data.x[large_node]
                copy_size = min(7, orig_features.size(0))
                resp1_features[:copy_size] = orig_features[:copy_size] * 0.6
                resp2_features[:copy_size] = orig_features[:copy_size] * 0.4

            # Reduce fan_out for split nodes
            if resp1_features.size(0) >= 2:
                resp1_features[1] = resp1_features[1] * 0.5  # Reduce fan_out
                resp2_features[1] = resp2_features[1] * 0.5

            data.x = torch.cat([data.x, resp1_features.unsqueeze(0), resp2_features.unsqueeze(0)], dim=0)

            # Redistribute some outgoing edges
            out_edges = data.edge_index[:, out_mask]

            if out_edges.size(1) > 2:
                # Split outgoing edges between new nodes
                mid = out_edges.size(1) // 2

                # Create new edges to responsibility nodes
                edges_to_resp1 = torch.stack([
                    torch.full((mid,), n_nodes, device=self.device),
                    out_edges[1, :mid]
                ])
                edges_to_resp2 = torch.stack([
                    torch.full((out_edges.size(1) - mid,), n_nodes + 1, device=self.device),
                    out_edges[1, mid:]
                ])

                # Keep original edges and add coordinator edges
                keep_mask = ~out_mask
                coord_edges = torch.tensor([[large_node, large_node], [n_nodes, n_nodes + 1]], device=self.device)

                new_edges = torch.cat([edges_to_resp1, edges_to_resp2], dim=1)
                data.edge_index = torch.cat([data.edge_index[:, keep_mask], coord_edges, new_edges], dim=1)

            return True, data, {'split_nodes': [n_nodes, n_nodes + 1], 'original': large_node}
        except Exception as e:
            return False, None, {'error': f'split_responsibility_failed: {str(e)}'}

    def _observer_pattern(self, subject: int, observer: int) -> Tuple[bool, Optional[Data], Dict]:
        """Apply observer pattern to reduce direct coupling - 7-feature compatible"""
        try:
            data = copy.deepcopy(self.current_data)

            # Find all observers of the subject
            observer_mask = data.edge_index[0] == subject
            observers = data.edge_index[1, observer_mask]

            if len(observers) < 2:
                return False, None, {'error': 'insufficient_observers'}

            # Add notification hub (lightweight coordinator) with 7 features
            n_nodes = data.x.size(0)

            # Create lightweight notification features (7 features)
            notif_features = torch.zeros(7, device=data.x.device, dtype=torch.float32)
            notif_features[0] = 1.0  # fan_in
            notif_features[1] = float(len(observers))  # fan_out = number of observers
            notif_features[2] = 0.1  # degree_centrality
            notif_features[3] = notif_features[0] / (notif_features[1] + 1e-8)  # in_out_ratio
            notif_features[4] = 0.05  # pagerank
            notif_features[5] = 0.0  # betweenness_centrality
            notif_features[6] = 0.5  # closeness_centrality

            data.x = torch.cat([data.x, notif_features.unsqueeze(0)], dim=0)

            # Replace direct edges with observer pattern
            keep_mask = ~observer_mask
            data.edge_index = data.edge_index[:, keep_mask]

            # Add new pattern edges
            subj_to_notif = torch.tensor([[subject], [n_nodes]], device=self.device)
            notif_to_obs = torch.stack([torch.full_like(observers, n_nodes), observers])

            new_edges = torch.cat([subj_to_notif, notif_to_obs], dim=1)
            data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)

            return True, data, {'notification_hub': n_nodes, 'observers': observers.tolist()}
        except Exception as e:
            return False, None, {'error': f'observer_pattern_failed: {str(e)}'}

    def _strategy_pattern(self, context: int, strategy: int) -> Tuple[bool, Optional[Data], Dict]:
        """Extract strategy pattern to reduce coupling - 7-feature compatible"""
        try:
            data = copy.deepcopy(self.current_data)

            # Check if context has multiple outgoing connections
            out_mask = data.edge_index[0] == context
            strategies = data.edge_index[1, out_mask]

            if len(strategies) < 2:
                return False, None, {'error': 'insufficient_strategies'}

            # Add strategy interface
            n_nodes = data.x.size(0)

            # Create interface features (7 features)
            interface_features = torch.zeros(7, device=data.x.device, dtype=torch.float32)

            # Average of strategies features if available
            if len(strategies) > 0 and data.x.size(0) > max(strategies):
                valid_strategies = [s for s in strategies if s < data.x.size(0)]
                if valid_strategies:
                    strategy_features = data.x[valid_strategies]
                    interface_features = strategy_features.mean(dim=0)
                    # Ensure exactly 7 features
                    if interface_features.size(0) > 7:
                        interface_features = interface_features[:7]
                    elif interface_features.size(0) < 7:
                        padding = torch.zeros(7 - interface_features.size(0), device=data.x.device)
                        interface_features = torch.cat([interface_features, padding])

            # Set reasonable defaults if needed
            interface_features[0] = max(interface_features[0], 1.0)  # fan_in
            interface_features[1] = max(interface_features[1], 1.0)  # fan_out

            data.x = torch.cat([data.x, interface_features.unsqueeze(0)], dim=0)

            # Redirect through strategy interface
            keep_mask = ~out_mask
            data.edge_index = data.edge_index[:, keep_mask]

            # Add context -> interface and interface -> strategies
            ctx_to_iface = torch.tensor([[context], [n_nodes]], device=self.device)
            iface_to_strat = torch.stack([torch.full_like(strategies, n_nodes), strategies])

            new_edges = torch.cat([ctx_to_iface, iface_to_strat], dim=1)
            data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)

            return True, data, {'strategy_interface': n_nodes, 'strategies': strategies.tolist()}
        except Exception as e:
            return False, None, {'error': f'strategy_pattern_failed: {str(e)}'}

    def _remove_middleman(self, middleman: int, _: int) -> Tuple[bool, Optional[Data], Dict]:
        """Remove unnecessary middleman to reduce indirection"""
        try:
            data = copy.deepcopy(self.current_data)

            # Get predecessors and successors
            in_mask = data.edge_index[1] == middleman
            out_mask = data.edge_index[0] == middleman

            predecessors = data.edge_index[0, in_mask]
            successors = data.edge_index[1, out_mask]

            if len(predecessors) == 0 or len(successors) == 0:
                return False, None, {'error': 'not_a_middleman'}

            # Remove middleman edges
            remove_mask = in_mask | out_mask
            data.edge_index = data.edge_index[:, ~remove_mask]

            # Add direct connections (bypassing middleman)
            direct_edges = []
            for pred in predecessors:
                for succ in successors:
                    pred_val = pred.item() if torch.is_tensor(pred) else pred
                    succ_val = succ.item() if torch.is_tensor(succ) else succ
                    if pred_val != succ_val:  # Avoid self-loops
                        direct_edges.append([pred_val, succ_val])

            if direct_edges:
                direct_edges = torch.tensor(direct_edges, device=self.device).t()
                data.edge_index = torch.cat([data.edge_index, direct_edges], dim=1)

            return True, data, {'removed_middleman': middleman,
                                'direct_connections': len(direct_edges[0]) if direct_edges else 0}
        except Exception as e:
            return False, None, {'error': f'remove_middleman_failed: {str(e)}'}

    def _evaluate_final_state(self) -> float:
        """Evaluate final state for terminal reward"""
        try:
            if self.initial_hub_score is None:
                return 0.0

            final_hub_score = self._get_current_hub_score()
            improvement = self.initial_hub_score - final_hub_score

            # Success criteria
            if final_hub_score < 0.3:  # Strong success
                reward = self.REWARD_SUCCESS * (1.0 + improvement)
            elif improvement > 0.15:  # Good improvement
                reward = self.REWARD_PARTIAL_SUCCESS * (1.0 + improvement)
            elif improvement > 0.05:  # Slight improvement
                reward = self.REWARD_PARTIAL_SUCCESS * 0.5
            else:  # No improvement or worse
                reward = self.REWARD_FAILURE

            # Efficiency bonus
            efficiency = 1.0 - (self.steps / self.max_steps)
            reward += efficiency * 3.0

            return reward
        except Exception as e:
            logger.warning(f"Failed to evaluate final state: {e}")
            return 0.0