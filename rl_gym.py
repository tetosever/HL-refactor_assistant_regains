#!/usr/bin/env python3
"""
Updated RL Environment with Centralized Hyperparameter Configuration

Key changes:
- Uses centralized reward configuration from hyperparameters_configuration.py
- All reward parameters loaded from config file
- Incremental feature updates instead of full reconstruction
- Direct computation of affected node features
- No dependency on UnifiedDataLoader for intermediate graphs
- Eliminates the "Failed to recompute features" warning
"""

import copy
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List, Set
import math

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

import logging

# Import centralized configuration
from hyperparameters_configuration import get_improved_config

logger = logging.getLogger(__name__)


@dataclass
class RefactoringResult:
    """Result of applying a refactoring pattern"""
    success: bool
    new_data: Optional[Data]
    info: Dict[str, Any]


class IncrementalFeatureComputer:
    """
    Incremental feature computation that updates only affected nodes
    Uses the same algorithms as UnifiedFeatureComputer but more efficiently
    """

    @staticmethod
    def compute_single_node_features(node_id: int, G: nx.Graph,
                                     global_centrality: Optional[Dict] = None) -> List[float]:
        """
        Compute the 7 structural features for a single node

        Args:
            node_id: ID of the node to compute features for
            G: NetworkX graph
            global_centrality: Pre-computed global centrality measures (optional)

        Returns:
            List of 7 feature values
        """
        if node_id not in G:
            return [0.0] * 7

        try:
            # Basic degree metrics
            fan_in = float(G.in_degree(node_id))
            fan_out = float(G.out_degree(node_id))
            total_degree = fan_in + fan_out

            num_nodes = len(G)
            eps = 1e-8

            # Degree centrality
            degree_centrality = total_degree / (num_nodes - 1 + eps)

            # In-out ratio
            in_out_ratio = fan_in / (fan_out + eps)

            # Get global centrality measures
            if global_centrality is None:
                global_centrality = IncrementalFeatureComputer._compute_global_centrality(G)

            pagerank = float(global_centrality['pagerank'].get(node_id, 0))
            betweenness = float(global_centrality['betweenness'].get(node_id, 0))
            closeness = float(global_centrality['closeness'].get(node_id, 0))

            return [fan_in, fan_out, degree_centrality, in_out_ratio,
                    pagerank, betweenness, closeness]

        except Exception as e:
            logger.warning(f"Failed to compute features for node {node_id}: {e}")
            return [0.0] * 7

    @staticmethod
    def _compute_global_centrality(G: nx.Graph) -> Dict[str, Dict]:
        """Compute global centrality measures efficiently"""
        try:
            num_nodes = len(G)

            if num_nodes <= 100:  # Same threshold as UnifiedFeatureComputer
                pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-4)
                betweenness = nx.betweenness_centrality(G, normalized=True)
                closeness = nx.closeness_centrality(G, distance=None, wf_improved=True)
            else:
                # Simplified computation for large graphs
                total_edges = G.number_of_edges()
                pagerank = {n: float(G.degree(n)) / (2 * total_edges + 1e-8) for n in G.nodes()}
                betweenness = {n: 0.0 for n in G.nodes()}
                closeness = {n: 1.0 / (num_nodes - 1 + 1e-8) for n in G.nodes()}

            return {
                'pagerank': pagerank,
                'betweenness': betweenness,
                'closeness': closeness
            }
        except Exception as e:
            logger.warning(f"Failed to compute global centrality: {e}")
            # Return empty dicts with fallback
            empty_dict = {n: 0.0 for n in G.nodes()}
            return {
                'pagerank': empty_dict,
                'betweenness': empty_dict,
                'closeness': empty_dict
            }

    @staticmethod
    def get_affected_nodes(modification_type: str, source_node: int, target_node: int,
                           G: nx.Graph, radius: int = 2) -> Set[int]:
        """
        Determine which nodes need feature updates based on modification type

        Args:
            modification_type: Type of modification applied
            source_node: Source node of modification
            target_node: Target node of modification
            G: Current graph state
            radius: Radius for neighborhood updates

        Returns:
            Set of node IDs that need feature updates
        """
        affected = {source_node, target_node}

        # For most centrality measures, we need to update neighbors too
        try:
            # Add direct neighbors of modified nodes
            if source_node in G:
                predecessors = set(G.predecessors(source_node))
                successors = set(G.successors(source_node))
                affected.update(predecessors)
                affected.update(successors)

            if target_node in G and target_node != source_node:
                predecessors = set(G.predecessors(target_node))
                successors = set(G.successors(target_node))
                affected.update(predecessors)
                affected.update(successors)

            # For certain modifications, expand the radius
            if modification_type in ['node_addition', 'node_removal', 'major_restructuring']:
                # Expand to 2-hop neighborhood for significant changes
                expanded = set(affected)
                for node in list(affected):
                    if node in G:
                        expanded.update(set(G.neighbors(node)))
                affected = expanded

        except Exception as e:
            logger.warning(f"Error determining affected nodes: {e}")

        return affected

    @staticmethod
    def update_features_incremental(data: Data, affected_nodes: Set[int],
                                    modification_info: Dict = None) -> Data:
        """
        Update features for affected nodes incrementally

        Args:
            data: PyG Data object to update
            affected_nodes: Set of node indices to update
            modification_info: Information about the modification applied

        Returns:
            Updated Data object
        """
        try:
            # Convert to NetworkX for feature computation
            G = to_networkx(data, to_undirected=False)

            # Filter affected nodes to only those that exist in the graph
            valid_affected = {node for node in affected_nodes if node < data.x.size(0) and node in G}

            if not valid_affected:
                logger.debug("No valid affected nodes to update")
                return data

            # Compute global centrality once for efficiency
            global_centrality = IncrementalFeatureComputer._compute_global_centrality(G)

            # Update features for each affected node
            for node_idx in valid_affected:
                try:
                    new_features = IncrementalFeatureComputer.compute_single_node_features(
                        node_idx, G, global_centrality
                    )

                    # Update the tensor in-place
                    data.x[node_idx] = torch.tensor(new_features, dtype=torch.float32, device=data.x.device)

                except Exception as e:
                    logger.warning(f"Failed to update features for node {node_idx}: {e}")
                    continue

            logger.debug(f"Successfully updated features for {len(valid_affected)} nodes")
            return data

        except Exception as e:
            logger.error(f"Incremental feature update failed: {e}")
            return data  # Return original data if update fails


class HubRefactoringEnv:
    """RL Environment with centralized hyperparameter configuration and incremental feature computation"""

    def __init__(self, initial_data: Data, discriminator: nn.Module,
                 max_steps: int = 20, device: torch.device = torch.device('cpu'),
                 reward_config=None):
        self.device = device
        self.discriminator = discriminator
        self.max_steps = max_steps

        # Store initial graph
        self.initial_data = initial_data.to(device)
        self.current_data = None
        self.steps = 0

        # Caching for expensive computations
        self._cached_hub_score = None
        self._graph_hash = None

        # Load centralized reward configuration
        if reward_config is None:
            config = get_improved_config()
            reward_config = config['rewards']
            logger.debug("Loaded centralized reward configuration")

        self.reward_config = reward_config

        # CENTRALIZED REWARD PARAMETERS - loaded from config
        self.REWARD_SUCCESS = reward_config.REWARD_SUCCESS
        self.REWARD_PARTIAL_SUCCESS = reward_config.REWARD_PARTIAL_SUCCESS
        self.REWARD_FAILURE = reward_config.REWARD_FAILURE
        self.REWARD_STEP = reward_config.REWARD_STEP
        self.REWARD_HUB_REDUCTION = reward_config.REWARD_HUB_REDUCTION
        self.REWARD_INVALID = reward_config.REWARD_INVALID
        self.REWARD_SMALL_IMPROVEMENT = reward_config.REWARD_SMALL_IMPROVEMENT

        # Hub score thresholds from config
        self.HUB_SCORE_EXCELLENT = reward_config.HUB_SCORE_EXCELLENT
        self.HUB_SCORE_GOOD = reward_config.HUB_SCORE_GOOD
        self.HUB_SCORE_ACCEPTABLE = reward_config.HUB_SCORE_ACCEPTABLE

        # Progressive reward scaling from config
        self.improvement_multiplier_boost = reward_config.improvement_multiplier_boost
        self.improvement_multiplier_decay = reward_config.improvement_multiplier_decay
        self.improvement_multiplier_max = reward_config.improvement_multiplier_max
        self.improvement_multiplier_min = reward_config.improvement_multiplier_min
        self.improvement_multiplier = 1.0

        # Tracking
        self.action_history = []
        self.initial_hub_score = None
        self.best_hub_score = None
        self.hub_score_history = []

        logger.debug(f"Environment initialized with centralized config:")
        logger.debug(f"  REWARD_SUCCESS: {self.REWARD_SUCCESS}")
        logger.debug(f"  REWARD_SMALL_IMPROVEMENT: {self.REWARD_SMALL_IMPROVEMENT}")
        logger.debug(f"  HUB_SCORE_EXCELLENT: {self.HUB_SCORE_EXCELLENT}")

    def update_reward_config(self, new_reward_config):
        """Update reward configuration dynamically"""
        self.reward_config = new_reward_config

        # Update all reward parameters
        self.REWARD_SUCCESS = new_reward_config.REWARD_SUCCESS
        self.REWARD_PARTIAL_SUCCESS = new_reward_config.REWARD_PARTIAL_SUCCESS
        self.REWARD_FAILURE = new_reward_config.REWARD_FAILURE
        self.REWARD_STEP = new_reward_config.REWARD_STEP
        self.REWARD_HUB_REDUCTION = new_reward_config.REWARD_HUB_REDUCTION
        self.REWARD_INVALID = new_reward_config.REWARD_INVALID
        self.REWARD_SMALL_IMPROVEMENT = new_reward_config.REWARD_SMALL_IMPROVEMENT

        # Update thresholds
        self.HUB_SCORE_EXCELLENT = new_reward_config.HUB_SCORE_EXCELLENT
        self.HUB_SCORE_GOOD = new_reward_config.HUB_SCORE_GOOD
        self.HUB_SCORE_ACCEPTABLE = new_reward_config.HUB_SCORE_ACCEPTABLE

        # Update progressive scaling
        self.improvement_multiplier_boost = new_reward_config.improvement_multiplier_boost
        self.improvement_multiplier_decay = new_reward_config.improvement_multiplier_decay
        self.improvement_multiplier_max = new_reward_config.improvement_multiplier_max
        self.improvement_multiplier_min = new_reward_config.improvement_multiplier_min

        logger.debug("Reward configuration updated dynamically")

    def _ensure_data_consistency(self, data: Data) -> Data:
        """Ensure batch and edge_attr consistency"""
        num_nodes = data.x.size(0)
        num_edges = data.edge_index.size(1)

        if not hasattr(data, 'batch') or data.batch is None or data.batch.size(0) != num_nodes:
            data.batch = torch.zeros(num_nodes, dtype=torch.long, device=data.x.device)

        if not hasattr(data, 'edge_attr') or data.edge_attr is None or data.edge_attr.size(0) != num_edges:
            data.edge_attr = torch.ones(num_edges, 1, dtype=torch.float32, device=data.x.device)

        return data

    def reset(self) -> Data:
        """Reset environment with centralized config initialization"""
        self.current_data = copy.deepcopy(self.initial_data)
        self._ensure_data_consistency(self.current_data)
        self.steps = 0
        self.action_history.clear()
        self.hub_score_history.clear()

        # Reset cache
        self._cached_hub_score = None
        self._graph_hash = None

        # Compute initial hub score with better error handling
        initial_score = self._get_current_hub_score()
        self.initial_hub_score = initial_score
        self.best_hub_score = initial_score
        self.hub_score_history.append(initial_score)

        # Reset improvement multiplier
        self.improvement_multiplier = 1.0

        logger.debug(f"Environment reset - Initial hub score: {initial_score:.4f} (centralized config)")
        return self.current_data

    def _get_graph_hash(self, data: Data) -> str:
        """Generate hash for graph structure"""
        try:
            edge_hash = hash(tuple(data.edge_index.flatten().tolist()))
            node_hash = hash(tuple(data.x.flatten().tolist()))
            return f"{edge_hash}_{node_hash}"
        except Exception as e:
            logger.warning(f"Failed to compute graph hash: {e}")
            return str(hash(str(data)))

    def _get_current_hub_score(self) -> float:
        """Hub score computation with better error handling"""
        try:
            self._ensure_data_consistency(self.current_data)
            current_hash = self._get_graph_hash(self.current_data)

            # Return cached score if graph hasn't changed
            if self._cached_hub_score is not None and self._graph_hash == current_hash:
                return self._cached_hub_score

            # Compute new score with better error handling
            with torch.no_grad():
                # Ensure discriminator is in eval mode
                self.discriminator.eval()

                # Get discriminator prediction
                disc_out = self.discriminator(self.current_data)

                if disc_out is None or 'logits' not in disc_out:
                    logger.warning("Discriminator returned invalid output")
                    return 0.5  # Neutral fallback

                logits = disc_out['logits']
                if logits.dim() == 0 or logits.size(0) == 0:
                    logger.warning("Discriminator logits are empty")
                    return 0.5

                # Get probability of being "smelly" (hub-like)
                probs = F.softmax(logits, dim=1)
                if probs.size(1) < 2:
                    logger.warning("Discriminator output has insufficient classes")
                    return 0.5

                score = probs[0, 1].item()  # Probability of class 1 (smelly/hub-like)

            # Validate score
            if not (0.0 <= score <= 1.0) or math.isnan(score):
                logger.warning(f"Invalid hub score computed: {score}, using fallback")
                score = 0.5

            # Cache the result
            self._cached_hub_score = score
            self._graph_hash = current_hash

            return score

        except Exception as e:
            logger.warning(f"Failed to compute hub score: {e}")
            return 0.5  # Fallback to neutral score

    def step(self, action: Tuple[int, int, int, bool]) -> Tuple[Data, float, bool, Dict]:
        """Step with centralized reward calculation and hub tracking"""
        source, target, pattern, terminate = action
        self.steps += 1

        # Base step penalty from centralized config
        reward = self.REWARD_STEP
        done = False
        info = {'valid': False, 'pattern': pattern, 'success': False, 'hub_improvement': 0.0}

        try:
            # Check termination first
            if terminate or self.steps >= self.max_steps:
                done = True
                final_reward = self._evaluate_final_state()
                reward += final_reward
                info['termination'] = 'requested' if terminate else 'max_steps'
                info['final_reward'] = final_reward
                return self.current_data, reward, done, info

            # Validate action
            if source >= self.current_data.x.size(0) or target >= self.current_data.x.size(0):
                reward += self.REWARD_INVALID
                info['error'] = 'invalid_node_index'
                return self.current_data, reward, done, info

            # Get current hub score BEFORE modification
            old_hub_score = self._get_current_hub_score()

            # Apply refactoring pattern
            success, new_data, pattern_info = self._apply_pattern_incremental(pattern, source, target)

            if success and new_data is not None:
                # Update current data
                self.current_data = new_data

                # Invalidate cache after graph modification
                self._cached_hub_score = None
                self._graph_hash = None

                # Get NEW hub score AFTER modification
                new_hub_score = self._get_current_hub_score()

                # Calculate improvement (positive = better, negative = worse)
                hub_improvement = old_hub_score - new_hub_score

                # Track best score
                if new_hub_score < self.best_hub_score:
                    self.best_hub_score = new_hub_score
                    info['new_best'] = True
                    # Apply boost from centralized config
                    self.improvement_multiplier = min(
                        self.improvement_multiplier * self.improvement_multiplier_boost,
                        self.improvement_multiplier_max
                    )

                # Add to history
                self.hub_score_history.append(new_hub_score)

                # CENTRALIZED CONFIG REWARD CALCULATION
                if hub_improvement > 0.15:  # Significant improvement
                    reward += self.REWARD_HUB_REDUCTION * hub_improvement * self.improvement_multiplier
                    info['reward_type'] = 'significant_improvement'
                elif hub_improvement > 0.05:  # Good improvement
                    reward += self.REWARD_PARTIAL_SUCCESS * hub_improvement * self.improvement_multiplier
                    info['reward_type'] = 'good_improvement'
                elif hub_improvement > 0.01:  # Small improvement - uses centralized config value
                    reward += self.REWARD_SMALL_IMPROVEMENT * hub_improvement * self.improvement_multiplier
                    info['reward_type'] = 'small_improvement'
                elif hub_improvement > -0.02:  # Neutral (no significant change)
                    reward += 0.1  # Small positive reward for valid action
                    info['reward_type'] = 'neutral'
                else:  # Made it worse
                    reward += self.REWARD_FAILURE * abs(hub_improvement)
                    info['reward_type'] = 'worse'
                    # Apply decay from centralized config
                    self.improvement_multiplier = max(
                        self.improvement_multiplier * self.improvement_multiplier_decay,
                        self.improvement_multiplier_min
                    )

                # BONUS REWARDS based on centralized config thresholds
                if new_hub_score < self.HUB_SCORE_EXCELLENT:
                    reward += 5.0  # Excellent score bonus
                    info['score_bonus'] = 'excellent'
                elif new_hub_score < self.HUB_SCORE_GOOD:
                    reward += 2.0  # Good score bonus
                    info['score_bonus'] = 'good'
                elif new_hub_score < self.HUB_SCORE_ACCEPTABLE:
                    reward += 0.5  # Acceptable score bonus
                    info['score_bonus'] = 'acceptable'

                # Record action with improvement
                self.action_history.append((source, target, pattern, hub_improvement))

                info.update({
                    'valid': True,
                    'success': True,
                    'hub_improvement': hub_improvement,
                    'old_hub_score': old_hub_score,
                    'new_hub_score': new_hub_score,
                    'best_hub_score': self.best_hub_score,
                    'pattern_info': pattern_info,
                    'features_updated_incrementally': True,
                    'affected_nodes': pattern_info.get('affected_nodes', []),
                    'improvement_multiplier': self.improvement_multiplier,
                    'reward_config_used': 'centralized'
                })
            else:
                # Pattern failed - penalty from centralized config
                reward += self.REWARD_INVALID
                info['error'] = pattern_info.get('error', 'pattern_failed')
                info['reward_type'] = 'pattern_failed'

        except Exception as e:
            logger.warning(f"Error in step: {e}")
            reward += self.REWARD_INVALID
            info['error'] = f'step_error: {str(e)}'
            info['reward_type'] = 'error'

        return self.current_data, reward, done, info

    def _apply_pattern_incremental(self, pattern: int, source: int, target: int) -> Tuple[bool, Optional[Data], Dict]:
        """Apply refactoring pattern with better error handling"""
        patterns = {
            0: self._extract_interface_incremental,
            1: self._dependency_injection_incremental,
            2: self._split_by_responsibility_incremental,
            3: self._observer_pattern_incremental,
            4: self._strategy_pattern_incremental,
            5: self._remove_middleman_incremental
        }

        if pattern not in patterns:
            return False, None, {'error': 'unknown_pattern'}

        try:
            success, new_data, info = patterns[pattern](source, target)

            # Additional validation
            if success and new_data is not None:
                # Ensure the new data is valid
                if new_data.x.size(0) == 0 or new_data.edge_index.size(1) == 0:
                    logger.warning(f"Pattern {pattern} created empty graph")
                    return False, None, {'error': 'empty_graph_created'}

                # Ensure features are valid
                if torch.any(torch.isnan(new_data.x)) or torch.any(torch.isinf(new_data.x)):
                    logger.warning(f"Pattern {pattern} created invalid features")
                    return False, None, {'error': 'invalid_features_created'}

            return success, new_data, info

        except Exception as e:
            logger.warning(f"Pattern {pattern} failed with exception: {e}")
            return False, None, {'error': f'pattern_exception: {str(e)}'}

    def _extract_interface_incremental(self, hub: int, client: int) -> Tuple[bool, Optional[Data], Dict]:
        """Extract interface to decouple hub from clients with incremental updates"""
        try:
            data = copy.deepcopy(self.current_data)

            # Check if edge exists
            edge_mask = (data.edge_index[0] == hub) & (data.edge_index[1] == client)
            if not edge_mask.any():
                return False, None, {'error': 'no_edge_to_decouple'}

            # Add interface node
            n_nodes = data.x.size(0)

            # Create placeholder features for new node (will be computed incrementally)
            interface_features = torch.zeros(1, 7, device=data.x.device, dtype=torch.float32)
            data.x = torch.cat([data.x, interface_features], dim=0)

            # Update node mappings if they exist
            if hasattr(data, 'original_node_ids'):
                interface_id = f"interface_{n_nodes}"
                data.original_node_ids.append(interface_id)

                if hasattr(data, 'node_id_to_index'):
                    data.node_id_to_index[interface_id] = n_nodes
                if hasattr(data, 'index_to_node_id'):
                    data.index_to_node_id[n_nodes] = interface_id

            # Redirect edge through interface
            edge_mask = ~((data.edge_index[0] == hub) & (data.edge_index[1] == client))
            new_edges = torch.tensor([[hub, n_nodes], [n_nodes, client]], device=self.device).t()
            data.edge_index = torch.cat([data.edge_index[:, edge_mask], new_edges], dim=1)

            # Update edge attributes
            if hasattr(data, 'edge_attr'):
                data.edge_attr = data.edge_attr[edge_mask]
                new_edge_attr = torch.ones(2, 1, device=self.device, dtype=torch.float32)
                data.edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)

            # Incremental feature update
            affected_nodes = IncrementalFeatureComputer.get_affected_nodes(
                'interface_extraction', hub, client, to_networkx(data, to_undirected=False)
            )
            affected_nodes.add(n_nodes)  # Include the new interface node

            data = IncrementalFeatureComputer.update_features_incremental(
                data, affected_nodes, {'pattern': 'extract_interface', 'interface_node': n_nodes}
            )

            return True, data, {
                'interface_node': n_nodes,
                'decoupled': (hub, client),
                'affected_nodes': list(affected_nodes),
                'modification_type': 'interface_extraction'
            }
        except Exception as e:
            return False, None, {'error': f'extract_interface_failed: {str(e)}'}

    def _dependency_injection_incremental(self, dependent: int, dependency: int) -> Tuple[bool, Optional[Data], Dict]:
        """Apply dependency injection to reduce coupling with incremental updates"""
        try:
            data = copy.deepcopy(self.current_data)

            edge_mask = (data.edge_index[0] == dependent) & (data.edge_index[1] == dependency)
            if not edge_mask.any():
                return False, None, {'error': 'no_dependency_to_inject'}

            # Remove direct dependency edge
            keep_mask = ~edge_mask
            data.edge_index = data.edge_index[:, keep_mask]

            # Update edge attributes
            if hasattr(data, 'edge_attr'):
                data.edge_attr = data.edge_attr[keep_mask]

            # Incremental feature update
            affected_nodes = IncrementalFeatureComputer.get_affected_nodes(
                'dependency_injection', dependent, dependency, to_networkx(data, to_undirected=False)
            )

            data = IncrementalFeatureComputer.update_features_incremental(
                data, affected_nodes, {'pattern': 'dependency_injection'}
            )

            return True, data, {
                'injected_dependency': (dependent, dependency),
                'affected_nodes': list(affected_nodes),
                'modification_type': 'dependency_injection'
            }
        except Exception as e:
            return False, None, {'error': f'dependency_injection_failed: {str(e)}'}

    def _split_by_responsibility_incremental(self, large_node: int, _: int) -> Tuple[bool, Optional[Data], Dict]:
        """Split node by responsibilities with incremental updates"""
        try:
            data = copy.deepcopy(self.current_data)

            # Check if node has enough connections to split
            out_mask = data.edge_index[0] == large_node
            in_mask = data.edge_index[1] == large_node

            out_degree = out_mask.sum().item()
            in_degree = in_mask.sum().item()

            if out_degree + in_degree < 4:
                return False, None, {'error': 'insufficient_connections_to_split'}

            # Add two responsibility nodes
            n_nodes = data.x.size(0)

            # Create placeholder features (will be computed incrementally)
            resp_features = torch.zeros(2, 7, device=data.x.device, dtype=torch.float32)
            data.x = torch.cat([data.x, resp_features], dim=0)

            # Update node mappings
            if hasattr(data, 'original_node_ids'):
                resp1_id = f"resp1_{n_nodes}"
                resp2_id = f"resp2_{n_nodes + 1}"
                data.original_node_ids.extend([resp1_id, resp2_id])

                if hasattr(data, 'node_id_to_index'):
                    data.node_id_to_index[resp1_id] = n_nodes
                    data.node_id_to_index[resp2_id] = n_nodes + 1
                if hasattr(data, 'index_to_node_id'):
                    data.index_to_node_id[n_nodes] = resp1_id
                    data.index_to_node_id[n_nodes + 1] = resp2_id

            # Redistribute outgoing edges
            out_edges = data.edge_index[:, out_mask]

            if out_edges.size(1) > 2:
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

                # Update edge attributes
                if hasattr(data, 'edge_attr'):
                    data.edge_attr = data.edge_attr[keep_mask]
                    num_new_edges = coord_edges.size(1) + new_edges.size(1)
                    new_edge_attr = torch.ones(num_new_edges, 1, device=self.device, dtype=torch.float32)
                    data.edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)

            # Incremental feature update
            affected_nodes = IncrementalFeatureComputer.get_affected_nodes(
                'node_splitting', large_node, large_node, to_networkx(data, to_undirected=False)
            )
            affected_nodes.update([n_nodes, n_nodes + 1])  # Include new responsibility nodes

            data = IncrementalFeatureComputer.update_features_incremental(
                data, affected_nodes, {'pattern': 'split_responsibility', 'split_nodes': [n_nodes, n_nodes + 1]}
            )

            return True, data, {
                'split_nodes': [n_nodes, n_nodes + 1],
                'original': large_node,
                'affected_nodes': list(affected_nodes),
                'modification_type': 'node_splitting'
            }
        except Exception as e:
            return False, None, {'error': f'split_responsibility_failed: {str(e)}'}

    def _observer_pattern_incremental(self, subject: int, observer: int) -> Tuple[bool, Optional[Data], Dict]:
        """Apply observer pattern to reduce direct coupling with incremental updates"""
        try:
            data = copy.deepcopy(self.current_data)

            # Find all observers of the subject
            observer_mask = data.edge_index[0] == subject
            observers = data.edge_index[1, observer_mask]

            if len(observers) < 2:
                return False, None, {'error': 'insufficient_observers'}

            # Add notification hub
            n_nodes = data.x.size(0)

            # Create placeholder features (will be computed incrementally)
            notif_features = torch.zeros(1, 7, device=data.x.device, dtype=torch.float32)
            data.x = torch.cat([data.x, notif_features], dim=0)

            # Update node mappings
            if hasattr(data, 'original_node_ids'):
                notif_id = f"notif_hub_{n_nodes}"
                data.original_node_ids.append(notif_id)

                if hasattr(data, 'node_id_to_index'):
                    data.node_id_to_index[notif_id] = n_nodes
                if hasattr(data, 'index_to_node_id'):
                    data.index_to_node_id[n_nodes] = notif_id

            # Replace direct edges with observer pattern
            keep_mask = ~observer_mask
            data.edge_index = data.edge_index[:, keep_mask]

            # Add new pattern edges
            subj_to_notif = torch.tensor([[subject], [n_nodes]], device=self.device)
            notif_to_obs = torch.stack([torch.full_like(observers, n_nodes), observers])

            new_edges = torch.cat([subj_to_notif, notif_to_obs], dim=1)
            data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)

            # Update edge attributes
            if hasattr(data, 'edge_attr'):
                data.edge_attr = data.edge_attr[keep_mask]
                new_edge_attr = torch.ones(new_edges.size(1), 1, device=self.device, dtype=torch.float32)
                data.edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)

            # Incremental feature update
            affected_nodes = IncrementalFeatureComputer.get_affected_nodes(
                'observer_pattern', subject, subject, to_networkx(data, to_undirected=False)
            )
            affected_nodes.add(n_nodes)  # Include notification hub
            affected_nodes.update(observers.tolist())  # Include all observers

            data = IncrementalFeatureComputer.update_features_incremental(
                data, affected_nodes, {'pattern': 'observer', 'notification_hub': n_nodes}
            )

            return True, data, {
                'notification_hub': n_nodes,
                'observers': observers.tolist(),
                'affected_nodes': list(affected_nodes),
                'modification_type': 'observer_pattern'
            }
        except Exception as e:
            return False, None, {'error': f'observer_pattern_failed: {str(e)}'}

    def _strategy_pattern_incremental(self, context: int, strategy: int) -> Tuple[bool, Optional[Data], Dict]:
        """Extract strategy pattern to reduce coupling with incremental updates"""
        try:
            data = copy.deepcopy(self.current_data)

            # Check if context has multiple outgoing connections
            out_mask = data.edge_index[0] == context
            strategies = data.edge_index[1, out_mask]

            if len(strategies) < 2:
                return False, None, {'error': 'insufficient_strategies'}

            # Add strategy interface
            n_nodes = data.x.size(0)

            # Create placeholder features (will be computed incrementally)
            interface_features = torch.zeros(1, 7, device=data.x.device, dtype=torch.float32)
            data.x = torch.cat([data.x, interface_features], dim=0)

            # Update node mappings
            if hasattr(data, 'original_node_ids'):
                interface_id = f"strategy_iface_{n_nodes}"
                data.original_node_ids.append(interface_id)

                if hasattr(data, 'node_id_to_index'):
                    data.node_id_to_index[interface_id] = n_nodes
                if hasattr(data, 'index_to_node_id'):
                    data.index_to_node_id[n_nodes] = interface_id

            # Redirect through strategy interface
            keep_mask = ~out_mask
            data.edge_index = data.edge_index[:, keep_mask]

            # Add context -> interface and interface -> strategies
            ctx_to_iface = torch.tensor([[context], [n_nodes]], device=self.device)
            iface_to_strat = torch.stack([torch.full_like(strategies, n_nodes), strategies])

            new_edges = torch.cat([ctx_to_iface, iface_to_strat], dim=1)
            data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)

            # Update edge attributes
            if hasattr(data, 'edge_attr'):
                data.edge_attr = data.edge_attr[keep_mask]
                new_edge_attr = torch.ones(new_edges.size(1), 1, device=self.device, dtype=torch.float32)
                data.edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)

            # Incremental feature update
            affected_nodes = IncrementalFeatureComputer.get_affected_nodes(
                'strategy_pattern', context, context, to_networkx(data, to_undirected=False)
            )
            affected_nodes.add(n_nodes)  # Include strategy interface
            affected_nodes.update(strategies.tolist())  # Include all strategies

            data = IncrementalFeatureComputer.update_features_incremental(
                data, affected_nodes, {'pattern': 'strategy', 'strategy_interface': n_nodes}
            )

            return True, data, {
                'strategy_interface': n_nodes,
                'strategies': strategies.tolist(),
                'affected_nodes': list(affected_nodes),
                'modification_type': 'strategy_pattern'
            }
        except Exception as e:
            return False, None, {'error': f'strategy_pattern_failed: {str(e)}'}

    def _remove_middleman_incremental(self, middleman: int, _: int) -> Tuple[bool, Optional[Data], Dict]:
        """Remove unnecessary middleman to reduce indirection with incremental updates"""
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

                # Update edge attributes
                if hasattr(data, 'edge_attr'):
                    data.edge_attr = data.edge_attr[~remove_mask]
                    new_edge_attr = torch.ones(direct_edges.size(1), 1, device=self.device, dtype=torch.float32)
                    data.edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)

            # Incremental feature update
            affected_nodes = IncrementalFeatureComputer.get_affected_nodes(
                'middleman_removal', middleman, middleman, to_networkx(data, to_undirected=False)
            )
            affected_nodes.update(predecessors.tolist())
            affected_nodes.update(successors.tolist())

            data = IncrementalFeatureComputer.update_features_incremental(
                data, affected_nodes, {'pattern': 'remove_middleman', 'removed_node': middleman}
            )

            return True, data, {
                'removed_middleman': middleman,
                'direct_connections': direct_edges.size(1) if direct_edges else 0,
                'affected_nodes': list(affected_nodes),
                'modification_type': 'middleman_removal'
            }
        except Exception as e:
            return False, None, {'error': f'remove_middleman_failed: {str(e)}'}

    def _evaluate_final_state(self) -> float:
        """Final state evaluation with centralized config thresholds"""
        try:
            if self.initial_hub_score is None:
                return 0.0

            final_hub_score = self._get_current_hub_score()
            total_improvement = self.initial_hub_score - final_hub_score

            # Calculate relative improvement
            relative_improvement = total_improvement / (self.initial_hub_score + 1e-8)

            # Base reward calculation using centralized config thresholds
            if final_hub_score < self.HUB_SCORE_EXCELLENT:  # Excellent final state
                base_reward = self.REWARD_SUCCESS * 1.5
            elif final_hub_score < self.HUB_SCORE_GOOD:  # Good final state
                base_reward = self.REWARD_SUCCESS
            elif final_hub_score < self.HUB_SCORE_ACCEPTABLE:  # Acceptable final state
                base_reward = self.REWARD_PARTIAL_SUCCESS
            elif total_improvement > 0.1:  # Significant improvement but not great final state
                base_reward = self.REWARD_PARTIAL_SUCCESS * 0.7
            elif total_improvement > 0.05:  # Some improvement
                base_reward = self.REWARD_PARTIAL_SUCCESS * 0.3
            else:  # No improvement or worse
                base_reward = self.REWARD_FAILURE

            # Efficiency bonus (fewer steps = better)
            efficiency = 1.0 - (self.steps / self.max_steps)
            efficiency_bonus = efficiency * 5.0

            # Consistency bonus (steady improvement)
            consistency_bonus = 0.0
            if len(self.hub_score_history) > 1:
                improvements = [self.hub_score_history[i - 1] - self.hub_score_history[i]
                                for i in range(1, len(self.hub_score_history))]
                positive_improvements = [imp for imp in improvements if imp > 0]
                if len(positive_improvements) > len(improvements) * 0.6:  # More than 60% positive
                    consistency_bonus = 3.0

            total_reward = base_reward + efficiency_bonus + consistency_bonus

            logger.debug(f"Final evaluation (centralized config) - Initial: {self.initial_hub_score:.4f}, "
                         f"Final: {final_hub_score:.4f}, Total improvement: {total_improvement:.4f}, "
                         f"Reward: {total_reward:.2f}")

            return total_reward

        except Exception as e:
            logger.warning(f"Failed to evaluate final state: {e}")
            return 0.0

    def get_reward_config_summary(self) -> Dict[str, Any]:
        """Get summary of current reward configuration for debugging"""
        return {
            'REWARD_SUCCESS': self.REWARD_SUCCESS,
            'REWARD_PARTIAL_SUCCESS': self.REWARD_PARTIAL_SUCCESS,
            'REWARD_FAILURE': self.REWARD_FAILURE,
            'REWARD_STEP': self.REWARD_STEP,
            'REWARD_HUB_REDUCTION': self.REWARD_HUB_REDUCTION,
            'REWARD_INVALID': self.REWARD_INVALID,
            'REWARD_SMALL_IMPROVEMENT': self.REWARD_SMALL_IMPROVEMENT,
            'HUB_SCORE_EXCELLENT': self.HUB_SCORE_EXCELLENT,
            'HUB_SCORE_GOOD': self.HUB_SCORE_GOOD,
            'HUB_SCORE_ACCEPTABLE': self.HUB_SCORE_ACCEPTABLE,
            'improvement_multiplier_boost': self.improvement_multiplier_boost,
            'improvement_multiplier_decay': self.improvement_multiplier_decay,
            'improvement_multiplier_max': self.improvement_multiplier_max,
            'improvement_multiplier_min': self.improvement_multiplier_min,
            'current_improvement_multiplier': self.improvement_multiplier,
            'config_source': 'centralized_hyperparameters_configuration'
        }

    def log_reward_calculation_details(self, hub_improvement: float, reward_type: str) -> None:
        """Log detailed reward calculation for debugging"""
        logger.debug(f"Reward calculation details (centralized config):")
        logger.debug(f"  Hub improvement: {hub_improvement:.4f}")
        logger.debug(f"  Reward type: {reward_type}")
        logger.debug(f"  Improvement multiplier: {self.improvement_multiplier:.3f}")
        logger.debug(f"  REWARD_SMALL_IMPROVEMENT: {self.REWARD_SMALL_IMPROVEMENT}")
        logger.debug(f"  HUB_SCORE_EXCELLENT threshold: {self.HUB_SCORE_EXCELLENT}")
        logger.debug(f"  HUB_SCORE_GOOD threshold: {self.HUB_SCORE_GOOD}")
        logger.debug(f"  HUB_SCORE_ACCEPTABLE threshold: {self.HUB_SCORE_ACCEPTABLE}")