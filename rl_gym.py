#!/usr/bin/env python3
"""
Updated RL Environment with Hub-Focused Refactoring Patterns
VERSIONE OTTIMIZZATA con logging ridotto e performance migliorate
"""

import copy
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List, Set

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

import logging

# Configurazione logging ottimizzata per questo modulo
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Solo warning ed errori per default

# Import centralized configuration
from hyperparameters_configuration import get_improved_config


@dataclass
class RefactoringResult:
    """Result of applying a refactoring pattern"""
    success: bool
    new_data: Optional[Data]
    info: Dict[str, Any]


class IncrementalFeatureComputer:
    """Incremental feature computation - versione ottimizzata"""

    @staticmethod
    def compute_single_node_features(node_id: int, G: nx.Graph,
                                     global_centrality: Optional[Dict] = None) -> List[float]:
        """Compute 7 structural features for a single node"""
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
            # Solo per debug - non loggare per ogni failure
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Feature computation failed for node {node_id}: {e}")
            return [0.0] * 7

    @staticmethod
    def _compute_global_centrality(G: nx.Graph) -> Dict[str, Dict]:
        """Compute global centrality measures efficiently"""
        try:
            num_nodes = len(G)

            if num_nodes <= 100:
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
            # Solo errori critici per grafi grandi
            if len(G.nodes()) > 100:
                logger.error(f"Critical centrality computation failed for large graph ({len(G.nodes())} nodes): {e}")

            # Fallback
            empty_dict = {n: 0.0 for n in G.nodes()}
            return {
                'pagerank': empty_dict,
                'betweenness': empty_dict,
                'closeness': empty_dict
            }

    @staticmethod
    def get_affected_nodes(modification_type: str, source_node: int, target_node: int,
                           G: nx.Graph, radius: int = 2) -> Set[int]:
        """Determine which nodes need feature updates"""
        affected = {source_node, target_node}

        try:
            # Add direct neighbors of modified nodes
            if source_node in G:
                affected.update(set(G.predecessors(source_node)))
                affected.update(set(G.successors(source_node)))

            if target_node in G and target_node != source_node:
                affected.update(set(G.predecessors(target_node)))
                affected.update(set(G.successors(target_node)))

            # For certain modifications, expand the radius
            if modification_type in ['node_addition', 'node_removal', 'major_restructuring']:
                expanded = set(affected)
                for node in list(affected):
                    if node in G:
                        expanded.update(set(G.neighbors(node)))
                affected = expanded

        except Exception as e:
            # Solo per debug
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Error determining affected nodes: {e}")

        return affected

    @staticmethod
    def update_features_incremental(data: Data, affected_nodes: Set[int],
                                    modification_info: Dict = None) -> Data:
        """Update features for affected nodes incrementally"""
        try:
            # Convert to NetworkX for feature computation
            G = to_networkx(data, to_undirected=False)

            # Filter affected nodes to only those that exist
            valid_affected = {node for node in affected_nodes if node < data.x.size(0) and node in G}

            if not valid_affected:
                return data

            # Compute global centrality once for efficiency
            global_centrality = IncrementalFeatureComputer._compute_global_centrality(G)

            # Update features for each affected node
            for node_idx in valid_affected:
                try:
                    new_features = IncrementalFeatureComputer.compute_single_node_features(
                        node_idx, G, global_centrality
                    )
                    data.x[node_idx] = torch.tensor(new_features, dtype=torch.float32, device=data.x.device)
                except Exception:
                    continue  # Skip failed nodes

            # Solo per debug
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Updated features for {len(valid_affected)} nodes")

            return data

        except Exception as e:
            logger.error(f"Critical incremental feature update failed: {e}")
            return data


class HubRefactoringEnv:
    """RL Environment with Hub-Focused refactoring patterns - Versione Ottimizzata"""

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

        self.reward_config = reward_config

        # CENTRALIZED REWARD PARAMETERS
        self.REWARD_SUCCESS = reward_config.REWARD_SUCCESS
        self.REWARD_PARTIAL_SUCCESS = reward_config.REWARD_PARTIAL_SUCCESS
        self.REWARD_FAILURE = reward_config.REWARD_FAILURE
        self.REWARD_STEP = reward_config.REWARD_STEP
        self.REWARD_HUB_REDUCTION = reward_config.REWARD_HUB_REDUCTION
        self.REWARD_INVALID = reward_config.REWARD_INVALID
        self.REWARD_SMALL_IMPROVEMENT = reward_config.REWARD_SMALL_IMPROVEMENT

        # Hub score thresholds
        self.HUB_SCORE_EXCELLENT = reward_config.HUB_SCORE_EXCELLENT
        self.HUB_SCORE_GOOD = reward_config.HUB_SCORE_GOOD
        self.HUB_SCORE_ACCEPTABLE = reward_config.HUB_SCORE_ACCEPTABLE

        # Progressive reward scaling
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

        # Log solo inizializzazione se debug attivo
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Environment initialized with Hub-Focused patterns")

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
        """Reset environment"""
        self.current_data = copy.deepcopy(self.initial_data)
        self._ensure_data_consistency(self.current_data)
        self.steps = 0
        self.action_history.clear()
        self.hub_score_history.clear()

        # Reset cache
        self._cached_hub_score = None
        self._graph_hash = None

        # Compute initial hub score
        initial_score = self._get_current_hub_score()
        self.initial_hub_score = initial_score
        self.best_hub_score = initial_score
        self.hub_score_history.append(initial_score)

        # Reset improvement multiplier
        self.improvement_multiplier = 1.0

        # Log solo se debug attivo
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Environment reset - hub score: {initial_score:.4f}")

        return self.current_data

    def _get_graph_hash(self, data: Data) -> str:
        """Generate hash for graph structure"""
        try:
            edge_hash = hash(tuple(data.edge_index.flatten().tolist()))
            node_hash = hash(tuple(data.x.flatten().tolist()))
            return f"{edge_hash}_{node_hash}"
        except Exception:
            return str(hash(str(data)))

    def _get_current_hub_score(self) -> float:
        """Hub score computation with caching"""
        try:
            self._ensure_data_consistency(self.current_data)
            current_hash = self._get_graph_hash(self.current_data)

            # Return cached score if graph hasn't changed
            if self._cached_hub_score is not None and self._graph_hash == current_hash:
                return self._cached_hub_score

            # Compute new score
            with torch.no_grad():
                self.discriminator.eval()
                disc_out = self.discriminator(self.current_data)

                if disc_out is None or 'logits' not in disc_out:
                    return 0.5  # Neutral fallback

                logits = disc_out['logits']
                if logits.dim() == 0 or logits.size(0) == 0:
                    return 0.5

                # Get probability of being "smelly" (hub-like)
                probs = F.softmax(logits, dim=1)
                if probs.size(1) < 2:
                    return 0.5

                score = probs[0, 1].item()

            # Validate score
            if not (0.0 <= score <= 1.0) or math.isnan(score):
                score = 0.5

            # Cache the result
            self._cached_hub_score = score
            self._graph_hash = current_hash

            return score

        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Hub score computation failed: {e}")
            return 0.5

    def step(self, action: Tuple[int, int, int, bool]) -> Tuple[Data, float, bool, Dict]:
        """Step with optimized logging"""
        source, target, pattern, terminate = action
        self.steps += 1

        reward = self.REWARD_STEP
        done = False

        # Initialize info with default values
        info = {
            'valid': False,
            'pattern': pattern,
            'success': False,
            'hub_improvement': 0.0,
            'old_hub_score': 0.5,
            'new_hub_score': 0.5,
            'best_hub_score': self.best_hub_score if self.best_hub_score else 0.5,
            'step_metrics_calculated': False,
            'step_error': None
        }

        try:
            # Check termination
            if terminate or self.steps >= self.max_steps:
                done = True
                final_reward = self._evaluate_final_state()
                reward += final_reward
                info.update({
                    'termination': 'requested' if terminate else 'max_steps',
                    'final_reward': final_reward,
                    'step_metrics_calculated': True
                })
                return self.current_data, reward, done, info

            # Validate action - non loggare come errore, è normale
            if source >= self.current_data.x.size(0) or target >= self.current_data.x.size(0):
                reward += self.REWARD_INVALID
                info.update({
                    'error': 'invalid_node_index',
                    'step_error': f'Invalid indices: source={source}, target={target}, max_nodes={self.current_data.x.size(0)}'
                })
                return self.current_data, reward, done, info

            # Get hub scores
            old_hub_score = self._get_current_hub_score_safe()
            info['old_hub_score'] = old_hub_score

            # Apply refactoring pattern
            success, new_data, pattern_info = self._apply_pattern_incremental(pattern, source, target)

            if success and new_data is not None:
                # Update current data
                self.current_data = new_data

                # Invalidate cache after graph modification
                self._cached_hub_score = None
                self._graph_hash = None

                # Get new hub score
                new_hub_score = self._get_current_hub_score_safe()
                hub_improvement = old_hub_score - new_hub_score

                # Track best score
                if new_hub_score < self.best_hub_score:
                    self.best_hub_score = new_hub_score
                    info['new_best'] = True
                    self.improvement_multiplier = min(
                        self.improvement_multiplier * self.improvement_multiplier_boost,
                        self.improvement_multiplier_max
                    )

                self.hub_score_history.append(new_hub_score)

                # REWARD CALCULATION
                if hub_improvement > 0.15:
                    reward += self.REWARD_HUB_REDUCTION * hub_improvement * self.improvement_multiplier
                    info['reward_type'] = 'significant_improvement'
                elif hub_improvement > 0.05:
                    reward += self.REWARD_PARTIAL_SUCCESS * hub_improvement * self.improvement_multiplier
                    info['reward_type'] = 'good_improvement'
                elif hub_improvement > 0.01:
                    reward += self.REWARD_SMALL_IMPROVEMENT * hub_improvement * self.improvement_multiplier
                    info['reward_type'] = 'small_improvement'
                elif hub_improvement > -0.02:
                    reward += 0.1
                    info['reward_type'] = 'neutral'
                else:
                    reward += self.REWARD_FAILURE * abs(hub_improvement)
                    info['reward_type'] = 'worse'
                    self.improvement_multiplier = max(
                        self.improvement_multiplier * self.improvement_multiplier_decay,
                        self.improvement_multiplier_min
                    )

                # BONUS REWARDS
                if new_hub_score < self.HUB_SCORE_EXCELLENT:
                    reward += 5.0
                    info['score_bonus'] = 'excellent'
                elif new_hub_score < self.HUB_SCORE_GOOD:
                    reward += 2.0
                    info['score_bonus'] = 'good'
                elif new_hub_score < self.HUB_SCORE_ACCEPTABLE:
                    reward += 0.5
                    info['score_bonus'] = 'acceptable'

                # Record action
                self.action_history.append((source, target, pattern, hub_improvement))

                # Update info
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
                    'step_metrics_calculated': True,
                    'hub_score_is_fallback': old_hub_score == 0.5 and new_hub_score == 0.5
                })
            else:
                # Pattern failed
                reward += self.REWARD_INVALID
                info.update({
                    'error': pattern_info.get('error', 'pattern_failed'),
                    'reward_type': 'pattern_failed',
                    'step_error': f"Pattern {pattern} failed: {pattern_info.get('error', 'unknown')}"
                })

        except Exception as e:
            logger.error(f"Critical error in step: {e}")
            reward += self.REWARD_INVALID
            info.update({
                'error': f'step_error: {str(e)}',
                'reward_type': 'error',
                'step_error': str(e)
            })

        return self.current_data, reward, done, info

    def _apply_pattern_incremental(self, pattern: int, source: int, target: int) -> Tuple[bool, Optional[Data], Dict]:
        """Apply hub-focused refactoring pattern"""
        patterns = {
            0: self._split_by_responsibility_incremental,
            1: self._extract_interface_incremental,
            2: self._dependency_injection_incremental,
            3: self._extract_superclass_incremental,
            4: self._move_method_incremental
        }

        if pattern not in patterns:
            return False, None, {'error': 'unknown_pattern'}

        try:
            success, new_data, info = patterns[pattern](source, target)

            # Validation critica
            if success and new_data is not None:
                if new_data.x.size(0) == 0 or new_data.edge_index.size(1) == 0:
                    logger.error(f"Pattern {pattern} created empty graph")
                    return False, None, {'error': 'empty_graph_created'}

                if torch.any(torch.isnan(new_data.x)) or torch.any(torch.isinf(new_data.x)):
                    logger.error(f"Pattern {pattern} created invalid features")
                    return False, None, {'error': 'invalid_features_created'}

            return success, new_data, info

        except Exception as e:
            logger.error(f"Pattern {pattern} critical failure: {e}")
            return False, None, {'error': f'pattern_exception: {str(e)}'}

    def _extract_interface_incremental(self, hub: int, client: int) -> Tuple[bool, Optional[Data], Dict]:
        """PATTERN 1: Extract interface"""
        try:
            data = copy.deepcopy(self.current_data)

            # Find valid connection
            edge_mask = (data.edge_index[0] == hub) & (data.edge_index[1] == client)

            if not edge_mask.any():
                out_edges = data.edge_index[:, data.edge_index[0] == hub]
                if out_edges.size(1) == 0:
                    in_edges = data.edge_index[:, data.edge_index[1] == hub]
                    if in_edges.size(1) == 0:
                        return False, None, {'error': 'hub_has_no_connections'}
                    client = in_edges[0, 0].item()
                    edge_mask = (data.edge_index[0] == client) & (data.edge_index[1] == hub)
                    if not edge_mask.any():
                        return False, None, {'error': 'no_valid_connections_found'}
                else:
                    client = out_edges[1, 0].item()
                    edge_mask = (data.edge_index[0] == hub) & (data.edge_index[1] == client)

            # Create interface
            n_nodes = data.x.size(0)
            interface_features = torch.zeros(1, 7, device=data.x.device, dtype=torch.float32)
            data.x = torch.cat([data.x, interface_features], dim=0)

            # Update node mappings if they exist
            if hasattr(data, 'original_node_ids'):
                interface_id = f"interface_{n_nodes}_hub{hub}_client{client}"
                data.original_node_ids.append(interface_id)
                if hasattr(data, 'node_id_to_index'):
                    data.node_id_to_index[interface_id] = n_nodes
                if hasattr(data, 'index_to_node_id'):
                    data.index_to_node_id[n_nodes] = interface_id

            # Redirect edge through interface
            keep_mask = ~edge_mask
            new_edges = torch.tensor([[hub, n_nodes], [n_nodes, client]], device=self.device).t()
            data.edge_index = torch.cat([data.edge_index[:, keep_mask], new_edges], dim=1)

            # Update edge attributes
            if hasattr(data, 'edge_attr'):
                data.edge_attr = data.edge_attr[keep_mask]
                new_edge_attr = torch.ones(2, 1, device=self.device, dtype=torch.float32)
                data.edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)

            # Incremental feature update
            affected_nodes = IncrementalFeatureComputer.get_affected_nodes(
                'interface_extraction', hub, client, to_networkx(data, to_undirected=False)
            )
            affected_nodes.add(n_nodes)

            data = IncrementalFeatureComputer.update_features_incremental(
                data, affected_nodes, {'pattern': 'extract_interface', 'interface_node': n_nodes}
            )

            return True, data, {
                'interface_node': n_nodes,
                'decoupled': (hub, client),
                'affected_nodes': list(affected_nodes),
                'pattern_name': 'Extract Interface'
            }
        except Exception as e:
            return False, None, {'error': f'extract_interface_failed: {str(e)}'}

    def _dependency_injection_incremental(self, dependent: int, dependency: int) -> Tuple[bool, Optional[Data], Dict]:
        """PATTERN 2: Dependency injection"""
        try:
            data = copy.deepcopy(self.current_data)

            # Find edge in either direction
            forward_edge = (data.edge_index[0] == dependent) & (data.edge_index[1] == dependency)
            backward_edge = (data.edge_index[0] == dependency) & (data.edge_index[1] == dependent)

            edge_found = forward_edge.any() or backward_edge.any()

            if not edge_found:
                out_edges = data.edge_index[:, data.edge_index[0] == dependent]
                in_edges = data.edge_index[:, data.edge_index[1] == dependent]

                if out_edges.size(1) > 0:
                    dependency = out_edges[1, 0].item()
                    forward_edge = (data.edge_index[0] == dependent) & (data.edge_index[1] == dependency)
                elif in_edges.size(1) > 0:
                    dependency = in_edges[0, 0].item()
                    backward_edge = (data.edge_index[0] == dependency) & (data.edge_index[1] == dependent)
                else:
                    return False, None, {'error': 'no_dependencies_available'}

            # Remove appropriate edge
            if forward_edge.any():
                keep_mask = ~forward_edge
                direction = 'forward'
            else:
                keep_mask = ~backward_edge
                direction = 'backward'

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
                'direction_removed': direction,
                'affected_nodes': list(affected_nodes),
                'pattern_name': 'Dependency Injection'
            }
        except Exception as e:
            return False, None, {'error': f'dependency_injection_failed: {str(e)}'}

    def _extract_superclass_incremental(self, hub: int, target: int) -> Tuple[bool, Optional[Data], Dict]:
        """PATTERN 3: Extract superclass"""
        try:
            data = copy.deepcopy(self.current_data)

            # Find commonalities
            hub_out_mask = data.edge_index[0] == hub
            target_out_mask = data.edge_index[0] == target

            hub_targets = set(data.edge_index[1, hub_out_mask].tolist())
            target_targets = set(data.edge_index[1, target_out_mask].tolist())

            common_targets = hub_targets.intersection(target_targets)

            # If no common outgoing, try incoming
            if len(common_targets) < 1:
                hub_in_mask = data.edge_index[1] == hub
                target_in_mask = data.edge_index[1] == target

                hub_sources = set(data.edge_index[0, hub_in_mask].tolist())
                target_sources = set(data.edge_index[0, target_in_mask].tolist())

                common_sources = hub_sources.intersection(target_sources)

                if len(common_sources) >= 1:
                    common_targets = common_sources
                    extraction_direction = 'incoming'
                else:
                    # Structural similarity fallback
                    hub_degree = hub_out_mask.sum() + (data.edge_index[1] == hub).sum()
                    target_degree = target_out_mask.sum() + (data.edge_index[1] == target).sum()

                    if hub_degree >= 1 and target_degree >= 2:
                        common_targets = set()
                        extraction_direction = 'structural_similarity'
                    else:
                        return False, None, {'error': 'insufficient_structural_similarity'}
            else:
                extraction_direction = 'outgoing'

            # Create superclass node
            n_nodes = data.x.size(0)
            superclass_features = torch.zeros(1, 7, device=data.x.device, dtype=torch.float32)
            data.x = torch.cat([data.x, superclass_features], dim=0)

            # Update node mappings
            if hasattr(data, 'original_node_ids'):
                superclass_id = f"superclass_{n_nodes}_hub{hub}_target{target}"
                data.original_node_ids.append(superclass_id)
                if hasattr(data, 'node_id_to_index'):
                    data.node_id_to_index[superclass_id] = n_nodes
                if hasattr(data, 'index_to_node_id'):
                    data.index_to_node_id[n_nodes] = superclass_id

            # Handle different extraction types
            new_edges = []
            if extraction_direction == 'outgoing' and common_targets:
                # Remove common dependencies
                edges_to_remove = torch.zeros(data.edge_index.size(1), dtype=torch.bool, device=self.device)
                for common_target in common_targets:
                    hub_to_common = (data.edge_index[0] == hub) & (data.edge_index[1] == common_target)
                    target_to_common = (data.edge_index[0] == target) & (data.edge_index[1] == common_target)
                    edges_to_remove = edges_to_remove | hub_to_common | target_to_common

                keep_mask = ~edges_to_remove
                data.edge_index = data.edge_index[:, keep_mask]

                # Add new edges
                new_edges.append([hub, n_nodes])
                new_edges.append([target, n_nodes])
                for common_target in common_targets:
                    new_edges.append([n_nodes, common_target])

            elif extraction_direction == 'incoming' and common_targets:
                # Handle common sources
                edges_to_remove = torch.zeros(data.edge_index.size(1), dtype=torch.bool, device=self.device)
                for common_source in common_targets:
                    source_to_hub = (data.edge_index[0] == common_source) & (data.edge_index[1] == hub)
                    source_to_target = (data.edge_index[0] == common_source) & (data.edge_index[1] == target)
                    edges_to_remove = edges_to_remove | source_to_hub | source_to_target

                keep_mask = ~edges_to_remove
                data.edge_index = data.edge_index[:, keep_mask]

                new_edges.append([hub, n_nodes])
                new_edges.append([target, n_nodes])
                for common_source in common_targets:
                    new_edges.append([common_source, n_nodes])

            else:
                # Structural similarity - just create inheritance
                new_edges.append([hub, n_nodes])
                new_edges.append([target, n_nodes])

            # Add new edges
            if new_edges:
                new_edges_tensor = torch.tensor(new_edges, device=self.device).t()
                data.edge_index = torch.cat([data.edge_index, new_edges_tensor], dim=1)

                # Update edge attributes
                if hasattr(data, 'edge_attr'):
                    if extraction_direction in ['outgoing', 'incoming']:
                        data.edge_attr = data.edge_attr[keep_mask]
                    new_edge_attr = torch.ones(len(new_edges), 1, device=self.device, dtype=torch.float32)
                    data.edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)

            # Incremental feature update
            affected_nodes = IncrementalFeatureComputer.get_affected_nodes(
                'extract_superclass', hub, target, to_networkx(data, to_undirected=False)
            )
            affected_nodes.add(n_nodes)
            affected_nodes.update(common_targets)

            data = IncrementalFeatureComputer.update_features_incremental(
                data, affected_nodes, {'pattern': 'extract_superclass', 'superclass_node': n_nodes}
            )

            return True, data, {
                'superclass_node': n_nodes,
                'common_dependencies': list(common_targets),
                'refactored_nodes': [hub, target],
                'affected_nodes': list(affected_nodes),
                'pattern_name': 'Extract Superclass'
            }
        except Exception as e:
            return False, None, {'error': f'extract_superclass_failed: {str(e)}'}

    def _move_method_incremental(self, from_hub: int, to_target: int) -> Tuple[bool, Optional[Data], Dict]:
        """PATTERN 4: Move method"""
        try:
            data = copy.deepcopy(self.current_data)

            # Find methods to move
            hub_out_mask = data.edge_index[0] == from_hub
            hub_targets = data.edge_index[1, hub_out_mask]

            if hub_targets.size(0) == 0:
                # Try incoming edges
                hub_in_mask = data.edge_index[1] == from_hub
                hub_sources = data.edge_index[0, hub_in_mask]

                if hub_sources.size(0) == 0:
                    return False, None, {'error': 'hub_has_no_movable_methods'}

                methods_to_move = hub_sources
                move_direction = 'incoming'
            else:
                methods_to_move = hub_targets
                move_direction = 'outgoing'

            # Determine how many methods to move
            total_methods = methods_to_move.size(0)
            if total_methods == 1:
                num_to_move = 1
            elif total_methods <= 3:
                num_to_move = 1
            else:
                num_to_move = max(1, total_methods // 3)

            # Select methods to move
            selected_methods = methods_to_move[:num_to_move]

            # Move the methods
            moved_methods = []
            edges_to_remove = torch.zeros(data.edge_index.size(1), dtype=torch.bool, device=self.device)
            new_edges = []

            for method_target in selected_methods:
                method_target_int = method_target.item()
                moved_methods.append(method_target_int)

                if move_direction == 'outgoing':
                    edge_mask = (data.edge_index[0] == from_hub) & (data.edge_index[1] == method_target_int)
                    edges_to_remove = edges_to_remove | edge_mask
                    new_edges.append([to_target, method_target_int])
                else:
                    edge_mask = (data.edge_index[0] == method_target_int) & (data.edge_index[1] == from_hub)
                    edges_to_remove = edges_to_remove | edge_mask
                    new_edges.append([method_target_int, to_target])

            # Apply changes
            keep_mask = ~edges_to_remove
            data.edge_index = data.edge_index[:, keep_mask]

            if new_edges:
                new_edges_tensor = torch.tensor(new_edges, device=self.device).t()
                data.edge_index = torch.cat([data.edge_index, new_edges_tensor], dim=1)

                # Update edge attributes
                if hasattr(data, 'edge_attr'):
                    data.edge_attr = data.edge_attr[keep_mask]
                    new_edge_attr = torch.ones(len(new_edges), 1, device=self.device, dtype=torch.float32)
                    data.edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)

            # Incremental feature update
            affected_nodes = IncrementalFeatureComputer.get_affected_nodes(
                'move_method', from_hub, to_target, to_networkx(data, to_undirected=False)
            )
            affected_nodes.update(moved_methods)

            data = IncrementalFeatureComputer.update_features_incremental(
                data, affected_nodes, {'pattern': 'move_method', 'moved_methods': moved_methods}
            )

            return True, data, {
                'moved_methods': moved_methods,
                'from_node': from_hub,
                'to_node': to_target,
                'move_direction': move_direction,
                'methods_moved_count': len(moved_methods),
                'affected_nodes': list(affected_nodes),
                'pattern_name': 'Move Method'
            }
        except Exception as e:
            return False, None, {'error': f'move_method_failed: {str(e)}'}

    def _split_by_responsibility_incremental(self, large_node: int, _: int) -> Tuple[bool, Optional[Data], Dict]:
        """PATTERN 0: Split node by responsibilities"""
        try:
            data = copy.deepcopy(self.current_data)

            out_mask = data.edge_index[0] == large_node
            in_mask = data.edge_index[1] == large_node

            out_degree = out_mask.sum().item()
            in_degree = in_mask.sum().item()

            # Lower threshold for better applicability
            if out_degree + in_degree < 2:
                return False, None, {'error': 'insufficient_connections_to_split'}

            n_nodes = data.x.size(0)

            resp_features = torch.zeros(2, 7, device=data.x.device, dtype=torch.float32)
            data.x = torch.cat([data.x, resp_features], dim=0)

            # Update node mappings
            if hasattr(data, 'original_node_ids'):
                resp1_id = f"class_moved_{n_nodes}"
                resp2_id = f"class_moved_{n_nodes + 1}"
                data.original_node_ids.extend([resp1_id, resp2_id])

                if hasattr(data, 'node_id_to_index'):
                    data.node_id_to_index[resp1_id] = n_nodes
                    data.node_id_to_index[resp2_id] = n_nodes + 1
                if hasattr(data, 'index_to_node_id'):
                    data.index_to_node_id[n_nodes] = resp1_id
                    data.index_to_node_id[n_nodes + 1] = resp2_id

            # Redistribute outgoing edges
            out_edges = data.edge_index[:, out_mask]

            if out_edges.size(1) > 0:
                if out_edges.size(1) == 1:
                    mid = 1
                else:
                    mid = max(1, out_edges.size(1) // 2)

                edges_to_resp1 = torch.stack([
                    torch.full((mid,), n_nodes, device=self.device),
                    out_edges[1, :mid]
                ])

                if out_edges.size(1) > mid:
                    edges_to_resp2 = torch.stack([
                        torch.full((out_edges.size(1) - mid,), n_nodes + 1, device=self.device),
                        out_edges[1, mid:]
                    ])
                    new_edges = torch.cat([edges_to_resp1, edges_to_resp2], dim=1)
                else:
                    new_edges = edges_to_resp1

                # Keep original edges (without outgoing) and add coordinator edges
                keep_mask = ~out_mask
                coord_edges = torch.tensor([[large_node, large_node], [n_nodes, n_nodes + 1]], device=self.device)

                data.edge_index = torch.cat([data.edge_index[:, keep_mask], coord_edges, new_edges], dim=1)

                # Update edge attributes
                if hasattr(data, 'edge_attr'):
                    data.edge_attr = data.edge_attr[keep_mask]
                    num_new_edges = coord_edges.size(1) + new_edges.size(1)
                    new_edge_attr = torch.ones(num_new_edges, 1, device=self.device, dtype=torch.float32)
                    data.edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)

            # Incremental feature update
            affected_nodes = IncrementalFeatureComputer.get_affected_nodes(
                'move_class', large_node, large_node, to_networkx(data, to_undirected=False)
            )
            affected_nodes.update([n_nodes, n_nodes + 1])

            data = IncrementalFeatureComputer.update_features_incremental(
                data, affected_nodes, {'pattern': 'move_class', 'moved_classes': [n_nodes, n_nodes + 1]}
            )

            return True, data, {
                'moved_classes': [n_nodes, n_nodes + 1],
                'original_hub': large_node,
                'affected_nodes': list(affected_nodes),
                'pattern_name': 'Split Responsibility',
                'total_connections_used': out_degree + in_degree
            }
        except Exception as e:
            return False, None, {'error': f'move_class_failed: {str(e)}'}

    def _evaluate_final_state(self) -> float:
        """Final state evaluation"""
        try:
            if self.initial_hub_score is None:
                return 0.0

            final_hub_score = self._get_current_hub_score()
            total_improvement = self.initial_hub_score - final_hub_score

            # Calculate relative improvement
            relative_improvement = total_improvement / (self.initial_hub_score + 1e-8)

            # Base reward calculation
            if final_hub_score < self.HUB_SCORE_EXCELLENT:
                base_reward = self.REWARD_SUCCESS * 1.5
            elif final_hub_score < self.HUB_SCORE_GOOD:
                base_reward = self.REWARD_SUCCESS
            elif final_hub_score < self.HUB_SCORE_ACCEPTABLE:
                base_reward = self.REWARD_PARTIAL_SUCCESS
            elif total_improvement > 0.1:
                base_reward = self.REWARD_PARTIAL_SUCCESS * 0.7
            elif total_improvement > 0.05:
                base_reward = self.REWARD_PARTIAL_SUCCESS * 0.3
            else:
                base_reward = self.REWARD_FAILURE

            # Efficiency bonus
            efficiency = 1.0 - (self.steps / self.max_steps)
            efficiency_bonus = efficiency * 5.0

            # Consistency bonus
            consistency_bonus = 0.0
            if len(self.hub_score_history) > 1:
                improvements = [self.hub_score_history[i - 1] - self.hub_score_history[i]
                                for i in range(1, len(self.hub_score_history))]
                positive_improvements = [imp for imp in improvements if imp > 0]
                if len(positive_improvements) > len(improvements) * 0.6:
                    consistency_bonus = 3.0

            total_reward = base_reward + efficiency_bonus + consistency_bonus

            # Log solo per debug
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Final eval - improvement: {total_improvement:.4f}, reward: {total_reward:.2f}")

            return total_reward

        except Exception as e:
            logger.error(f"Final state evaluation failed: {e}")
            return 0.0

    def _get_current_hub_score_safe(self) -> float:
        """Safer hub score computation"""
        try:
            return self._get_current_hub_score()
        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Hub score fallback: {e}")
            return 0.5

    def get_reward_config_summary(self) -> Dict[str, Any]:
        """Get current reward configuration (no logging)"""
        return {
            'REWARD_SUCCESS': self.REWARD_SUCCESS,
            'REWARD_SMALL_IMPROVEMENT': self.REWARD_SMALL_IMPROVEMENT,
            'HUB_SCORE_EXCELLENT': self.HUB_SCORE_EXCELLENT,
            'current_improvement_multiplier': self.improvement_multiplier,
            'config_source': 'centralized_hyperparameters_configuration'
        }

    def get_pattern_info(self) -> Dict[int, str]:
        """Get information about available patterns"""
        return {
            0: "Split Responsibility (Move Class) - Primary hub resolution pattern",
            1: "Extract Interface (Interface Segregation) - Decouple hub dependencies",
            2: "Dependency Injection (Break Dependency) - Remove direct dependencies",
            3: "Extract Superclass (Pull Up) - Factor out common dependencies",
            4: "Move Method - Redistribute hub responsibilities"
        }


# Funzioni di utilità per configurare il logging
def set_environment_logging_level(level: int = logging.WARNING):
    """Configura il livello di logging per l'environment"""
    env_logger = logging.getLogger(__name__)
    env_logger.setLevel(level)

    # Riduci anche altri logger correlati
    logging.getLogger('networkx').setLevel(logging.ERROR)
    logging.getLogger('torch_geometric').setLevel(logging.ERROR)


def enable_environment_debug():
    """Abilita debug temporaneo per troubleshooting"""
    env_logger = logging.getLogger(__name__)
    env_logger.setLevel(logging.DEBUG)
    env_logger.debug("Environment debug mode enabled")


def disable_environment_debug():
    """Disabilita debug per training normale"""
    env_logger = logging.getLogger(__name__)
    env_logger.setLevel(logging.WARNING)


# Configurazione di default quando il modulo viene importato
set_environment_logging_level(logging.WARNING)