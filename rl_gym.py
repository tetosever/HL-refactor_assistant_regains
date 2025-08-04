#!/usr/bin/env python3
"""
Updated RL Environment with Incremental Feature Computation

Key changes:
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
    """RL Environment with incremental feature computation for intermediate graphs"""

    def __init__(self, initial_data: Data, discriminator: nn.Module,
                 max_steps: int = 20, device: torch.device = torch.device('cpu')):
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

    def _ensure_data_consistency(self, data: Data) -> Data:
        """Ensure batch and edge_attr consistency with unified format"""
        num_nodes = data.x.size(0)
        num_edges = data.edge_index.size(1)

        if not hasattr(data, 'batch') or data.batch is None or data.batch.size(0) != num_nodes:
            data.batch = torch.zeros(num_nodes, dtype=torch.long, device=data.x.device)

        # Ensure edge_attr exists with correct format (weight=1.0 as in dataset creation)
        if not hasattr(data, 'edge_attr') or data.edge_attr is None or data.edge_attr.size(0) != num_edges:
            data.edge_attr = torch.ones(num_edges, 1, dtype=torch.float32, device=data.x.device)

        return data

    def reset(self) -> Data:
        """Reset environment with unified data format"""
        self.current_data = copy.deepcopy(self.initial_data)
        self._ensure_data_consistency(self.current_data)
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
            self._ensure_data_consistency(self.current_data)
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

    def step(self, action: Tuple[int, int, int, bool]) -> Tuple[Data, float, bool, Dict]:
        """Execute refactoring action with incremental feature computation"""
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

            # Apply refactoring pattern with incremental feature update
            old_hub_score = self._get_current_hub_score()
            success, new_data, pattern_info = self._apply_pattern_incremental(pattern, source, target)

            if success and new_data is not None:
                # Update current data with incrementally updated features
                self.current_data = new_data

                # Invalidate cache after graph modification
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
                    'pattern_info': pattern_info,
                    'features_updated_incrementally': True,
                    'affected_nodes': pattern_info.get('affected_nodes', [])
                })
            else:
                reward += self.REWARD_INVALID
                info['error'] = pattern_info.get('error', 'pattern_failed')

        except Exception as e:
            logger.warning(f"Error in step: {e}")
            reward += self.REWARD_INVALID
            info['error'] = f'step_error: {str(e)}'

        return self.current_data, reward, done, info

    def _apply_pattern_incremental(self, pattern: int, source: int, target: int) -> Tuple[bool, Optional[Data], Dict]:
        """Apply specific refactoring pattern with incremental feature computation"""
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
            return patterns[pattern](source, target)
        except Exception as e:
            logger.warning(f"Pattern {pattern} failed: {e}")
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