#!/usr/bin/env python3
"""
Enhanced feature extraction focusing on hub-specific metrics:
Fan-In, Fan-Out, Degree-Centrality, In/Out Ratio, PageRank, Betweenness Centrality, Closeness

This module extracts the most relevant features for hub detection from 1-hop subgraphs.
"""

import logging
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

logger = logging.getLogger(__name__)


class HubFocusedFeatureExtractor:
    """Extract the core features for hub detection from ego graphs"""

    @staticmethod
    def compute_hub_features(G: nx.Graph, center_node: str) -> Optional[Dict]:
        """
        Compute the 7 core hub-detection features:
        1. Fan-In (in-degree)
        2. Fan-Out (out-degree) 
        3. Degree-Centrality
        4. In/Out Ratio
        5. PageRank
        6. Betweenness Centrality
        7. Closeness Centrality
        """
        if center_node not in G:
            return None

        try:
            # Basic degree metrics
            in_degrees = dict(G.in_degree())
            out_degrees = dict(G.out_degree())

            # Compute centrality metrics efficiently
            if len(G) <= 100:  # For reasonable-sized graphs
                pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-4)
                betweenness = nx.betweenness_centrality(G, normalized=True)
                closeness = nx.closeness_centrality(G, distance=None, wf_improved=True)
            else:
                # For large graphs, use approximations or simplified metrics
                logger.warning(f"Large graph ({len(G)} nodes), using simplified centrality metrics")
                total_edges = G.number_of_edges()
                pagerank = {n: float(G.degree(n)) / (2 * total_edges + 1e-8) for n in G.nodes()}
                betweenness = {n: 0.0 for n in G.nodes()}  # Skip expensive computation
                closeness = {n: 1.0 / (len(G) - 1 + 1e-8) for n in G.nodes()}  # Approximation

            # Build feature dict for all nodes
            node_features = {}

            for node in G.nodes():
                fan_in = float(in_degrees.get(node, 0))
                fan_out = float(out_degrees.get(node, 0))
                total_degree = fan_in + fan_out

                # Core hub features
                features = {
                    'fan_in': fan_in,
                    'fan_out': fan_out,
                    'degree_centrality': total_degree / (len(G) - 1 + 1e-8),
                    'in_out_ratio': fan_in / (fan_out + 1e-8),
                    'pagerank': float(pagerank.get(node, 0)),
                    'betweenness_centrality': float(betweenness.get(node, 0)),
                    'closeness_centrality': float(closeness.get(node, 0))
                }

                node_features[node] = features

            return node_features

        except Exception as e:
            logger.warning(f"Failed to compute hub features: {e}")
            return None

    @staticmethod
    def compute_graph_level_features(G: nx.Graph, center_node: str) -> Dict:
        """Compute graph-level features for the ego graph"""
        try:
            # Basic graph properties
            num_nodes = len(G)
            num_edges = G.number_of_edges()
            density = nx.density(G)

            # Center node properties
            center_in_degree = G.in_degree(center_node)
            center_out_degree = G.out_degree(center_node)
            center_total_degree = center_in_degree + center_out_degree

            # Connectivity patterns
            is_strongly_connected = nx.is_strongly_connected(G)
            is_weakly_connected = nx.is_weakly_connected(G)

            # Component analysis
            num_strongly_connected = nx.number_strongly_connected_components(G)
            num_weakly_connected = nx.number_weakly_connected_components(G)

            # Hub dominance metrics
            all_degrees = [d for n, d in G.degree()]
            max_degree = max(all_degrees) if all_degrees else 0
            avg_degree = np.mean(all_degrees) if all_degrees else 0
            center_degree_ratio = center_total_degree / (max_degree + 1e-8)

            return {
                'subgraph_size': num_nodes,
                'subgraph_edges': num_edges,
                'graph_density': density,
                'center_fan_in': center_in_degree,
                'center_fan_out': center_out_degree,
                'center_total_degree': center_total_degree,
                'is_strongly_connected': int(is_strongly_connected),
                'is_weakly_connected': int(is_weakly_connected),
                'num_strong_components': num_strongly_connected,
                'num_weak_components': num_weakly_connected,
                'max_degree': max_degree,
                'avg_degree': avg_degree,
                'center_degree_dominance': center_degree_ratio
            }

        except Exception as e:
            logger.warning(f"Failed to compute graph-level features: {e}")
            return {}


def create_hub_focused_data(
        G: nx.Graph,
        center_node: str,
        is_smelly: bool,
        config: dict
) -> Optional[Data]:
    """
    Create PyG Data object with hub-focused features for 1-hop ego graph
    """

    # Validate center node exists
    if center_node not in G:
        logger.debug(f"Center node {center_node} not found in graph")
        return None

    # Extract 1-hop ego graph
    try:
        ego_graph = nx.ego_graph(G, center_node, radius=1, undirected=False)
    except Exception as e:
        logger.warning(f"Failed to extract ego graph for {center_node}: {e}")
        return None

    # Size validation
    min_size = config.get('min_subgraph_size', 3)
    if len(ego_graph) < min_size:
        logger.debug(f"Ego graph too small: {len(ego_graph)} < {min_size}")
        return None

    max_size = config.get('max_subgraph_size', 200)  # Reasonable limit for 1-hop
    if len(ego_graph) > max_size:
        logger.debug(f"Ego graph too large: {len(ego_graph)} > {max_size}")
        return None

    # Remove isolated nodes if configured
    if config.get('remove_isolated_nodes', True):
        isolated = list(nx.isolates(ego_graph))
        if isolated:
            ego_graph.remove_nodes_from(isolated)
            if len(ego_graph) < min_size:
                return None

    # Minimum edges check
    min_edges = config.get('min_edges', 1)
    if ego_graph.number_of_edges() < min_edges:
        logger.debug(f"Not enough edges: {ego_graph.number_of_edges()} < {min_edges}")
        return None

    # Extract hub-focused features
    extractor = HubFocusedFeatureExtractor()
    node_features = extractor.compute_hub_features(ego_graph, center_node)

    if node_features is None:
        logger.warning(f"Failed to compute hub features for {center_node}")
        return None

    # Apply features to graph nodes (clear existing attributes first)
    for node, attrs in ego_graph.nodes(data=True):
        attrs.clear()  # Remove all original attributes
        if node in node_features:
            attrs.update(node_features[node])
        else:
            # Fallback for missing nodes
            attrs.update({
                'fan_in': 0.0,
                'fan_out': 0.0,
                'degree_centrality': 0.0,
                'in_out_ratio': 0.0,
                'pagerank': 0.0,
                'betweenness_centrality': 0.0,
                'closeness_centrality': 0.0
            })

    # Simplify edge attributes (uniform weights)
    for u, v, attrs in ego_graph.edges(data=True):
        attrs.clear()
        attrs['weight'] = 1.0  # Simple uniform edge weight

    # Convert to PyG Data
    try:
        node_attrs = ['fan_in', 'fan_out', 'degree_centrality', 'in_out_ratio',
                      'pagerank', 'betweenness_centrality', 'closeness_centrality']
        edge_attrs = ['weight']

        data = from_networkx(ego_graph, group_node_attrs=node_attrs, group_edge_attrs=edge_attrs)

    except Exception as e:
        logger.warning(f"Failed to convert to PyG Data: {e}")
        return None

    # Add metadata
    data.is_smelly = torch.tensor([int(is_smelly)], dtype=torch.long)
    data.center_node = center_node

    # Graph-level features
    graph_features = extractor.compute_graph_level_features(ego_graph, center_node)
    for key, value in graph_features.items():
        setattr(data, key, torch.tensor([float(value)], dtype=torch.float32))

    # Additional metadata for tracking
    data.subgraph_size = len(ego_graph)
    data.num_edges_sg = ego_graph.number_of_edges()

    # Create comprehensive ID mapping and store original IDs
    node_list = list(ego_graph.nodes())  # Deterministic order
    node_mapping = {n: i for i, n in enumerate(node_list)}
    data.center_node_idx = torch.tensor([node_mapping.get(center_node, 0)], dtype=torch.long)

    # Store original node IDs with multiple formats for compatibility
    data.original_node_ids = [str(n) for n in node_list]  # String format
    data.node_id_to_index = node_mapping  # Original ID -> tensor index mapping
    data.index_to_node_id = {i: str(n) for i, n in enumerate(node_list)}  # Reverse mapping

    # Store edge ID mapping for complete traceability
    edge_id_mapping = []
    original_edge_ids = []

    for i, (u, v, attrs) in enumerate(ego_graph.edges(data=True)):
        # Try to get original edge ID from attributes, or create one
        edge_id = attrs.get('id', attrs.get('edge_id', f"{u}->{v}"))
        original_edge_ids.append(str(edge_id))
        edge_id_mapping.append({
            'edge_index': i,
            'source_id': str(u),
            'target_id': str(v),
            'source_tensor_idx': node_mapping.get(u, -1),
            'target_tensor_idx': node_mapping.get(v, -1),
            'original_edge_id': str(edge_id)
        })

    data.original_edge_ids = original_edge_ids
    data.edge_id_mapping = edge_id_mapping

    return data


def validate_extracted_features(data: Data) -> bool:
    """Validate that extracted features are reasonable"""
    try:
        # Check basic structure
        if not hasattr(data, 'x') or data.x is None:
            return False

        if data.x.size(1) != 7:  # Should have exactly 7 hub features
            logger.warning(f"Expected 7 features, got {data.x.size(1)}")
            return False

        # Check for NaN or infinite values
        if torch.isnan(data.x).any() or torch.isinf(data.x).any():
            logger.warning("Found NaN or infinite values in features")
            return False

        # Check feature ranges (basic sanity checks)
        fan_in = data.x[:, 0]  # fan_in should be >= 0
        fan_out = data.x[:, 1]  # fan_out should be >= 0
        pagerank = data.x[:, 4]  # pagerank should be in [0, 1]

        if (fan_in < 0).any() or (fan_out < 0).any():
            logger.warning("Found negative degree values")
            return False

        if (pagerank < 0).any() or (pagerank > 1.1).any():  # Small tolerance
            logger.warning("PageRank values out of expected range")
            return False

        return True

    except Exception as e:
        logger.warning(f"Feature validation failed: {e}")
        return False


def extract_features_from_graphml(
        graphml_path,
        center_nodes: list,
        smell_map: dict,
        version_hash: str,
        config: dict
) -> list:
    """
    Extract hub-focused features from a single GraphML file

    Args:
        graphml_path: Path to the GraphML file
        center_nodes: List of center node IDs to process
        smell_map: Dictionary mapping component IDs to smelly versions
        version_hash: Current version hash
        config: Configuration dictionary

    Returns:
        List of PyG Data objects with extracted features
    """
    extracted_data = []

    try:
        # Load the graph
        G = nx.read_graphml(graphml_path)
        logger.debug(f"Loaded graph: {len(G)} nodes, {G.number_of_edges()} edges")

        # Check graph size limits
        max_graph_size = config.get('max_graph_size', 10000)
        if len(G) > max_graph_size:
            logger.warning(f"Graph too large ({len(G)} nodes), skipping")
            return extracted_data

        # Process each center node
        for center_node in center_nodes:
            # Determine if this is a smelly instance
            smelly_versions = smell_map.get(center_node, [])
            is_smelly = version_hash in smelly_versions

            # Skip non-smelly if not configured to include them
            if not is_smelly and not config.get('include_non_smelly', False):
                continue

            # Extract features
            data = create_hub_focused_data(G, center_node, is_smelly, config)

            if data is not None:
                # Validate extracted features
                if validate_extracted_features(data):
                    # Add version information
                    data.version_hash = version_hash
                    data.graphml_path = str(graphml_path)

                    extracted_data.append(data)
                else:
                    logger.warning(f"Feature validation failed for {center_node}")

        logger.debug(f"Extracted {len(extracted_data)} valid subgraphs from {graphml_path.name}")

    except Exception as e:
        logger.error(f"Failed to process {graphml_path}: {e}")

    return extracted_data


def get_hub_feature_stats(data_list: list) -> dict:
    """Generate statistics about extracted hub features"""
    if not data_list:
        return {}

    # Collect all features
    all_features = torch.cat([data.x for data in data_list], dim=0)
    feature_names = ['fan_in', 'fan_out', 'degree_centrality', 'in_out_ratio',
                     'pagerank', 'betweenness_centrality', 'closeness_centrality']

    stats = {}
    for i, name in enumerate(feature_names):
        feature_values = all_features[:, i]
        stats[name] = {
            'mean': float(feature_values.mean()),
            'std': float(feature_values.std()),
            'min': float(feature_values.min()),
            'max': float(feature_values.max()),
            'median': float(feature_values.median())
        }

    # Additional statistics
    stats['total_samples'] = len(data_list)
    stats['total_nodes'] = int(all_features.size(0))
    stats['avg_subgraph_size'] = sum(data.subgraph_size for data in data_list) / len(data_list)

    # Smelly vs non-smelly distribution
    smelly_count = sum(1 for data in data_list if hasattr(data, 'is_smelly') and data.is_smelly.item())
    stats['smelly_samples'] = smelly_count
    stats['non_smelly_samples'] = len(data_list) - smelly_count

    return stats


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Test hub-focused feature extraction")
    parser.add_argument("graphml_file", type=Path, help="Path to GraphML file")
    parser.add_argument("center_node", help="Center node ID")
    parser.add_argument("--output", type=Path, help="Output directory for test results")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    # Test configuration
    config = {
        'min_subgraph_size': 3,
        'max_subgraph_size': 200,
        'remove_isolated_nodes': True,
        'min_edges': 1
    }

    try:
        # Load graph and extract features
        G = nx.read_graphml(args.graphml_file)
        logger.info(f"Loaded graph: {len(G)} nodes, {G.number_of_edges()} edges")

        # Test feature extraction
        data = create_hub_focused_data(G, args.center_node, is_smelly=True, config=config)

        if data is not None:
            logger.info(f"✅ Successfully extracted features for {args.center_node}")
            logger.info(f"   - Subgraph size: {data.subgraph_size}")
            logger.info(f"   - Features shape: {data.x.shape}")
            logger.info(f"   - Feature validation: {validate_extracted_features(data)}")

            # Show feature summary
            feature_names = ['fan_in', 'fan_out', 'degree_centrality', 'in_out_ratio',
                             'pagerank', 'betweenness_centrality', 'closeness_centrality']

            center_idx = data.center_node_idx.item()
            center_features = data.x[center_idx]

            logger.info(f"   - Center node features:")
            for i, name in enumerate(feature_names):
                logger.info(f"     {name}: {center_features[i].item():.4f}")

            # Save results if output directory specified
            if args.output:
                args.output.mkdir(parents=True, exist_ok=True)

                # Save the Data object
                torch.save(data, args.output / f"{args.center_node}_test.pt")

                # Save feature statistics
                stats = get_hub_feature_stats([data])
                with open(args.output / f"{args.center_node}_stats.json", 'w') as f:
                    json.dump(stats, f, indent=2)

                logger.info(f"   - Results saved to {args.output}")

        else:
            logger.error(f"❌ Failed to extract features for {args.center_node}")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise