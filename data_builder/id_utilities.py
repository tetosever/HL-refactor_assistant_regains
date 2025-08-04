#!/usr/bin/env python3
"""
ID Management Utilities

Provides comprehensive utilities for managing and converting between original node/edge IDs
and PyTorch Geometric tensor indices in the hub detection pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class IDMapper:
    """Utility class for managing ID mappings in PyG data objects"""

    @staticmethod
    def get_original_node_id(data: Data, tensor_index: int) -> Optional[str]:
        """Get original node ID from tensor index"""
        if hasattr(data, 'index_to_node_id'):
            return data.index_to_node_id.get(tensor_index)
        elif hasattr(data, 'original_node_ids') and 0 <= tensor_index < len(data.original_node_ids):
            return data.original_node_ids[tensor_index]
        return None

    @staticmethod
    def get_tensor_index(data: Data, original_node_id: str) -> Optional[int]:
        """Get tensor index from original node ID"""
        if hasattr(data, 'node_id_to_index'):
            return data.node_id_to_index.get(original_node_id)
        elif hasattr(data, 'original_node_ids'):
            try:
                return data.original_node_ids.index(original_node_id)
            except ValueError:
                return None
        return None

    @staticmethod
    def get_center_node_info(data: Data) -> Dict:
        """Get comprehensive info about the center node"""
        info = {}

        if hasattr(data, 'center_node'):
            info['center_node_original_id'] = data.center_node

        if hasattr(data, 'center_node_idx'):
            center_idx = data.center_node_idx.item()
            info['center_node_tensor_index'] = center_idx

            # Get center node features if available
            if hasattr(data, 'x') and data.x is not None:
                center_features = data.x[center_idx]
                feature_names = ['fan_in', 'fan_out', 'degree_centrality', 'in_out_ratio',
                                 'pagerank', 'betweenness_centrality', 'closeness_centrality']

                if len(center_features) == len(feature_names):
                    info['center_node_features'] = {
                        name: float(center_features[i])
                        for i, name in enumerate(feature_names)
                    }

        return info

    @staticmethod
    def get_edge_info(data: Data, edge_index: int) -> Dict:
        """Get comprehensive info about a specific edge"""
        info = {}

        if hasattr(data, 'edge_id_mapping') and 0 <= edge_index < len(data.edge_id_mapping):
            info.update(data.edge_id_mapping[edge_index])

        # Add edge features if available
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            if 0 <= edge_index < data.edge_attr.size(0):
                info['edge_features'] = data.edge_attr[edge_index].tolist()

        return info

    @staticmethod
    def validate_id_mappings(data: Data) -> Tuple[bool, List[str]]:
        """Validate that ID mappings are consistent"""
        issues = []

        # Check node ID mapping consistency
        if hasattr(data, 'original_node_ids') and hasattr(data, 'node_id_to_index'):
            expected_size = len(data.original_node_ids)
            actual_size = len(data.node_id_to_index)

            if expected_size != actual_size:
                issues.append(f"Node ID mapping size mismatch: {expected_size} vs {actual_size}")

            # Check reverse mapping
            for i, node_id in enumerate(data.original_node_ids):
                if data.node_id_to_index.get(node_id) != i:
                    issues.append(f"Inconsistent mapping for node {node_id}")

        # Check center node consistency
        if hasattr(data, 'center_node') and hasattr(data, 'center_node_idx'):
            center_id = data.center_node
            center_idx = data.center_node_idx.item()

            if hasattr(data, 'original_node_ids'):
                if center_idx >= len(data.original_node_ids):
                    issues.append(f"Center node index {center_idx} out of range")
                elif data.original_node_ids[center_idx] != center_id:
                    issues.append(f"Center node ID mismatch: {center_id} vs {data.original_node_ids[center_idx]}")

        # Check tensor dimensions
        if hasattr(data, 'x') and hasattr(data, 'original_node_ids'):
            if data.x.size(0) != len(data.original_node_ids):
                issues.append(f"Feature matrix size {data.x.size(0)} != node count {len(data.original_node_ids)}")

        return len(issues) == 0, issues


def save_id_mappings(data: Data, output_path: Path):
    """Save ID mappings to a separate JSON file for external reference"""
    mappings = {
        'metadata': {
            'project_name': getattr(data, 'project_name', 'unknown'),
            'center_node_original_id': getattr(data, 'center_node', 'unknown'),
            'center_node_tensor_index': data.center_node_idx.item() if hasattr(data, 'center_node_idx') else -1,
            'subgraph_size': getattr(data, 'subgraph_size', data.num_nodes if hasattr(data, 'num_nodes') else 0),
            'num_edges': getattr(data, 'num_edges_sg', data.num_edges if hasattr(data, 'num_edges') else 0),
            'is_smelly': data.is_smelly.item() if hasattr(data, 'is_smelly') else None
        },
        'node_mappings': {},
        'edge_mappings': []
    }

    # Save node mappings
    if hasattr(data, 'original_node_ids'):
        for i, node_id in enumerate(data.original_node_ids):
            mappings['node_mappings'][str(i)] = {
                'original_id': node_id,
                'is_center': (i == data.center_node_idx.item()) if hasattr(data, 'center_node_idx') else False
            }

            # Add features if available
            if hasattr(data, 'x') and data.x is not None:
                features = data.x[i].tolist()
                feature_names = ['fan_in', 'fan_out', 'degree_centrality', 'in_out_ratio',
                                 'pagerank', 'betweenness_centrality', 'closeness_centrality']

                if len(features) == len(feature_names):
                    mappings['node_mappings'][str(i)]['features'] = {
                        name: features[j] for j, name in enumerate(feature_names)
                    }

    # Save edge mappings
    if hasattr(data, 'edge_id_mapping'):
        mappings['edge_mappings'] = data.edge_id_mapping

    # Save to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved ID mappings to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to save ID mappings: {e}")


def load_and_reconstruct_graph(pt_file: Path, mappings_file: Optional[Path] = None) -> Tuple[Data, Dict]:
    """Load PyG data and reconstruct original graph structure with IDs"""

    # Load the data
    data = torch.load(pt_file, map_location='cpu')

    # Load mappings if provided
    mappings = {}
    if mappings_file and mappings_file.exists():
        try:
            mappings = json.loads(mappings_file.read_text(encoding='utf-8'))
        except Exception as e:
            logger.warning(f"Failed to load mappings from {mappings_file}: {e}")

    # Create networkx graph with original IDs for visualization/analysis
    import networkx as nx

    G = nx.DiGraph()

    # Add nodes with original IDs and features
    if hasattr(data, 'original_node_ids'):
        for i, node_id in enumerate(data.original_node_ids):
            attrs = {'tensor_index': i}

            # Add features
            if hasattr(data, 'x') and data.x is not None:
                features = data.x[i].tolist()
                feature_names = ['fan_in', 'fan_out', 'degree_centrality', 'in_out_ratio',
                                 'pagerank', 'betweenness_centrality', 'closeness_centrality']

                if len(features) == len(feature_names):
                    for j, name in enumerate(feature_names):
                        attrs[name] = features[j]

            # Mark center node
            if hasattr(data, 'center_node_idx') and i == data.center_node_idx.item():
                attrs['is_center'] = True
                attrs['is_smelly'] = data.is_smelly.item() if hasattr(data, 'is_smelly') else None

            G.add_node(node_id, **attrs)

    # Add edges with original IDs
    if hasattr(data, 'edge_index'):
        edge_list = data.edge_index.t().tolist()

        for i, (src_idx, tgt_idx) in enumerate(edge_list):
            # Get original IDs
            src_id = data.original_node_ids[src_idx] if hasattr(data, 'original_node_ids') else str(src_idx)
            tgt_id = data.original_node_ids[tgt_idx] if hasattr(data, 'original_node_ids') else str(tgt_idx)

            edge_attrs = {'tensor_edge_index': i}

            # Add edge features
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                if i < data.edge_attr.size(0):
                    edge_attrs['weight'] = float(data.edge_attr[i, 0])

            # Add original edge ID if available
            if hasattr(data, 'original_edge_ids') and i < len(data.original_edge_ids):
                edge_attrs['original_edge_id'] = data.original_edge_ids[i]

            G.add_edge(src_id, tgt_id, **edge_attrs)

    return data, {'networkx_graph': G, 'mappings': mappings}


def export_subgraph_info(data: Data, output_path: Path):
    """Export comprehensive subgraph information in human-readable format"""

    info = {
        'subgraph_metadata': {
            'project': getattr(data, 'project_name', 'unknown'),
            'center_node_id': getattr(data, 'center_node', 'unknown'),
            'version_hash': getattr(data, 'version_hash', 'unknown'),
            'is_smelly': data.is_smelly.item() if hasattr(data, 'is_smelly') else None,
            'subgraph_size': getattr(data, 'subgraph_size', data.num_nodes if hasattr(data, 'num_nodes') else 0),
            'num_edges': getattr(data, 'num_edges_sg', data.num_edges if hasattr(data, 'num_edges') else 0),
            'extraction_timestamp': data.extraction_timestamp.item() if hasattr(data, 'extraction_timestamp') else None
        },
        'center_node_analysis': IDMapper.get_center_node_info(data),
        'node_details': [],
        'edge_details': [],
        'validation_report': {}
    }

    # Validate mappings
    is_valid, issues = IDMapper.validate_id_mappings(data)
    info['validation_report'] = {
        'is_valid': is_valid,
        'issues': issues
    }

    # Node details
    if hasattr(data, 'original_node_ids'):
        for i, node_id in enumerate(data.original_node_ids):
            node_info = {
                'tensor_index': i,
                'original_id': node_id,
                'is_center': (i == data.center_node_idx.item()) if hasattr(data, 'center_node_idx') else False
            }

            # Add features
            if hasattr(data, 'x') and data.x is not None:
                features = data.x[i].tolist()
                feature_names = ['fan_in', 'fan_out', 'degree_centrality', 'in_out_ratio',
                                 'pagerank', 'betweenness_centrality', 'closeness_centrality']

                if len(features) == len(feature_names):
                    node_info['features'] = {name: features[j] for j, name in enumerate(feature_names)}

            info['node_details'].append(node_info)

    # Edge details
    if hasattr(data, 'edge_index'):
        edge_list = data.edge_index.t().tolist()
        for i, (src_idx, tgt_idx) in enumerate(edge_list):
            edge_info = IDMapper.get_edge_info(data, i)
            edge_info.update({
                'tensor_edge_index': i,
                'source_tensor_index': src_idx,
                'target_tensor_index': tgt_idx
            })

            # Add original node IDs
            if hasattr(data, 'original_node_ids'):
                if src_idx < len(data.original_node_ids):
                    edge_info['source_original_id'] = data.original_node_ids[src_idx]
                if tgt_idx < len(data.original_node_ids):
                    edge_info['target_original_id'] = data.original_node_ids[tgt_idx]

            info['edge_details'].append(edge_info)

    # Save to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        logger.info(f"Exported subgraph info to {output_path}")
    except Exception as e:
        logger.error(f"Failed to export subgraph info: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ID mapping utilities for hub detection data")
    parser.add_argument("command", choices=["validate", "export", "reconstruct"],
                        help="Command to execute")
    parser.add_argument("pt_file", type=Path, help="Path to .pt file")
    parser.add_argument("--output", type=Path, help="Output path for export/reconstruct")
    parser.add_argument("--mappings", type=Path, help="Path to mappings JSON file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    try:
        data = torch.load(args.pt_file, map_location='cpu')

        if args.command == "validate":
            is_valid, issues = IDMapper.validate_id_mappings(data)
            print(f"Validation result: {'✅ VALID' if is_valid else '❌ INVALID'}")
            if issues:
                print("Issues found:")
                for issue in issues:
                    print(f"  - {issue}")

        elif args.command == "export":
            output_path = args.output or args.pt_file.with_suffix('.info.json')
            export_subgraph_info(data, output_path)
            print(f"Exported info to {output_path}")

        elif args.command == "reconstruct":
            data_loaded, reconstruction = load_and_reconstruct_graph(args.pt_file, args.mappings)
            G = reconstruction['networkx_graph']
            print(f"Reconstructed graph: {len(G)} nodes, {G.number_of_edges()} edges")

            if args.output:
                import networkx as nx

                nx.write_graphml(G, args.output)
                print(f"Saved reconstructed graph to {args.output}")

    except Exception as e:
        logger.error(f"Command failed: {e}")
        raise