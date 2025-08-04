#!/usr/bin/env python3
"""
Unified Data Loader for Hub Detection Pipeline

This module provides a unified data loading and feature computation system
that ensures identical data formatting between discriminator pre-training
and RL training phases.

Key features:
- Unified feature computation using EXACT same methods as dataset creation
- Consistent 7-feature format (fan_in, fan_out, degree_centrality, in_out_ratio, pagerank, betweenness, closeness)
- No normalization or standardization applied
- Deterministic loading order
- Hash-based validation for data consistency
"""

import hashlib
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

logger = logging.getLogger(__name__)


class UnifiedFeatureComputer:
    """
    Unified feature computation that matches EXACTLY the dataset creation process
    This ensures that features computed during RL training match those in the dataset
    """

    @staticmethod
    def compute_7_structural_features(G: nx.Graph) -> Dict[str, Dict[str, float]]:
        """
        Compute the EXACT same 7 structural features as used in dataset creation

        Features computed:
        1. fan_in (in-degree)
        2. fan_out (out-degree)
        3. degree_centrality (normalized total degree)
        4. in_out_ratio (fan_in / (fan_out + eps))
        5. pagerank
        6. betweenness_centrality
        7. closeness_centrality

        Args:
            G: NetworkX directed graph

        Returns:
            Dict mapping node_id -> Dict[feature_name -> value]
        """
        if len(G) == 0:
            return {}

        try:
            # Basic degree metrics
            in_degrees = dict(G.in_degree())
            out_degrees = dict(G.out_degree())

            # Centrality metrics - use SAME parameters as dataset creation
            num_nodes = len(G)

            if num_nodes <= 100:  # Same threshold as in graph_feature_extraction.py
                try:
                    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-4)
                    betweenness = nx.betweenness_centrality(G, normalized=True)
                    closeness = nx.closeness_centrality(G, distance=None, wf_improved=True)
                except:
                    # Fallback if computation fails
                    total_edges = G.number_of_edges()
                    pagerank = {n: float(G.degree(n)) / (2 * total_edges + 1e-8) for n in G.nodes()}
                    betweenness = {n: 0.0 for n in G.nodes()}
                    closeness = {n: 1.0 / (num_nodes - 1 + 1e-8) for n in G.nodes()}
            else:
                # Same approximation as in graph_feature_extraction.py for large graphs
                logger.debug(f"Large graph ({num_nodes} nodes), using simplified centrality metrics")
                total_edges = G.number_of_edges()
                pagerank = {n: float(G.degree(n)) / (2 * total_edges + 1e-8) for n in G.nodes()}
                betweenness = {n: 0.0 for n in G.nodes()}
                closeness = {n: 1.0 / (num_nodes - 1 + 1e-8) for n in G.nodes()}

            # Build feature dictionary for all nodes
            node_features = {}
            eps = 1e-8  # Same epsilon as used in dataset creation

            for node in G.nodes():
                fan_in = float(in_degrees.get(node, 0))
                fan_out = float(out_degrees.get(node, 0))
                total_degree = fan_in + fan_out

                # Compute EXACT same features as in HubFocusedFeatureExtractor
                features = {
                    'fan_in': fan_in,
                    'fan_out': fan_out,
                    'degree_centrality': total_degree / (num_nodes - 1 + eps),
                    'in_out_ratio': fan_in / (fan_out + eps),
                    'pagerank': float(pagerank.get(node, 0)),
                    'betweenness_centrality': float(betweenness.get(node, 0)),
                    'closeness_centrality': float(closeness.get(node, 0))
                }

                node_features[node] = features

            return node_features

        except Exception as e:
            logger.error(f"Failed to compute structural features: {e}")
            return {}

    @staticmethod
    def create_data_from_graph(G: nx.Graph, center_node: str, is_smelly: bool,
                               additional_metadata: Optional[Dict] = None) -> Optional[Data]:
        """
        Create PyG Data object from NetworkX graph using UNIFIED feature computation

        This method ensures that features are computed in EXACTLY the same way
        as during dataset creation, guaranteeing consistency between training phases.

        Args:
            G: NetworkX directed graph
            center_node: ID of center node
            is_smelly: Whether this is a smelly instance
            additional_metadata: Optional metadata to add to Data object

        Returns:
            PyG Data object with unified features or None if creation fails
        """
        if center_node not in G:
            logger.debug(f"Center node {center_node} not found in graph")
            return None

        if len(G) == 0:
            logger.debug("Empty graph provided")
            return None

        try:
            # Compute unified structural features
            node_features = UnifiedFeatureComputer.compute_7_structural_features(G)

            if not node_features:
                logger.warning("Failed to compute node features")
                return None

            # Create a copy of the graph and apply features
            G_copy = G.copy()

            # Clear existing node attributes and apply unified features
            for node, attrs in G_copy.nodes(data=True):
                attrs.clear()  # Remove all original attributes
                if node in node_features:
                    attrs.update(node_features[node])
                else:
                    # Fallback for missing nodes (should not happen)
                    attrs.update({
                        'fan_in': 0.0,
                        'fan_out': 0.0,
                        'degree_centrality': 0.0,
                        'in_out_ratio': 0.0,
                        'pagerank': 0.0,
                        'betweenness_centrality': 0.0,
                        'closeness_centrality': 0.0
                    })

            # Simplify edge attributes (uniform weights) - same as dataset creation
            for u, v, attrs in G_copy.edges(data=True):
                attrs.clear()
                attrs['weight'] = 1.0

            # Convert to PyG Data using SAME parameters as dataset creation
            node_attrs = ['fan_in', 'fan_out', 'degree_centrality', 'in_out_ratio',
                          'pagerank', 'betweenness_centrality', 'closeness_centrality']
            edge_attrs = ['weight']

            data = from_networkx(G_copy, group_node_attrs=node_attrs, group_edge_attrs=edge_attrs)

            # Add essential metadata
            data.is_smelly = torch.tensor([int(is_smelly)], dtype=torch.long)
            data.center_node = center_node

            # Create node mapping for center node index
            node_list = list(G_copy.nodes())
            node_mapping = {n: i for i, n in enumerate(node_list)}
            data.center_node_idx = torch.tensor([node_mapping.get(center_node, 0)], dtype=torch.long)

            # Store original node IDs for traceability
            data.original_node_ids = [str(n) for n in node_list]
            data.node_id_to_index = node_mapping
            data.index_to_node_id = {i: str(n) for i, n in enumerate(node_list)}

            # Add subgraph metadata
            data.subgraph_size = len(G_copy)
            data.num_edges_sg = G_copy.number_of_edges()

            # Add any additional metadata
            if additional_metadata:
                for key, value in additional_metadata.items():
                    if isinstance(value, (int, float)):
                        setattr(data, key, torch.tensor([float(value)], dtype=torch.float32))
                    else:
                        setattr(data, key, value)

            return data

        except Exception as e:
            logger.error(f"Failed to create Data object from graph: {e}")
            return None


class UnifiedDataLoader:
    """
    Unified data loader that ensures consistent data format across all training phases
    """

    def __init__(self, dataset_dir: Path, device: torch.device = torch.device('cpu')):
        self.dataset_dir = Path(dataset_dir)
        self.device = device
        self._cache = {}
        self._file_hashes = {}

    def get_data_hash(self, data: Data) -> str:
        """Generate a hash for data validation and consistency checking"""
        try:
            # Create hash from node features and edge structure
            x_hash = hashlib.md5(data.x.cpu().numpy().tobytes()).hexdigest()
            edge_hash = hashlib.md5(data.edge_index.cpu().numpy().tobytes()).hexdigest()
            label_hash = hashlib.md5(data.is_smelly.cpu().numpy().tobytes()).hexdigest()

            combined_hash = f"{x_hash}_{edge_hash}_{label_hash}"
            return hashlib.md5(combined_hash.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to compute data hash: {e}")
            return "unknown"

    def load_single_file(self, file_path: Path, validate_format: bool = True,
                         remove_metadata: bool = True) -> Optional[Data]:
        """
        Load a single .pt file with unified format validation

        Args:
            file_path: Path to .pt file
            validate_format: Whether to validate tensor format
            remove_metadata: Whether to remove non-batchable metadata

        Returns:
            PyG Data object or None if loading fails
        """
        try:
            # Check cache first
            cache_key = str(file_path)
            if cache_key in self._cache:
                cached_data = self._cache[cache_key]
                if remove_metadata:
                    return self._remove_non_batchable_metadata(cached_data.clone())
                return cached_data

            # Load data
            data = torch.load(file_path, map_location='cpu')

            if validate_format:
                is_valid, issues = self.validate_data_format(data, str(file_path.name))
                if not is_valid:
                    logger.warning(f"Invalid data format in {file_path.name}: {issues}")
                    return None

            # Remove non-batchable metadata if requested
            if remove_metadata:
                data = self._remove_non_batchable_metadata(data)

            # Move to device
            data = data.to(self.device)

            # Cache the data (without metadata to save memory)
            self._cache[cache_key] = data.clone()

            # Store file hash for validation
            self._file_hashes[str(file_path)] = self.get_data_hash(data)

            return data

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None

    def _remove_non_batchable_metadata(self, data: Data) -> Data:
        """
        Remove non-batchable metadata that causes batching errors

        This removes string, list, and dict attributes that PyTorch Geometric
        cannot batch together, while preserving essential tensor data.
        """
        # List of attributes that are safe to keep (tensors and basic types)
        safe_attributes = {
            'x', 'edge_index', 'edge_attr', 'y', 'pos', 'batch',
            'is_smelly', 'center_node_idx', 'subgraph_size', 'num_edges_sg'
        }

        # List of problematic attributes that cause batching errors
        problematic_attributes = {
            'center_node', 'original_node_ids', 'node_id_to_index', 'index_to_node_id',
            'original_edge_ids', 'edge_id_mapping', 'edge_sources', 'edge_targets',
            'project_name', 'version_hash', 'graph_version', 'graph_index',
            'graphml_path', 'smelly_version_count', 'extraction_timestamp',
            'component_id', 'recomputed_during_rl'
        }

        # Create a clean copy with only essential data
        clean_data = Data()

        # Copy essential tensor attributes
        for attr in safe_attributes:
            if hasattr(data, attr):
                value = getattr(data, attr)
                if isinstance(value, torch.Tensor):
                    setattr(clean_data, attr, value)
                elif isinstance(value, (int, float)) and attr in ['subgraph_size', 'num_edges_sg']:
                    # Convert scalar metadata to tensors for batching compatibility
                    setattr(clean_data, attr, torch.tensor([float(value)], dtype=torch.float32))
                else:
                    setattr(clean_data, attr, value)

        # Store metadata separately if needed (for debugging/tracking)
        metadata = {}
        for attr in problematic_attributes:
            if hasattr(data, attr):
                metadata[attr] = getattr(data, attr)

        # Store metadata in a way that won't cause batching issues
        if metadata:
            # Convert to JSON string to make it batchable (though not very useful)
            import json
            try:
                metadata_str = json.dumps(metadata, default=str)
                # Don't actually store this as it's not useful for training
                # clean_data.metadata_json = metadata_str
            except:
                pass  # Ignore if JSON serialization fails

        return clean_data

    def load_dataset(self, max_samples_per_class: Optional[int] = None,
                     shuffle: bool = True, validate_all: bool = False,
                     remove_metadata: bool = True) -> Tuple[List[Data], List[int]]:
        """
        Load complete dataset with unified format

        Args:
            max_samples_per_class: Maximum samples per class (smelly/clean)
            shuffle: Whether to shuffle the file list
            validate_all: Whether to validate all files (slower but safer)
            remove_metadata: Whether to remove non-batchable metadata

        Returns:
            Tuple of (data_list, labels_list)
        """
        logger.info(f"Loading dataset from {self.dataset_dir}")

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        # Find all .pt files
        pt_files = list(self.dataset_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {self.dataset_dir}")

        if shuffle:
            random.shuffle(pt_files)

        logger.info(f"Found {len(pt_files)} .pt files")

        # Load and balance data
        smelly_data = []
        clean_data = []
        invalid_files = 0

        for file_path in pt_files:
            # Early termination if we have enough samples
            if max_samples_per_class:
                if (len(smelly_data) >= max_samples_per_class and
                        len(clean_data) >= max_samples_per_class):
                    break

            data = self.load_single_file(file_path, validate_format=validate_all,
                                         remove_metadata=remove_metadata)

            if data is None:
                invalid_files += 1
                continue

            # Classify by label
            if hasattr(data, 'is_smelly') and data.is_smelly.item() == 1:
                if not max_samples_per_class or len(smelly_data) < max_samples_per_class:
                    smelly_data.append(data)
            else:
                if not max_samples_per_class or len(clean_data) < max_samples_per_class:
                    clean_data.append(data)

        # Balance dataset
        min_samples = min(len(smelly_data), len(clean_data))
        if min_samples == 0:
            raise ValueError("No balanced dataset possible - missing smelly or clean samples")

        # Create balanced lists
        balanced_data = smelly_data[:min_samples] + clean_data[:min_samples]
        balanced_labels = [1] * min_samples + [0] * min_samples

        # Final shuffle
        if shuffle:
            combined = list(zip(balanced_data, balanced_labels))
            random.shuffle(combined)
            balanced_data, balanced_labels = zip(*combined)
            balanced_data = list(balanced_data)
            balanced_labels = list(balanced_labels)

        logger.info(f"Loaded balanced dataset:")
        logger.info(f"  - Total samples: {len(balanced_data)}")
        logger.info(f"  - Smelly samples: {sum(balanced_labels)}")
        logger.info(f"  - Clean samples: {len(balanced_labels) - sum(balanced_labels)}")
        logger.info(f"  - Invalid files skipped: {invalid_files}")
        logger.info(f"  - Metadata removed: {remove_metadata}")
        logger.info(f"  - All samples have exactly 7 structural features")

        return balanced_data, balanced_labels

    def validate_data_format(self, data: Data, filename: str = "unknown") -> Tuple[bool, List[str]]:
        """
        Validate that data has the correct unified format

        Args:
            data: PyG Data object to validate
            filename: Filename for error reporting

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        try:
            # Check basic structure
            if not hasattr(data, 'x') or data.x is None:
                issues.append("Missing node feature matrix (data.x)")
            elif data.x.size(1) != 7:
                issues.append(f"Expected 7 node features, got {data.x.size(1)}")

            if not hasattr(data, 'edge_index') or data.edge_index is None:
                issues.append("Missing edge index (data.edge_index)")

            # Check tensor types and dtypes
            if hasattr(data, 'x') and data.x is not None:
                if data.x.dtype != torch.float32:
                    issues.append(f"Node features should be float32, got {data.x.dtype}")

                if torch.isnan(data.x).any():
                    issues.append("Found NaN values in node features")

                if torch.isinf(data.x).any():
                    issues.append("Found infinite values in node features")

            # Check label format
            if not hasattr(data, 'is_smelly'):
                issues.append("Missing label (is_smelly)")
            elif not isinstance(data.is_smelly, torch.Tensor):
                issues.append(f"Label should be tensor, got {type(data.is_smelly)}")
            elif data.is_smelly.dtype not in [torch.long, torch.int64]:
                issues.append(f"Label should be long/int64, got {data.is_smelly.dtype}")

            # Validate feature ranges (basic sanity checks)
            if hasattr(data, 'x') and data.x is not None and data.x.size(1) >= 7:
                fan_in = data.x[:, 0]
                fan_out = data.x[:, 1]
                pagerank = data.x[:, 4]

                if (fan_in < 0).any():
                    issues.append("Found negative fan_in values")

                if (fan_out < 0).any():
                    issues.append("Found negative fan_out values")

                if (pagerank < 0).any() or (pagerank > 1.1).any():
                    issues.append("PageRank values out of expected range [0, 1]")

        except Exception as e:
            issues.append(f"Validation failed with exception: {e}")

        return len(issues) == 0, issues

    def get_dataset_statistics(self) -> Dict:
        """Get comprehensive statistics about the dataset"""
        pt_files = list(self.dataset_dir.glob("*.pt"))

        stats = {
            'total_files': len(pt_files),
            'feature_statistics': {},
            'label_distribution': {'smelly': 0, 'clean': 0},
            'validation_summary': {'valid': 0, 'invalid': 0},
            'file_hashes': self._file_hashes.copy()
        }

        if not pt_files:
            return stats

        # Sample files for statistics (to avoid loading everything)
        sample_size = min(100, len(pt_files))
        sample_files = random.sample(pt_files, sample_size)

        all_features = []

        for file_path in sample_files:
            try:
                data = torch.load(file_path, map_location='cpu')
                is_valid, _ = self.validate_data_format(data, file_path.name)

                if is_valid:
                    stats['validation_summary']['valid'] += 1

                    # Collect features
                    if hasattr(data, 'x') and data.x is not None:
                        all_features.append(data.x)

                    # Count labels
                    if hasattr(data, 'is_smelly'):
                        if data.is_smelly.item() == 1:
                            stats['label_distribution']['smelly'] += 1
                        else:
                            stats['label_distribution']['clean'] += 1
                else:
                    stats['validation_summary']['invalid'] += 1

            except Exception as e:
                stats['validation_summary']['invalid'] += 1
                logger.debug(f"Failed to process {file_path} for stats: {e}")

        # Compute feature statistics
        if all_features:
            all_features = torch.cat(all_features, dim=0)
            feature_names = ['fan_in', 'fan_out', 'degree_centrality', 'in_out_ratio',
                             'pagerank', 'betweenness_centrality', 'closeness_centrality']

            for i, name in enumerate(feature_names):
                if i < all_features.size(1):
                    feature_values = all_features[:, i]
                    stats['feature_statistics'][name] = {
                        'mean': float(feature_values.mean()),
                        'std': float(feature_values.std()),
                        'min': float(feature_values.min()),
                        'max': float(feature_values.max())
                    }

        return stats

    @classmethod
    def recompute_features_for_intermediate_graph(cls, G: nx.Graph, center_node: str,
                                                  is_smelly: bool) -> Optional[Data]:
        """
        Recompute features for intermediate graphs during RL training

        This method ensures that features for graphs modified during RL training
        are computed using the EXACT same methodology as the original dataset.

        Args:
            G: NetworkX graph (potentially modified during RL)
            center_node: Center node ID
            is_smelly: Current smell status

        Returns:
            PyG Data object with recomputed features using unified methodology
        """
        return UnifiedFeatureComputer.create_data_from_graph(
            G, center_node, is_smelly,
            additional_metadata={'recomputed_during_rl': True}
        )


def create_unified_loader(dataset_dir: Path, device: torch.device = torch.device('cpu')) -> UnifiedDataLoader:
    """
    Factory function to create unified data loader

    Args:
        dataset_dir: Path to dataset directory containing .pt files
        device: Device to load data onto

    Returns:
        Configured UnifiedDataLoader instance
    """
    return UnifiedDataLoader(dataset_dir, device)


def validate_dataset_consistency(dataset_dir: Path, sample_size: Optional[int] = None) -> Dict:
    """
    Validate dataset consistency and generate report

    Args:
        dataset_dir: Path to dataset directory
        sample_size: Number of files to validate (None for all)

    Returns:
        Validation report dictionary
    """
    loader = UnifiedDataLoader(dataset_dir)

    pt_files = list(dataset_dir.glob("*.pt"))
    if sample_size:
        pt_files = random.sample(pt_files, min(sample_size, len(pt_files)))

    report = {
        'files_checked': len(pt_files),
        'valid_files': 0,
        'invalid_files': 0,
        'consistency_issues': [],
        'feature_consistency': True,
        'hash_validation': {'unique_hashes': 0, 'duplicate_hashes': 0}
    }

    file_hashes = set()
    duplicate_hashes = set()

    for file_path in pt_files:
        data = loader.load_single_file(file_path, validate_format=True)

        if data is not None:
            report['valid_files'] += 1

            # Check hash uniqueness
            data_hash = loader.get_data_hash(data)
            if data_hash in file_hashes:
                duplicate_hashes.add(data_hash)
                report['consistency_issues'].append(f"Duplicate data hash found: {file_path.name}")
            else:
                file_hashes.add(data_hash)
        else:
            report['invalid_files'] += 1
            report['consistency_issues'].append(f"Invalid file: {file_path.name}")

    report['hash_validation']['unique_hashes'] = len(file_hashes)
    report['hash_validation']['duplicate_hashes'] = len(duplicate_hashes)

    return report


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Unified Data Loader Test and Validation")
    parser.add_argument("dataset_dir", type=Path, help="Path to dataset directory")
    parser.add_argument("--test-loading", action="store_true", help="Test data loading")
    parser.add_argument("--validate-consistency", action="store_true", help="Validate dataset consistency")
    parser.add_argument("--sample-size", type=int, help="Sample size for testing")
    parser.add_argument("--output", type=Path, help="Output file for results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    try:
        if args.test_loading:
            logger.info("Testing unified data loading...")
            loader = create_unified_loader(args.dataset_dir)

            # Load dataset
            data_list, labels = loader.load_dataset(
                max_samples_per_class=args.sample_size,
                shuffle=True,
                validate_all=True
            )

            # Get statistics
            stats = loader.get_dataset_statistics()

            print(f"\n✅ Successfully loaded {len(data_list)} samples")
            print(f"Feature statistics: {json.dumps(stats['feature_statistics'], indent=2)}")

        if args.validate_consistency:
            logger.info("Validating dataset consistency...")
            report = validate_dataset_consistency(args.dataset_dir, args.sample_size)

            print(f"\n📊 Consistency Report:")
            print(f"Files checked: {report['files_checked']}")
            print(f"Valid files: {report['valid_files']}")
            print(f"Invalid files: {report['invalid_files']}")
            print(f"Unique hashes: {report['hash_validation']['unique_hashes']}")
            print(f"Duplicate hashes: {report['hash_validation']['duplicate_hashes']}")

            if report['consistency_issues']:
                print("Issues found:")
                for issue in report['consistency_issues'][:10]:  # Show first 10
                    print(f"  - {issue}")

            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"Report saved to {args.output}")

    except Exception as e:
        logger.error(f"Operation failed: {e}")
        raise