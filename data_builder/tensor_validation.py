#!/usr/bin/env python3
"""
Tensor Validation Utilities for Discriminator Compatibility

Ensures that all saved .pt files contain properly formatted tensors
for graph neural network discriminator training.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class TensorValidator:
    """Validates tensor format and content for discriminator compatibility"""

    EXPECTED_NODE_FEATURES = 7  # fan_in, fan_out, degree_centrality, in_out_ratio, pagerank, betweenness, closeness
    EXPECTED_EDGE_FEATURES = 1  # weight

    @staticmethod
    def validate_data_object(data: Data, filename: str = "unknown") -> Tuple[bool, List[str]]:
        """
        Comprehensive validation of PyG Data object for discriminator compatibility

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        try:
            # 1. Check basic structure
            if not hasattr(data, 'x') or data.x is None:
                issues.append("Missing node feature matrix (data.x)")
                return False, issues

            if not hasattr(data, 'edge_index') or data.edge_index is None:
                issues.append("Missing edge index (data.edge_index)")
                return False, issues

            # 2. Validate tensor types
            if not isinstance(data.x, torch.Tensor):
                issues.append(f"Node features not a tensor: {type(data.x)}")

            if not isinstance(data.edge_index, torch.Tensor):
                issues.append(f"Edge index not a tensor: {type(data.edge_index)}")

            # 3. Check tensor dimensions
            if data.x.dim() != 2:
                issues.append(f"Node features should be 2D, got {data.x.dim()}D")

            if data.edge_index.dim() != 2:
                issues.append(f"Edge index should be 2D, got {data.edge_index.dim()}D")

            # 4. Check feature dimensions
            if data.x.size(1) != TensorValidator.EXPECTED_NODE_FEATURES:
                issues.append(f"Expected {TensorValidator.EXPECTED_NODE_FEATURES} node features, got {data.x.size(1)}")

            if data.edge_index.size(0) != 2:
                issues.append(f"Edge index should have 2 rows, got {data.edge_index.size(0)}")

            # 5. Check edge attributes if present
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                if not isinstance(data.edge_attr, torch.Tensor):
                    issues.append(f"Edge attributes not a tensor: {type(data.edge_attr)}")
                elif data.edge_attr.dim() != 2:
                    issues.append(f"Edge attributes should be 2D, got {data.edge_attr.dim()}D")
                elif data.edge_attr.size(1) != TensorValidator.EXPECTED_EDGE_FEATURES:
                    issues.append(
                        f"Expected {TensorValidator.EXPECTED_EDGE_FEATURES} edge features, got {data.edge_attr.size(1)}")

            # 6. Check for NaN/Inf values
            if torch.isnan(data.x).any():
                issues.append("Found NaN values in node features")

            if torch.isinf(data.x).any():
                issues.append("Found infinite values in node features")

            # 7. Check label format
            if hasattr(data, 'is_smelly'):
                if not isinstance(data.is_smelly, torch.Tensor):
                    issues.append(f"Label not a tensor: {type(data.is_smelly)}")
                elif data.is_smelly.dim() != 1:
                    issues.append(f"Label should be 1D, got {data.is_smelly.dim()}D")
                elif data.is_smelly.size(0) != 1:
                    issues.append(f"Label should have size 1, got {data.is_smelly.size(0)}")
                elif data.is_smelly.dtype not in [torch.long, torch.int64]:
                    issues.append(f"Label should be long/int64, got {data.is_smelly.dtype}")
            else:
                issues.append("Missing label (is_smelly)")

            # 8. Check center node index
            if hasattr(data, 'center_node_idx'):
                if not isinstance(data.center_node_idx, torch.Tensor):
                    issues.append(f"Center node index not a tensor: {type(data.center_node_idx)}")
                elif data.center_node_idx.item() >= data.x.size(0):
                    issues.append(
                        f"Center node index {data.center_node_idx.item()} out of range [0, {data.x.size(0) - 1}]")

            # 9. Check tensor dtypes for discriminator compatibility
            if data.x.dtype != torch.float32:
                issues.append(f"Node features should be float32, got {data.x.dtype}")

            if data.edge_index.dtype not in [torch.long, torch.int64]:
                issues.append(f"Edge index should be long/int64, got {data.edge_index.dtype}")

            # 10. Validate feature ranges (basic sanity checks)
            # Fan-in and fan-out should be non-negative
            fan_in = data.x[:, 0]
            fan_out = data.x[:, 1]

            if (fan_in < 0).any():
                issues.append("Found negative fan_in values")

            if (fan_out < 0).any():
                issues.append("Found negative fan_out values")

            # PageRank should be in [0, 1]
            if data.x.size(1) >= 5:  # pagerank is column 4
                pagerank = data.x[:, 4]
                if (pagerank < 0).any() or (pagerank > 1.1).any():  # Small tolerance
                    issues.append("PageRank values out of expected range [0, 1]")

        except Exception as e:
            issues.append(f"Validation failed with exception: {e}")

        return len(issues) == 0, issues

    @staticmethod
    def get_tensor_info(data: Data) -> Dict:
        """Get comprehensive tensor information for debugging"""
        info = {
            'node_features': {},
            'edge_info': {},
            'label_info': {},
            'metadata': {}
        }

        try:
            # Node features info
            if hasattr(data, 'x') and data.x is not None:
                info['node_features'] = {
                    'shape': list(data.x.shape),
                    'dtype': str(data.x.dtype),
                    'device': str(data.x.device),
                    'requires_grad': data.x.requires_grad,
                    'memory_usage_mb': data.x.numel() * data.x.element_size() / (1024 * 1024),
                    'value_range': {
                        'min': float(data.x.min()),
                        'max': float(data.x.max()),
                        'mean': float(data.x.mean()),
                        'std': float(data.x.std())
                    }
                }

                # Individual feature statistics
                feature_names = ['fan_in', 'fan_out', 'degree_centrality', 'in_out_ratio',
                                 'pagerank', 'betweenness_centrality', 'closeness_centrality']

                if data.x.size(1) == len(feature_names):
                    info['node_features']['per_feature_stats'] = {}
                    for i, name in enumerate(feature_names):
                        feat_values = data.x[:, i]
                        info['node_features']['per_feature_stats'][name] = {
                            'min': float(feat_values.min()),
                            'max': float(feat_values.max()),
                            'mean': float(feat_values.mean()),
                            'std': float(feat_values.std())
                        }

            # Edge info
            if hasattr(data, 'edge_index') and data.edge_index is not None:
                info['edge_info'] = {
                    'shape': list(data.edge_index.shape),
                    'dtype': str(data.edge_index.dtype),
                    'num_edges': data.edge_index.size(1),
                    'memory_usage_mb': data.edge_index.numel() * data.edge_index.element_size() / (1024 * 1024)
                }

                if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                    info['edge_info']['edge_attr'] = {
                        'shape': list(data.edge_attr.shape),
                        'dtype': str(data.edge_attr.dtype),
                        'value_range': {
                            'min': float(data.edge_attr.min()),
                            'max': float(data.edge_attr.max()),
                            'mean': float(data.edge_attr.mean())
                        }
                    }

            # Label info
            if hasattr(data, 'is_smelly') and data.is_smelly is not None:
                info['label_info'] = {
                    'shape': list(data.is_smelly.shape),
                    'dtype': str(data.is_smelly.dtype),
                    'value': int(data.is_smelly.item()),
                    'label_name': 'smelly' if data.is_smelly.item() else 'clean'
                }

            # Metadata
            info['metadata'] = {
                'total_memory_mb': sum([
                    data.x.numel() * data.x.element_size() if hasattr(data, 'x') and data.x is not None else 0,
                    data.edge_index.numel() * data.edge_index.element_size() if hasattr(data,
                                                                                        'edge_index') and data.edge_index is not None else 0,
                    data.edge_attr.numel() * data.edge_attr.element_size() if hasattr(data,
                                                                                      'edge_attr') and data.edge_attr is not None else 0
                ]) / (1024 * 1024),
                'has_id_mapping': hasattr(data, 'original_node_ids'),
                'subgraph_size': getattr(data, 'subgraph_size', 'unknown'),
                'project_name': getattr(data, 'project_name', 'unknown'),
                'center_node': getattr(data, 'center_node', 'unknown')
            }

        except Exception as e:
            info['error'] = str(e)

        return info


def validate_dataset_tensors(dataset_dir: Path, sample_size: Optional[int] = None) -> Dict:
    """
    Validate all .pt files in dataset directory for discriminator compatibility

    Args:
        dataset_dir: Directory containing .pt files
        sample_size: If specified, validate only a random sample of files

    Returns:
        Validation report dictionary
    """
    pt_files = list(dataset_dir.glob("*.pt"))

    if not pt_files:
        return {'error': 'No .pt files found in directory'}

    # Sample files if requested
    if sample_size and len(pt_files) > sample_size:
        import random
        pt_files = random.sample(pt_files, sample_size)

    report = {
        'total_files': len(pt_files),
        'valid_files': 0,
        'invalid_files': 0,
        'validation_errors': {},
        'tensor_stats': {
            'node_features': {'shapes': {}, 'dtypes': {}},
            'edge_features': {'shapes': {}, 'dtypes': {}},
            'labels': {'distribution': {'smelly': 0, 'clean': 0}}
        },
        'memory_usage': {'total_mb': 0, 'avg_mb_per_file': 0}
    }

    validator = TensorValidator()
    total_memory = 0

    for pt_file in pt_files:
        try:
            # Load data
            data = torch.load(pt_file, map_location='cpu')

            # Validate
            is_valid, issues = validator.validate_data_object(data, pt_file.name)

            if is_valid:
                report['valid_files'] += 1
            else:
                report['invalid_files'] += 1
                report['validation_errors'][pt_file.name] = issues

            # Collect tensor stats
            tensor_info = validator.get_tensor_info(data)

            # Node feature stats
            if 'node_features' in tensor_info and 'shape' in tensor_info['node_features']:
                shape_key = str(tensor_info['node_features']['shape'])
                dtype_key = tensor_info['node_features']['dtype']

                report['tensor_stats']['node_features']['shapes'][shape_key] = \
                    report['tensor_stats']['node_features']['shapes'].get(shape_key, 0) + 1
                report['tensor_stats']['node_features']['dtypes'][dtype_key] = \
                    report['tensor_stats']['node_features']['dtypes'].get(dtype_key, 0) + 1

            # Label distribution
            if 'label_info' in tensor_info and 'label_name' in tensor_info['label_info']:
                label_name = tensor_info['label_info']['label_name']
                report['tensor_stats']['labels']['distribution'][label_name] += 1

            # Memory usage
            if 'metadata' in tensor_info:
                file_memory = tensor_info['metadata'].get('total_memory_mb', 0)
                total_memory += file_memory

        except Exception as e:
            report['invalid_files'] += 1
            report['validation_errors'][pt_file.name] = [f"Failed to load: {e}"]

    # Final statistics
    report['memory_usage']['total_mb'] = total_memory
    report['memory_usage']['avg_mb_per_file'] = total_memory / len(pt_files) if pt_files else 0
    report['validation_success_rate'] = report['valid_files'] / len(pt_files) if pt_files else 0

    return report


def fix_tensor_format(data: Data) -> Data:
    """
    Fix common tensor format issues for discriminator compatibility

    Args:
        data: PyG Data object to fix

    Returns:
        Fixed PyG Data object
    """
    fixed_data = data.clone()

    try:
        # Fix node features dtype
        if hasattr(fixed_data, 'x') and fixed_data.x is not None:
            if fixed_data.x.dtype != torch.float32:
                fixed_data.x = fixed_data.x.float()

        # Fix edge index dtype
        if hasattr(fixed_data, 'edge_index') and fixed_data.edge_index is not None:
            if fixed_data.edge_index.dtype not in [torch.long, torch.int64]:
                fixed_data.edge_index = fixed_data.edge_index.long()

        # Fix edge attributes dtype
        if hasattr(fixed_data, 'edge_attr') and fixed_data.edge_attr is not None:
            if fixed_data.edge_attr.dtype != torch.float32:
                fixed_data.edge_attr = fixed_data.edge_attr.float()

        # Fix label dtype
        if hasattr(fixed_data, 'is_smelly') and fixed_data.is_smelly is not None:
            if fixed_data.is_smelly.dtype not in [torch.long, torch.int64]:
                fixed_data.is_smelly = fixed_data.is_smelly.long()

        # Replace NaN values with zeros
        if hasattr(fixed_data, 'x') and fixed_data.x is not None:
            fixed_data.x = torch.nan_to_num(fixed_data.x, nan=0.0, posinf=1e6, neginf=-1e6)

        # Ensure edge attributes exist with proper format
        if hasattr(fixed_data, 'edge_index') and fixed_data.edge_index is not None:
            if not hasattr(fixed_data, 'edge_attr') or fixed_data.edge_attr is None:
                num_edges = fixed_data.edge_index.size(1)
                fixed_data.edge_attr = torch.ones(num_edges, 1, dtype=torch.float32)

    except Exception as e:
        logger.warning(f"Failed to fix tensor format: {e}")
        return data  # Return original if fixing fails

    return fixed_data


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Validate tensor format for discriminator compatibility")
    parser.add_argument("dataset_dir", type=Path, help="Directory containing .pt files")
    parser.add_argument("--sample-size", type=int, help="Validate only a random sample of files")
    parser.add_argument("--output", type=Path, help="Save validation report to JSON file")
    parser.add_argument("--fix-files", action="store_true", help="Fix tensor format issues in place")
    parser.add_argument("--verbose", action="store_true", help="Show detailed validation info")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    try:
        # Run validation
        logger.info(f"Validating tensors in {args.dataset_dir}")
        report = validate_dataset_tensors(args.dataset_dir, args.sample_size)

        # Print summary
        print(f"\n📊 Tensor Validation Report")
        print(f"Total files: {report['total_files']}")
        print(f"Valid files: {report['valid_files']}")
        print(f"Invalid files: {report['invalid_files']}")
        print(f"Success rate: {report['validation_success_rate']:.2%}")
        print(f"Total memory usage: {report['memory_usage']['total_mb']:.2f} MB")
        print(f"Average per file: {report['memory_usage']['avg_mb_per_file']:.2f} MB")

        # Show label distribution
        label_dist = report['tensor_stats']['labels']['distribution']
        print(f"\nLabel distribution:")
        for label, count in label_dist.items():
            print(f"  {label}: {count}")

        # Show validation errors if any
        if report['validation_errors'] and args.verbose:
            print(f"\n❌ Validation errors:")
            for filename, issues in report['validation_errors'].items():
                print(f"  {filename}:")
                for issue in issues:
                    print(f"    - {issue}")

        # Fix files if requested
        if args.fix_files and report['invalid_files'] > 0:
            logger.info("Fixing tensor format issues...")
            fixed_count = 0

            for filename in report['validation_errors'].keys():
                filepath = args.dataset_dir / filename
                try:
                    data = torch.load(filepath, map_location='cpu')
                    fixed_data = fix_tensor_format(data)

                    # Validate the fix
                    validator = TensorValidator()
                    is_valid, _ = validator.validate_data_object(fixed_data, filename)

                    if is_valid:
                        torch.save(fixed_data, filepath)
                        fixed_count += 1
                        logger.info(f"Fixed {filename}")

                except Exception as e:
                    logger.error(f"Failed to fix {filename}: {e}")

            print(f"\n✅ Fixed {fixed_count} files")

        # Save report if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Report saved to {args.output}")

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        exit(1)