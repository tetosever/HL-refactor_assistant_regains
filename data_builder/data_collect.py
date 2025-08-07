#!/usr/bin/env python3
"""
Clean Dataset Creation Pipeline for Hub Detection

Refactored version maintaining exact logic but with cleaner structure.
Creates 1-hop ego graphs from larger dependency graphs for GCN/RL training.
"""

import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import hashlib

import networkx as nx
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

logger = logging.getLogger(__name__)

# Constants
EXCLUDED_CONSTRUCT_TYPES = {"PACKAGE"}
GRAPH_VERSION_PATTERN = re.compile(r"dependency-graph-(\d+)_([0-9a-fA-F]+)\.graphml")
HUB_FEATURES = [
    'fan_in', 'fan_out', 'degree_centrality', 'in_out_ratio',
    'pagerank', 'betweenness_centrality', 'closeness_centrality'
]


class SmellMapBuilder:
    """Builds component-to-smell mappings from Arcan output"""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.chars_file = project_dir / "smell-characteristics.csv"
        self.affects_file = project_dir / "smell-affects.csv"

    def validate_files(self) -> Tuple[bool, List[str]]:
        """Validate required CSV files exist with proper structure"""
        required_files = {
            "smell-characteristics.csv": ["vertexId", "versionId", "AffectedConstructType"],
            "smell-affects.csv": ["fromId", "toId", "versionId"]
        }

        errors = []
        for filename, required_cols in required_files.items():
            filepath = self.project_dir / filename

            if not filepath.exists():
                errors.append(f"Missing file: {filename}")
                continue

            try:
                df_sample = pd.read_csv(filepath, nrows=5)
                missing_cols = set(required_cols) - set(df_sample.columns)
                if missing_cols:
                    errors.append(f"File {filename} missing columns: {missing_cols}")
            except Exception as e:
                errors.append(f"Cannot read {filename}: {e}")

        return len(errors) == 0, errors

    def build_smell_map(self) -> Tuple[Dict[str, List[str]], Dict]:
        """Build component -> smelly_versions mapping"""
        # Validate input files
        is_valid, errors = self.validate_files()
        if not is_valid:
            raise ValueError(f"Input validation failed: {'; '.join(errors)}")

        # Load CSV files
        try:
            df_chars = pd.read_csv(self.chars_file)
            df_affects = pd.read_csv(self.affects_file)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV files: {e}")

        # Filter characteristics (exclude PACKAGE constructs)
        valid_constructs = df_chars[~df_chars["AffectedConstructType"].isin(EXCLUDED_CONSTRUCT_TYPES)].copy()

        if valid_constructs.empty:
            logger.warning("No valid smell components after filtering")
            return {}, {}

        # Prepare for join: rename columns to match
        valid_constructs = valid_constructs.rename(columns={"vertexId": "smellId"})
        df_affects_renamed = df_affects.rename(columns={"fromId": "smellId"})

        # Join on smellId and versionId
        joined = df_affects_renamed.merge(valid_constructs, on=["smellId", "versionId"], how="inner")

        if joined.empty:
            logger.warning("No records after join - check data consistency")
            return {}, {}

        # Build component -> versions mapping
        comp_to_versions = defaultdict(set)
        for _, row in joined.iterrows():
            comp_id = str(row["toId"])  # Component affected by smell
            version_id = str(row["versionId"])  # Version where it's affected
            comp_to_versions[comp_id].add(version_id)

        # Convert to sorted lists
        result = {comp: sorted(versions) for comp, versions in comp_to_versions.items()}

        # Statistics
        stats = {
            'total_affected_components': len(result),
            'total_smell_instances': sum(len(v) for v in result.values()),
            'avg_versions_per_component': sum(len(v) for v in result.values()) / len(result) if result else 0,
            'unique_smell_hubs': joined['smellId'].nunique(),
            'joined_records': len(joined)
        }

        logger.info(f"Built smell map: {stats['total_affected_components']} components, "
                    f"{stats['total_smell_instances']} smell instances")

        return result, stats


class HubFeatureExtractor:
    """Extracts hub-focused features from network graphs"""

    @staticmethod
    def compute_centrality_metrics(G: nx.Graph) -> Tuple[Dict, Dict, Dict]:
        """Compute centrality metrics efficiently based on graph size"""
        if len(G) <= 100:
            pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-4)
            betweenness = nx.betweenness_centrality(G, normalized=True)
            closeness = nx.closeness_centrality(G, distance=None, wf_improved=True)
        else:
            # Simplified metrics for large graphs
            total_edges = G.number_of_edges()
            pagerank = {n: float(G.degree(n)) / (2 * total_edges + 1e-8) for n in G.nodes()}
            betweenness = {n: 0.0 for n in G.nodes()}
            closeness = {n: 1.0 / (len(G) - 1 + 1e-8) for n in G.nodes()}

        return pagerank, betweenness, closeness

    @staticmethod
    def compute_node_features(G: nx.Graph) -> Dict[str, Dict[str, float]]:
        """Compute hub detection features for all nodes"""
        # Basic degree metrics
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())

        # Centrality metrics
        pagerank, betweenness, closeness = HubFeatureExtractor.compute_centrality_metrics(G)

        # Build feature dict for all nodes
        node_features = {}
        for node in G.nodes():
            fan_in = float(in_degrees.get(node, 0))
            fan_out = float(out_degrees.get(node, 0))
            total_degree = fan_in + fan_out

            node_features[str(node)] = {
                'fan_in': fan_in,
                'fan_out': fan_out,
                'degree_centrality': total_degree / (len(G) - 1 + 1e-8),
                'in_out_ratio': fan_in / (fan_out + 1e-8),
                'pagerank': float(pagerank.get(node, 0)),
                'betweenness_centrality': float(betweenness.get(node, 0)),
                'closeness_centrality': float(closeness.get(node, 0))
            }

        return node_features


class SubgraphExtractor:
    """Extracts and processes 1-hop ego subgraphs"""

    def __init__(self, config: Dict):
        self.config = config

    def extract_ego_graph(self, G: nx.Graph, center_node: str) -> Optional[nx.Graph]:
        """Extract 1-hop ego graph with validation"""
        if center_node not in G:
            return None

        # Check if center has neighbors
        if len(list(G.neighbors(center_node))) == 0:
            return None

        # Extract ego graph
        ego_graph = nx.ego_graph(G, center_node, radius=1, undirected=False)

        # Size validation
        min_size = self.config.get('min_subgraph_size', 3)
        max_size = self.config.get('max_subgraph_size', 200)

        if len(ego_graph) < min_size or len(ego_graph) > max_size:
            return None

        # Remove isolated nodes if configured
        if self.config.get('remove_isolated_nodes', True):
            isolated = list(nx.isolates(ego_graph))
            if isolated:
                ego_graph.remove_nodes_from(isolated)
                if len(ego_graph) < min_size:
                    return None

        # Check minimum edges
        min_edges = self.config.get('min_edges', 1)
        if ego_graph.number_of_edges() < min_edges:
            return None

        return ego_graph

    def create_pyg_data(self, ego_graph: nx.Graph, center_node: str, is_smelly: bool) -> Optional[Data]:
        """Convert NetworkX graph to PyTorch Geometric Data object"""
        try:
            # Compute and apply node features
            node_features = HubFeatureExtractor.compute_node_features(ego_graph)

            # Clear existing attributes and apply new features
            for node, attrs in ego_graph.nodes(data=True):
                attrs.clear()
                node_id = str(node)
                if node_id in node_features:
                    attrs.update(node_features[node_id])
                else:
                    # Fallback features
                    attrs.update({feat: 0.0 for feat in HUB_FEATURES})

            # Clear edge attributes (keep only connectivity)
            for u, v, attrs in ego_graph.edges(data=True):
                attrs.clear()

            # Convert to PyG Data
            data = from_networkx(ego_graph, group_node_attrs=HUB_FEATURES)

            # Add metadata
            node_list = list(ego_graph.nodes())
            node_mapping = {str(n): i for i, n in enumerate(node_list)}

            data.is_smelly = torch.tensor([int(is_smelly)], dtype=torch.long)
            data.center_node = center_node
            data.center_node_idx = torch.tensor([node_mapping.get(center_node, 0)], dtype=torch.long)
            data.original_node_ids = [str(n) for n in node_list]
            data.subgraph_size = len(ego_graph)
            data.num_edges_sg = ego_graph.number_of_edges()

            return data

        except Exception as e:
            logger.warning(f"Failed to create PyG data for {center_node}: {e}")
            return None


class DatasetBuilder:
    """Main dataset building pipeline"""

    def __init__(self, arcan_dir: Path, output_dir: Path, config: Dict):
        self.arcan_dir = arcan_dir
        self.output_dir = output_dir
        self.config = config
        self.subgraph_extractor = SubgraphExtractor(config)

    def load_or_build_smell_map(self, project: str) -> Dict[str, List[str]]:
        """Load existing smell map or build new one"""
        project_dir = self.arcan_dir / project
        cache_file = project_dir / "smell_map.json"

        if cache_file.exists():
            try:
                smell_map = json.loads(cache_file.read_text())
                logger.info(f"Loaded cached smell map for {project}: {len(smell_map)} components")
                return smell_map
            except Exception as e:
                logger.warning(f"Failed to load cached smell map: {e}")

        # Build new smell map
        builder = SmellMapBuilder(project_dir)
        smell_map, stats = builder.build_smell_map()

        # Save to cache
        try:
            with open(cache_file, 'w') as f:
                json.dump(smell_map, f, indent=2)
            logger.info(f"Saved smell map to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save smell map: {e}")

        return smell_map

    def process_graphml_file(self, graphml_path: Path, smell_map: Dict[str, List[str]]) -> List[Data]:
        """Process single GraphML file and extract subgraphs"""
        # Parse version info from filename
        match = GRAPH_VERSION_PATTERN.match(graphml_path.name)
        if not match:
            logger.warning(f"Cannot parse version from {graphml_path.name}")
            return []

        version_index, version_hash = match.groups()

        # Load graph
        try:
            G = nx.read_graphml(graphml_path)
        except Exception as e:
            logger.warning(f"Failed to load {graphml_path}: {e}")
            return []

        # Check graph size limits
        max_size = self.config.get('max_graph_size', 10000)
        if len(G) > max_size:
            logger.warning(f"Graph {graphml_path.name} too large ({len(G)} nodes), skipping")
            return []

        logger.debug(f"Processing {graphml_path.name}: {len(G)} nodes, {G.number_of_edges()} edges")

        extracted_data = []

        # Process each component in smell map
        for comp_id, smelly_versions in smell_map.items():
            if comp_id not in G:
                continue

            is_smelly = version_hash in smelly_versions

            # Skip non-smelly if not configured to include them
            if not is_smelly and not self.config.get('include_non_smelly', False):
                continue

            # Extract subgraph
            ego_graph = self.subgraph_extractor.extract_ego_graph(G, comp_id)
            if ego_graph is None:
                continue

            # Create PyG data object
            data = self.subgraph_extractor.create_pyg_data(ego_graph, comp_id, is_smelly)
            if data is None:
                continue

            # Add version metadata
            data.version_index = int(version_index)
            data.version_hash = version_hash
            data.graphml_path = str(graphml_path)

            extracted_data.append(data)

        logger.debug(f"Extracted {len(extracted_data)} subgraphs from {graphml_path.name}")
        return extracted_data

    def process_project(self, project: str) -> Dict:
        """Process entire project"""
        start_time = time.time()
        stats = {
            'project': project,
            'success': False,
            'processing_time': 0,
            'graphml_files_processed': 0,
            'total_subgraphs': 0,
            'smelly_count': 0,
            'clean_count': 0,
            'error': None
        }

        try:
            logger.info(f"Processing project: {project}")
            project_dir = self.arcan_dir / project

            if not project_dir.exists():
                stats['error'] = f"Project directory not found: {project_dir}"
                return stats

            # Load smell map
            smell_map = self.load_or_build_smell_map(project)
            if not smell_map:
                stats['error'] = "Empty or invalid smell map"
                return stats

            # Find GraphML files
            graphml_files = list(project_dir.glob("dependency-graph-*.graphml"))
            if not graphml_files:
                stats['error'] = "No GraphML files found"
                return stats

            logger.info(f"Found {len(graphml_files)} GraphML files to process")

            # Process each GraphML file
            total_subgraphs = 0
            smelly_count = 0
            clean_count = 0

            for graphml_file in graphml_files:
                extracted_data = self.process_graphml_file(graphml_file, smell_map)

                # Save extracted subgraphs
                for data in extracted_data:
                    # Generate filename
                    filename = (f"{project}_{data.center_node}_{total_subgraphs:04d}_"
                                f"{data.version_index:04d}_{data.version_hash[:8]}.pt")
                    output_path = self.output_dir / filename

                    # Add project metadata
                    data.project_name = project
                    data.extraction_timestamp = torch.tensor([int(time.time())], dtype=torch.long)

                    # Save
                    torch.save(data, output_path)
                    total_subgraphs += 1

                    if data.is_smelly.item():
                        smelly_count += 1
                    else:
                        clean_count += 1

                stats['graphml_files_processed'] += 1

            # Update statistics
            stats.update({
                'success': True,
                'total_subgraphs': total_subgraphs,
                'smelly_count': smelly_count,
                'clean_count': clean_count
            })

            logger.info(f"Completed {project}: {total_subgraphs} subgraphs "
                        f"({smelly_count} smelly, {clean_count} clean)")

        except Exception as e:
            stats['error'] = str(e)
            logger.error(f"Error processing {project}: {e}")

        finally:
            stats['processing_time'] = time.time() - start_time

        return stats

    def build_dataset(self, projects: List[str]) -> Dict:
        """Build complete dataset for multiple projects"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        pipeline_stats = {
            'total_projects': len(projects),
            'successful_projects': 0,
            'failed_projects': 0,
            'total_subgraphs': 0,
            'total_smelly': 0,
            'total_clean': 0,
            'project_results': []
        }

        logger.info(f"Starting dataset building for {len(projects)} projects")

        for project in projects:
            project_stats = self.process_project(project)
            pipeline_stats['project_results'].append(project_stats)

            if project_stats['success']:
                pipeline_stats['successful_projects'] += 1
                pipeline_stats['total_subgraphs'] += project_stats['total_subgraphs']
                pipeline_stats['total_smelly'] += project_stats['smelly_count']
                pipeline_stats['total_clean'] += project_stats['clean_count']
            else:
                pipeline_stats['failed_projects'] += 1
                logger.error(f"Project {project} failed: {project_stats['error']}")

        # Save pipeline statistics
        stats_file = self.output_dir.parent / "dataset_build_stats.json"
        try:
            with open(stats_file, 'w') as f:
                json.dump(pipeline_stats, f, indent=2, default=str)
            logger.info(f"Saved pipeline statistics to {stats_file}")
        except Exception as e:
            logger.warning(f"Failed to save statistics: {e}")

        logger.info(f"Dataset building completed:")
        logger.info(
            f"  - Successful projects: {pipeline_stats['successful_projects']}/{pipeline_stats['total_projects']}")
        logger.info(f"  - Total subgraphs: {pipeline_stats['total_subgraphs']}")
        logger.info(f"  - Smelly samples: {pipeline_stats['total_smelly']}")
        logger.info(f"  - Clean samples: {pipeline_stats['total_clean']}")

        return pipeline_stats


def validate_dataset(dataset_dir: Path) -> Dict:
    """Validate generated dataset"""
    pt_files = list(dataset_dir.glob("*.pt"))

    stats = {
        'total_files': len(pt_files),
        'valid_files': 0,
        'invalid_files': 0,
        'feature_stats': {'shapes': {}, 'dtypes': {}},
        'label_distribution': {'smelly': 0, 'clean': 0},
        'validation_errors': []
    }

    for pt_file in pt_files:
        try:
            data = torch.load(pt_file, map_location='cpu')

            # Basic validation
            valid = True
            error_msg = ""

            # Check required attributes
            required_attrs = ['x', 'edge_index', 'is_smelly']
            for attr in required_attrs:
                if not hasattr(data, attr) or getattr(data, attr) is None:
                    valid = False
                    error_msg = f"Missing {attr}"
                    break

            if valid:
                # Check tensor properties
                if data.x.size(1) != len(HUB_FEATURES):
                    valid = False
                    error_msg = f"Wrong feature count: {data.x.size(1)} != {len(HUB_FEATURES)}"
                elif torch.isnan(data.x).any() or torch.isinf(data.x).any():
                    valid = False
                    error_msg = "NaN or Inf values in features"

            if valid:
                stats['valid_files'] += 1

                # Collect statistics
                shape_key = str(list(data.x.shape))
                dtype_key = str(data.x.dtype)
                stats['feature_stats']['shapes'][shape_key] = stats['feature_stats']['shapes'].get(shape_key, 0) + 1
                stats['feature_stats']['dtypes'][dtype_key] = stats['feature_stats']['dtypes'].get(dtype_key, 0) + 1

                label = 'smelly' if data.is_smelly.item() else 'clean'
                stats['label_distribution'][label] += 1
            else:
                stats['invalid_files'] += 1
                stats['validation_errors'].append(f"{pt_file.name}: {error_msg}")

        except Exception as e:
            stats['invalid_files'] += 1
            stats['validation_errors'].append(f"{pt_file.name}: Load error - {e}")

    return stats


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Clean dataset creation pipeline")
    parser.add_argument("--config", type=Path, required=True, help="Configuration YAML file")
    parser.add_argument("--projects", nargs="+", help="Specific projects to process")
    parser.add_argument("--validate", action="store_true", help="Validate generated dataset")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    try:
        # Load configuration
        config_data = yaml.safe_load(args.config.read_text())

        # Setup paths
        arcan_dir = Path(config_data['paths']['arcan_output'])
        output_dir = Path(config_data['paths']['dataset_output'])

        # Get projects list
        projects = args.projects or config_data.get('projects', [])
        if not projects:
            raise ValueError("No projects specified")

        # Setup extraction config
        extraction_config = config_data.get('extraction', {})

        # Build dataset
        builder = DatasetBuilder(arcan_dir, output_dir, extraction_config)
        pipeline_stats = builder.build_dataset(projects)

        # Validate if requested
        if args.validate:
            logger.info("Validating generated dataset...")
            validation_stats = validate_dataset(output_dir)

            print(f"\n📊 Dataset Validation Results:")
            print(f"Total files: {validation_stats['total_files']}")
            print(f"Valid files: {validation_stats['valid_files']}")
            print(f"Invalid files: {validation_stats['invalid_files']}")
            print(f"Label distribution: {validation_stats['label_distribution']}")

            if validation_stats['validation_errors']:
                print(f"\nValidation errors:")
                for error in validation_stats['validation_errors'][:10]:  # Show first 10
                    print(f"  - {error}")

        print(f"\n✅ Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        exit(1)